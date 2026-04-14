import json
import logging
import os
import re
from pathlib import Path

from openai import OpenAI
from rank_bm25 import BM25Okapi

from summarizer.llm import SummaryResult, _call_api

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL   = "gpt-5.2"

PROMPTS_DIR           = Path(__file__).parent / "prompts"
QA_PROMPT_PATH        = PROMPTS_DIR / "qa_v1.md"
FOLLOW_UP_PROMPT_PATH = PROMPTS_DIR / "follow_up_v1.md"

MAX_SUMMARY_SECTIONS   = 3
MAX_RAW_CHUNKS         = 5   # 하이브리드 결과 반영 위해 3→5
CHUNK_BM25_MIN_SCORE   = 0.1
SECTION_BM25_MIN_SCORE = 0.1
RRF_K                  = 60   # RRF 상수

# 추천 질문 생성 컨텍스트에서 제거할 OCR 노이즈 패턴
_OCR_NOISE_RE = re.compile(
    r"\[OCR [^\]]*\]"
    r"|\[OCR\]"
    r"|※ 이미지 기반 PDF[^\n]*"
    r"|원문:\s*[^\]]*"
)

_UNANSWERABLE_PATTERNS = [
    "문서에서 해당 내용을 찾을 수 없",
    "문서에서 확인할 수 없",
    "제공된 문서에는 해당 내용이 없",
    "해당 정보는 문서에 포함되어 있지 않",
]

_QA_SYSTEM_PROMPT = (
    "당신은 금융 문서 내용을 기반으로 질문에 답변하는 전문 AI입니다. "
    "제공된 문서 내용에만 근거하여 답변하세요. "
    "원문 발췌 > 요약 섹션 > 전체 요약 순으로 우선 참조하세요. "
    "전체 요약에 답이 있으면 반드시 그 내용을 바탕으로 답변하세요. "
    "문서에 없는 내용은 절대 생성하지 마세요."
)


def _is_unanswerable(text: str) -> bool:
    return any(p in text for p in _UNANSWERABLE_PATTERNS)


def _strip_ocr_noise(text: str) -> str:
    """추천 질문 생성 전 OCR 노이즈 태그 제거."""
    return _OCR_NOISE_RE.sub("", text).strip()


def _call_qa(user_prompt: str) -> str:
    return _call_api(
        messages=[
            {"role": "system", "content": _QA_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=500,
    )


class SourceItem:
    def __init__(self, section: str, snippet: str):
        self.section = section
        self.snippet = snippet


class QAResult:
    def __init__(self, answer: str, sources: list[SourceItem], is_answerable: bool):
        self.answer        = answer
        self.sources       = sources
        self.is_answerable = is_answerable


class FollowUp:
    def __init__(self, question: str, section_indices: list[int]):
        self.question        = question
        self.section_indices = section_indices


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[가-힣a-zA-Z0-9]+", text)


def _fix_tilde(text: str) -> str:
    return re.sub(r'(\d+\.?\d*)~+(\d+\.?\d*)', r'\1-\2', text)


def _best_bullet_by_answer(answer: str, bullets: list[str]) -> str:
    if not bullets:
        return ""
    if len(bullets) == 1:
        return bullets[0]
    answer_tokens = set(_tokenize(answer))
    if not answer_tokens:
        return bullets[0]
    scored = [(len(answer_tokens & set(_tokenize(b))), b) for b in bullets]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored[0][0] > 0 else bullets[0]


def _rrf_fusion(
    bm25_chunks: list[str],
    dense_chunks: list[str],
    k: int = RRF_K,
    top_n: int = MAX_RAW_CHUNKS,
) -> list[str]:
    """
    Reciprocal Rank Fusion:
    score(d) = 1/(k + rank_bm25) + 1/(k + rank_dense)

    두 결과를 퓨전해 최종 Top-N 청크 반환.
    한쪽 결과에만 있는 청크는 다른 쪽 rank를 len+1로 처리.
    """
    all_chunks = list(dict.fromkeys(bm25_chunks + dense_chunks))  # 순서 유지 중복 제거
    scores: dict[str, float] = {c: 0.0 for c in all_chunks}

    for rank, chunk in enumerate(bm25_chunks, start=1):
        scores[chunk] += 1.0 / (k + rank)
    for rank, chunk in enumerate(dense_chunks, start=1):
        scores[chunk] += 1.0 / (k + rank)

    ranked = sorted(all_chunks, key=lambda c: scores[c], reverse=True)
    return ranked[:top_n]


def _find_relevant_chunks_hybrid(
    question: str,
    raw_chunks: list[str],
    doc_id: str | None = None,
) -> list[str]:
    """
    하이브리드 검색:
      1) BM25 sparse 검색
      2) Dense 검색 (Qdrant, doc_id 있을 때만)
      3) RRF 퓨전

    doc_id 없거나 Qdrant 미연결 시 BM25 단독으로 fallback.
    """
    # BM25
    bm25_chunks = _find_relevant_chunks_bm25(question, raw_chunks)

    # Dense (선택적)
    dense_chunks: list[str] = []
    if doc_id:
        try:
            from summarizer.embedder import search_chunks
            dense_chunks = search_chunks(question, doc_id, top_k=MAX_RAW_CHUNKS)
        except Exception as e:
            logger.warning("Dense 검색 실패 — BM25 단독 사용: %s", e)

    if not dense_chunks:
        return bm25_chunks

    fused = _rrf_fusion(bm25_chunks, dense_chunks)
    logger.info("RRF 퓨전 완료 — bm25:%d dense:%d → fused:%d",
                len(bm25_chunks), len(dense_chunks), len(fused))
    return fused


def _find_relevant_sections_bm25(question: str, summary: SummaryResult) -> list:
    if not summary.sections:
        return []
    section_texts = [sec.section + " " + " ".join(sec.bullets) for sec in summary.sections]
    tokenized_sections = [_tokenize(t) for t in section_texts]
    tokenized_query    = _tokenize(question)
    valid = [(sec, tok) for sec, tok in zip(summary.sections, tokenized_sections) if tok]
    if not valid:
        return []
    sections_valid, tokenized_valid = zip(*valid)
    bm25   = BM25Okapi(list(tokenized_valid))
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(scores, sections_valid), key=lambda x: x[0], reverse=True)
    return [sec for score, sec in ranked[:MAX_SUMMARY_SECTIONS] if score >= SECTION_BM25_MIN_SCORE]


def _find_relevant_chunks_bm25(question: str, raw_chunks: list[str]) -> list[str]:
    if not raw_chunks:
        return []
    tokenized_chunks = [_tokenize(c) for c in raw_chunks]
    tokenized_query  = _tokenize(question)
    valid = [(chunk, tok) for chunk, tok in zip(raw_chunks, tokenized_chunks) if tok]
    if not valid:
        return []
    chunks_valid, tokenized_valid = zip(*valid)
    bm25   = BM25Okapi(list(tokenized_valid))
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(scores, chunks_valid), key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in ranked[:MAX_RAW_CHUNKS] if score >= CHUNK_BM25_MIN_SCORE]


def _build_context(
    relevant_sections: list,
    relevant_chunks: list[str],
    overall: str = "",
    force_overall: bool = False,
) -> str:
    parts = []
    if not force_overall:
        if relevant_chunks:
            parts.append("## 원문 발췌")
            for i, chunk in enumerate(relevant_chunks, 1):
                parts.append(f"[원문 {i}]\n{chunk}")
        if relevant_sections:
            parts.append("## 요약 섹션")
            for sec in relevant_sections:
                parts.append(f"[{sec.section}]\n" + "\n".join(f"- {b}" for b in sec.bullets))
    if (force_overall or (not relevant_sections and not relevant_chunks)) and overall:
        parts.append("## 전체 요약")
        parts.append(overall[:3000])
    return "\n\n".join(parts)


def _make_sources(answer: str, relevant_sections: list) -> list[SourceItem]:
    seen: set[str] = set()
    result = []
    for sec in relevant_sections:
        key = sec.section.strip()
        if key in seen:
            continue
        seen.add(key)
        snippet = _fix_tilde(_best_bullet_by_answer(answer, sec.bullets))
        result.append(SourceItem(section=key, snippet=snippet))
    return result


def ask(
    question: str,
    summary: SummaryResult,
    raw_chunks: list[str] | None = None,
    pinned_section_indices: list[int] | None = None,
    doc_id: str | None = None,
) -> QAResult:
    """
    Q&A 메인 함수.

    doc_id 제공 시 BM25 + Dense(Qdrant) 하이브리드 검색 + RRF 퓨전 사용.
    doc_id 없으면 기존 BM25 단독 동작 (하위 호환).
    """
    logger.info("Q&A 시작 — 질문: %r (doc_id=%s)", question[:50], doc_id)

    if pinned_section_indices is not None:
        relevant_sections = [
            summary.sections[i]
            for i in pinned_section_indices
            if 0 <= i < len(summary.sections)
        ]
    else:
        relevant_sections = _find_relevant_sections_bm25(question, summary)

    # 하이브리드 청크 검색 (doc_id 있으면 RRF, 없으면 BM25)
    relevant_chunks = _find_relevant_chunks_hybrid(question, raw_chunks or [], doc_id=doc_id)

    template = QA_PROMPT_PATH.read_text(encoding="utf-8")

    try:
        if relevant_sections or relevant_chunks:
            context     = _build_context(relevant_sections, relevant_chunks)
            user_prompt = template.format(question=question, context=context)
            raw = _call_qa(user_prompt)

            if not _is_unanswerable(raw):
                sources = _make_sources(raw, relevant_sections)
                logger.info("Q&A 완료 — 출처 %d개", len(sources))
                return QAResult(answer=raw, sources=sources, is_answerable=True)

            logger.info("하이브리드 검색 답변 불가 — overall fallback 시도")

        if not summary.overall:
            logger.warning("overall 없음 — 답변 불가")
            return QAResult(answer="문서에서 해당 내용을 찾을 수 없습니다.", sources=[], is_answerable=False)

        logger.info("overall fallback 사용")
        context     = _build_context([], [], overall=summary.overall, force_overall=True)
        user_prompt = template.format(question=question, context=context)
        raw = _call_qa(user_prompt)

        if _is_unanswerable(raw):
            return QAResult(answer=raw, sources=[], is_answerable=False)

        logger.info("Q&A 완료 (overall fallback)")
        return QAResult(answer=raw, sources=[], is_answerable=True)

    except Exception as e:
        logger.error("Q&A 생성 실패: %s", e)
        return QAResult(answer="[답변 생성 실패]", sources=[], is_answerable=False)


def generate_follow_ups(summary: SummaryResult) -> list[FollowUp]:
    n = len(summary.sections)
    all_idx = list(range(n)) if n else [0]
    defaults = [
        FollowUp("재무지표 더 자세히 보여줘", all_idx),
        FollowUp("리스크 요인은 무엇인가요?", all_idx),
        FollowUp("향후 전망은?", all_idx),
    ]
    if not summary.overall:
        return defaults

    overall_context = _strip_ocr_noise(summary.overall)
    if not overall_context.strip():
        return defaults

    try:
        template    = FOLLOW_UP_PROMPT_PATH.read_text(encoding="utf-8")
        user_prompt = template.format(overall=overall_context)

        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": (
                    "당신은 금융 문서 분석 전문가입니다. "
                    "제공된 문서 요약에 근거해 독자가 실제로 물어볼 만한 구체적인 질문을 생성하세요."
                )},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        logger.info("추천 질문 LLM 응답 raw: %r", raw[:300])

        data  = json.loads(raw)
        items = data.get("questions", [])
        logger.info("추천 질문 items type=%s value=%r", type(items).__name__, items)

        result = []
        for item in items[:3]:
            if isinstance(item, dict):
                q   = item.get("q", "")
                idx = item.get("section_index", 0)
            else:
                q   = str(item)
                idx = 0
            if q:
                result.append(FollowUp(question=q, section_indices=[idx]))

        if len(result) >= 3:
            logger.info("추천 질문 생성 완료 — %d개", len(result))
            return result
        else:
            logger.warning("추천 질문 3개 미만 생성 (%d개) — fallback", len(result))

    except Exception as e:
        logger.warning("추천 질문 생성 실패: %s", e)

    return defaults
