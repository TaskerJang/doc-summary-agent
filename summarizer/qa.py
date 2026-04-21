import asyncio
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
MAX_RAW_CHUNKS         = 5    # RRF 퓨전 후보
RERANK_TOP_N           = 3    # reranker 최종 반환 수
CHUNK_BM25_MIN_SCORE   = 0.1
SECTION_BM25_MIN_SCORE = 0.1
RRF_K                  = 60

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

# ── reranker 싱글턴 ─────────────────────────────────────────────────────────
_reranker = None

async def _aget_reranker():
    """
    bge-reranker-v2-m3 싱글턴 (async).
    cross-encoder 방식으로 (질문, 청크) 쌍의 관련도를 0~1 스코어로 반환.
    미설치 또는 로드 실패 시 None 반환 → reranking 스킵.

    첫 호출 시 CrossEncoder 초기화가 모델 파일 로드(디스크 I/O + CPU)로
    수 초~수십 초 블로킹되므로 asyncio.to_thread로 감싸 이벤트 루프를
    살려둔다. (#107)
    """
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("bge-reranker-v2-m3 로딩 중...")
            _reranker = await asyncio.to_thread(
                CrossEncoder, "BAAI/bge-reranker-v2-m3", max_length=512
            )
            logger.info("bge-reranker-v2-m3 로딩 완료")
        except Exception as e:
            logger.warning("reranker 로드 실패 — reranking 스킵: %s", e)
            _reranker = False  # 재시도 방지용 sentinel
    return _reranker if _reranker else None


async def _rerank(question: str, chunks: list[str], top_n: int = RERANK_TOP_N) -> list[str]:
    """
    bge-reranker-v2-m3로 (질문, 청크) 쌍 관련도 스코어링 후 top_n 반환.
    reranker 없거나 청크 수가 top_n 이하면 그대로 반환.

    reranker.predict는 CPU에서 cross-encoder 추론으로 첫 호출 ~1분 걸리므로
    asyncio.to_thread로 분리. 이벤트 루프가 살아있어야 Chainlit websocket
    ping/pong이 유지되어 세션이 끊기지 않는다. (#107)
    """
    if len(chunks) <= top_n:
        return chunks
    reranker = await _aget_reranker()
    if reranker is None:
        return chunks[:top_n]
    try:
        pairs  = [(question, c) for c in chunks]
        scores = await asyncio.to_thread(reranker.predict, pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        result = [c for _, c in ranked[:top_n]]
        logger.info("Reranking 완료 — %d → %d청크", len(chunks), len(result))
        return result
    except Exception as e:
        logger.warning("Reranking 실패 — 원본 순서 유지: %s", e)
        return chunks[:top_n]


def _is_unanswerable(text: str) -> bool:
    return any(p in text for p in _UNANSWERABLE_PATTERNS)


def _strip_ocr_noise(text: str) -> str:
    return _OCR_NOISE_RE.sub("", text).strip()


async def _call_qa(user_prompt: str) -> str:
    """_call_api는 async def이므로 반드시 await 호출."""
    return await _call_api(
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
    """
    all_chunks = list(dict.fromkeys(bm25_chunks + dense_chunks))
    scores: dict[str, float] = {c: 0.0 for c in all_chunks}

    for rank, chunk in enumerate(bm25_chunks, start=1):
        scores[chunk] += 1.0 / (k + rank)
    for rank, chunk in enumerate(dense_chunks, start=1):
        scores[chunk] += 1.0 / (k + rank)

    ranked = sorted(all_chunks, key=lambda c: scores[c], reverse=True)
    return ranked[:top_n]


async def _find_relevant_chunks_hybrid(
    question: str,
    raw_chunks: list[str],
    doc_id: str | None = None,
) -> list[str]:
    """
    하이브리드 검색 파이프라인:
      1) BM25 sparse 검색        → Top-5 청크
      2) Dense 검색 (Qdrant)     → Top-5 청크
      3) RRF 퓨전                → Top-5 청크
      4) bge-reranker-v2-m3     → Top-3 청크 (최종 LLM 컨텍스트)

    doc_id 없거나 Qdrant 미연결 시 BM25 → reranker fallback.
    reranker 미설치 시 RRF 결과 그대로 반환.
    """
    bm25_chunks = _find_relevant_chunks_bm25(question, raw_chunks)

    dense_chunks: list[str] = []
    if doc_id:
        try:
            from summarizer.embedder import search_chunks
            dense_chunks = search_chunks(question, doc_id, top_k=MAX_RAW_CHUNKS)
        except Exception as e:
            logger.warning("Dense 검색 실패 — BM25 단독 사용: %s", e)

    if dense_chunks:
        fused = _rrf_fusion(bm25_chunks, dense_chunks)
        logger.info("RRF 퓨전 완료 — bm25:%d dense:%d → fused:%d",
                    len(bm25_chunks), len(dense_chunks), len(fused))
    else:
        fused = bm25_chunks

    # reranker로 최종 정밀도 향상
    return await _rerank(question, fused, top_n=RERANK_TOP_N)


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


async def ask(
    question: str,
    summary: SummaryResult,
    raw_chunks: list[str] | None = None,
    pinned_section_indices: list[int] | None = None,
    doc_id: str | None = None,
) -> QAResult:
    """
    Q&A 메인 함수 (async).

    _call_api가 async def이므로 ask()도 async로 전환. (#68 AsyncOpenAI 이관 후속)
    검색 파이프라인: BM25 + Dense(Qdrant) → RRF → bge-reranker-v2-m3 → LLM
    doc_id 없으면 BM25 단독 (하위 호환).
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

    relevant_chunks = await _find_relevant_chunks_hybrid(question, raw_chunks or [], doc_id=doc_id)

    template = QA_PROMPT_PATH.read_text(encoding="utf-8")

    try:
        if relevant_sections or relevant_chunks:
            context     = _build_context(relevant_sections, relevant_chunks)
            user_prompt = template.format(question=question, context=context)
            raw = await _call_qa(user_prompt)

            if not _is_unanswerable(raw):
                sources = _make_sources(raw, relevant_sections)
                logger.info("Q&A 완료 — 출처 %d개", len(sources))
                return QAResult(answer=raw, sources=sources, is_answerable=True)

            logger.info("검색 결과 답변 불가 — overall fallback 시도")

        if not summary.overall:
            logger.warning("overall 없음 — 답변 불가")
            return QAResult(answer="문서에서 해당 내용을 찾을 수 없습니다.", sources=[], is_answerable=False)

        logger.info("overall fallback 사용")
        context     = _build_context([], [], overall=summary.overall, force_overall=True)
        user_prompt = template.format(question=question, context=context)
        raw = await _call_qa(user_prompt)

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
