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

MAX_SUMMARY_SECTIONS = 3
MAX_RAW_CHUNKS       = 3
CHUNK_BM25_MIN_SCORE = 0.1

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


def _is_unanswerable(text: str) -> bool:
    return any(p in text for p in _UNANSWERABLE_PATTERNS)


def _strip_ocr_noise(text: str) -> str:
    """추천 질문 생성 전 OCR 노이즈 태그 제거."""
    return _OCR_NOISE_RE.sub("", text).strip()


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
    return [sec for _, sec in ranked[:MAX_SUMMARY_SECTIONS]]


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
) -> str:
    parts = []
    if relevant_chunks:
        parts.append("## 원문 발췌")
        for i, chunk in enumerate(relevant_chunks, 1):
            parts.append(f"[원문 {i}]\n{chunk}")
    if relevant_sections:
        parts.append("## 요약 섹션")
        for sec in relevant_sections:
            parts.append(f"[{sec.section}]\n" + "\n".join(f"- {b}" for b in sec.bullets))
    # BM25 매칭이 없을 때 전체 요약으로 fallback
    if not relevant_sections and not relevant_chunks and overall:
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
) -> QAResult:
    logger.info("Q&A 시작 — 질문: %r", question[:50])

    if pinned_section_indices is not None:
        relevant_sections = [
            summary.sections[i]
            for i in pinned_section_indices
            if 0 <= i < len(summary.sections)
        ]
    else:
        relevant_sections = _find_relevant_sections_bm25(question, summary)

    relevant_chunks = _find_relevant_chunks_bm25(question, raw_chunks or [])

    # BM25 매칭 없고 overall도 없으면 답변 불가
    if not relevant_sections and not relevant_chunks and not summary.overall:
        logger.warning("관련 섹션/청크 없음 — 답변 불가")
        return QAResult(answer="문서에서 해당 내용을 찾을 수 없습니다.", sources=[], is_answerable=False)

    context = _build_context(relevant_sections, relevant_chunks, overall=summary.overall)
    if not relevant_sections and not relevant_chunks:
        logger.info("BM25 매칭 없음 — overall fallback 사용")

    template    = QA_PROMPT_PATH.read_text(encoding="utf-8")
    user_prompt = template.format(question=question, context=context)

    try:
        raw = _call_api(
            messages=[
                {"role": "system", "content": (
                    "당신은 금융 문서 내용을 기반으로 질문에 답변하는 전문 AI입니다. "
                    "제공된 문서 내용에만 근거하여 답변하세요. "
                    "원문 발췌가 있으면 요약 섹션보다 원문 발췌를 우선 참조하세요. "
                    "문서에 없는 내용은 절대 생성하지 마세요."
                )},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
        )
        if _is_unanswerable(raw):
            return QAResult(answer=raw, sources=[], is_answerable=False)
        sources = _make_sources(raw, relevant_sections)
        logger.info("Q&A 완료 — 출처 %d개", len(sources))
        return QAResult(answer=raw, sources=sources, is_answerable=True)
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

    # overall(전체 요약)을 컨텍스트로 사용 — OCR 노이즈 적음
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
