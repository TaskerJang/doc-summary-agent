import json
import logging
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from summarizer.llm import SummaryResult, _call_api

logger = logging.getLogger(__name__)

PROMPTS_DIR           = Path(__file__).parent / "prompts"
QA_PROMPT_PATH        = PROMPTS_DIR / "qa_v1.md"
FOLLOW_UP_PROMPT_PATH = PROMPTS_DIR / "follow_up_v1.md"

# QA context에 포함할 최대 섹션/청크 수
MAX_SUMMARY_SECTIONS = 3
MAX_RAW_CHUNKS       = 3

# 원문 청크 BM25 임계값 (섹션 검색에는 미적용)
CHUNK_BM25_MIN_SCORE = 0.1

# LLM이 "찾을 수 없음"으로 답했다고 판단하는 패턴
_UNANSWERABLE_PATTERNS = [
    "찾을 수 없",
    "해당 내용을 찾",
    "문서에서 확인할 수 없",
    "문서에 없",
    "제공된 문서에는",
]


def _is_unanswerable(text: str) -> bool:
    """LLM 답변이 '찾을 수 없음' 패턴인지 확인."""
    return any(p in text for p in _UNANSWERABLE_PATTERNS)


class SourceItem:
    """출처 섹션명 + 핵심 내용 한 줄."""
    def __init__(self, section: str, snippet: str):
        self.section = section
        self.snippet = snippet


class QAResult:
    def __init__(self, answer: str, sources: list[SourceItem], is_answerable: bool):
        self.answer        = answer
        self.sources       = sources
        self.is_answerable = is_answerable


class FollowUp:
    """추천 질문 + 출처 섹션 인덱스."""
    def __init__(self, question: str, section_indices: list[int]):
        self.question        = question
        self.section_indices = section_indices


def _tokenize(text: str) -> list[str]:
    """간단한 공백/구두점 기반 토크나이저 (한국어 포함)."""
    return re.findall(r"[가-힣a-zA-Z0-9]+", text)


def _fix_tilde(text: str) -> str:
    """숫자 범위의 ~ 를 - 로 치환해 Chainlit 취소선 렌더링 방지."""
    return re.sub(r'(\d+\.?\d*)~+(\d+\.?\d*)', r'\1-\2', text)


def _best_bullet_by_answer(answer: str, bullets: list[str]) -> str:
    """
    LLM 답변 본문과 가장 토큰 겹침이 많은 bullet을 snippet으로 선택.
    겹치는 토큰이 없으면 첫 번째 bullet 반환.
    """
    if not bullets:
        return ""
    if len(bullets) == 1:
        return bullets[0]

    answer_tokens = set(_tokenize(answer))
    if not answer_tokens:
        return bullets[0]

    scored = []
    for b in bullets:
        b_tokens = set(_tokenize(b))
        overlap  = len(answer_tokens & b_tokens)
        scored.append((overlap, b))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_bullet = scored[0]

    if best_score == 0:
        return bullets[0]
    return best_bullet


def _find_relevant_sections_bm25(question: str, summary: SummaryResult) -> list:
    """
    BM25Okapi로 요약 섹션 중 질문과 관련성 높은 상위 MAX_SUMMARY_SECTIONS개 반환.
    임계값 없이 항상 상위 섹션을 반환 — 무관 질문 판별은 LLM에 위임.
    """
    if not summary.sections:
        return []

    section_texts = [
        sec.section + " " + " ".join(sec.bullets)
        for sec in summary.sections
    ]
    tokenized_sections = [_tokenize(t) for t in section_texts]
    tokenized_query    = _tokenize(question)

    valid = [
        (sec, tok) for sec, tok in zip(summary.sections, tokenized_sections) if tok
    ]
    if not valid:
        return []

    sections_valid, tokenized_valid = zip(*valid)
    bm25   = BM25Okapi(list(tokenized_valid))
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(scores, sections_valid), key=lambda x: x[0], reverse=True)
    result = [sec for _, sec in ranked[:MAX_SUMMARY_SECTIONS]]

    logger.debug(
        "섹션 BM25 선택 %d개 (top=%.3f, section=%r)",
        len(result), ranked[0][0] if ranked else 0,
        ranked[0][1].section if ranked else "",
    )
    return result


def _find_relevant_chunks_bm25(question: str, raw_chunks: list[str]) -> list[str]:
    """
    BM25Okapi로 질문과 관련성 높은 원문 청크 상위 MAX_RAW_CHUNKS개 반환.
    CHUNK_BM25_MIN_SCORE 미만인 청크는 노이즈로 판단해 제외.
    """
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
    result = [chunk for score, chunk in ranked[:MAX_RAW_CHUNKS] if score >= CHUNK_BM25_MIN_SCORE]

    if result:
        logger.debug("BM25 선택 청크 %d개 (top score=%.3f)", len(result), ranked[0][0])
    else:
        logger.debug("BM25 임계값 미달 — 원문 청크 미사용 (top score=%.3f)", ranked[0][0] if ranked else 0)

    return result


def _build_context(relevant_sections: list, relevant_chunks: list[str]) -> str:
    """
    [원문 발췌] + [요약 섹션] 순서로 context 구성.
    원문을 앞에 배치해 LLM이 요약보다 원문을 우선 참조하도록 유도.
    """
    parts = []

    if relevant_chunks:
        parts.append("## 원문 발췌")
        for i, chunk in enumerate(relevant_chunks, 1):
            parts.append(f"[원문 {i}]\n{chunk}")

    if relevant_sections:
        parts.append("## 요약 섹션")
        for sec in relevant_sections:
            section_text = f"[{sec.section}]\n" + "\n".join(f"- {b}" for b in sec.bullets)
            parts.append(section_text)

    return "\n\n".join(parts)


def _make_sources(answer: str, relevant_sections: list) -> list[SourceItem]:
    """
    섹션명 기준 dedup 후 LLM 답변과 토큰 겹침이 가장 많은 bullet을 snippet으로 선택.
    ~ 취소선 방지 처리 적용.
    """
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
    """
    질문에 대한 답변 생성.

    Args:
        question:               질문 문자열
        summary:                LLM 요약 결과 (SummaryResult)
        raw_chunks:             원문 청크 텍스트 리스트 (BM25 검색용)
        pinned_section_indices: 추천 질문 클릭 시 출처 섹션 인덱스 (BM25 스킵)
                                None이면 BM25로 섹션 검색
    """
    logger.info("Q&A 시작 — 질문: %r", question[:50])

    # 추천 질문: 생성 시점에 고정된 섹션 인덱스 사용 → BM25 재검색 불필요
    if pinned_section_indices is not None:
        relevant_sections = [
            summary.sections[i]
            for i in pinned_section_indices
            if 0 <= i < len(summary.sections)
        ]
        logger.debug("pinned 섹션 사용 %d개", len(relevant_sections))
    else:
        relevant_sections = _find_relevant_sections_bm25(question, summary)

    relevant_chunks = _find_relevant_chunks_bm25(question, raw_chunks or [])

    if not relevant_sections and not relevant_chunks:
        logger.warning("관련 섹션/청크 없음 — 답변 불가")
        return QAResult(
            answer="문서에서 해당 내용을 찾을 수 없습니다.",
            sources=[],
            is_answerable=False,
        )

    context = _build_context(relevant_sections, relevant_chunks)

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

        # LLM이 "찾을 수 없음"으로 답한 경우 출처 없이 처리
        if _is_unanswerable(raw):
            logger.info("LLM 답변 불가 패턴 감지 — is_answerable=False 처리")
            return QAResult(answer=raw, sources=[], is_answerable=False)

        sources = _make_sources(raw, relevant_sections)

        logger.info("Q&A 완료 — 출처 섹션 %d개, 원문 청크 %d개", len(sources), len(relevant_chunks))
        return QAResult(answer=raw, sources=sources, is_answerable=True)

    except Exception as e:
        logger.error("Q&A 생성 실패: %s", e)
        return QAResult(
            answer="[답변 생성 실패]",
            sources=[],
            is_answerable=False,
        )


def generate_follow_ups(summary: SummaryResult) -> list[FollowUp]:
    """
    섹션 bullets 기반으로 추천 질문 3개를 생성한다.
    각 질문에 출처 섹션 인덱스(section_index)를 포함해 반환한다.
    """
    defaults = [
        FollowUp("재무지표 더 자세히 보여줘", [0]),
        FollowUp("리스크 요인은 무엇인가요?", [0]),
        FollowUp("향후 전망은?", [0]),
    ]

    if not summary.sections:
        return defaults

    # 섹션 bullets 기반 context 구성 (최대 4섹션 × 3 bullets)
    section_lines = []
    for i, sec in enumerate(summary.sections[:4]):
        bullets_text = "\n".join(f"- {b}" for b in sec.bullets[:3])
        section_lines.append(f"[섹션 {i}] {sec.section}\n{bullets_text}")
    sections_context = "\n\n".join(section_lines)

    try:
        template    = FOLLOW_UP_PROMPT_PATH.read_text(encoding="utf-8")
        user_prompt = template.format(overall=sections_context)

        raw = _call_api(
            messages=[
                {"role": "system", "content": (
                    "당신은 금융 문서 분석 전문가입니다. "
                    "제공된 섹션 내용에 근거해 독자가 실제로 물어볼 만한 구체적인 질문을 생성하세요."
                )},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=300,
        )

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data  = json.loads(raw)
        items = data.get("questions", [])

        result = []
        for item in items[:3]:
            if isinstance(item, dict):
                q   = item.get("q", "")
                idx = item.get("section_index", 0)
            else:
                # 구버전 fallback: 문자열 리스트
                q   = str(item)
                idx = 0
            if q:
                result.append(FollowUp(question=q, section_indices=[idx]))

        if len(result) >= 3:
            return result

    except Exception as e:
        logger.warning("추천 질문 생성 실패: %s", e)

    return defaults
