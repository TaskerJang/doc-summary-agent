import json
import logging
from pathlib import Path

from summarizer.llm import SummaryResult, _call_api

logger = logging.getLogger(__name__)

PROMPTS_DIR           = Path(__file__).parent / "prompts"
QA_PROMPT_PATH        = PROMPTS_DIR / "qa_v1.md"
FOLLOW_UP_PROMPT_PATH = PROMPTS_DIR / "follow_up_v1.md"

# QA context에 포함할 최대 섹션/청크 수
MAX_SUMMARY_SECTIONS = 3
MAX_RAW_CHUNKS       = 3


class QAResult:
    def __init__(self, answer: str, sources: list[str], is_answerable: bool):
        self.answer        = answer
        self.sources       = sources
        self.is_answerable = is_answerable


def _find_relevant_sections(question: str, summary: SummaryResult) -> list:
    keywords = set(question.replace("?", "").replace(".", "").split())
    scored = []
    for sec in summary.sections:
        text  = sec.section + " " + " ".join(sec.bullets)
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scored.append((score, sec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [sec for _, sec in scored[:MAX_SUMMARY_SECTIONS]]


def _find_relevant_chunks(question: str, raw_chunks: list[str]) -> list[str]:
    """
    원문 청크 중 질문 키워드와 겹치는 상위 MAX_RAW_CHUNKS개 반환.
    키워드 매칭 점수 0이면 전체에서 첫 MAX_RAW_CHUNKS개 fallback.
    """
    if not raw_chunks:
        return []
    keywords = set(question.replace("?", "").replace(".", "").split())
    scored = [(sum(1 for kw in keywords if kw in chunk), chunk) for chunk in raw_chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [chunk for score, chunk in scored[:MAX_RAW_CHUNKS] if score > 0]
    return top if top else raw_chunks[:MAX_RAW_CHUNKS]


def _build_context(relevant_sections: list, relevant_chunks: list[str]) -> str:
    """
    [요약 섹션] + [원문 청크] 를 합쳐 context 문자열 구성.
    원문 청크를 앞에 배치해 LLM이 원문을 우선 참조하도록 유도.
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


def ask(
    question: str,
    summary: SummaryResult,
    raw_chunks: list[str] | None = None,
) -> QAResult:
    """
    질문에 대한 답변 생성.

    Args:
        question: 질문 문자열
        summary:  LLM 요약 결과 (SummaryResult)
        raw_chunks: 원문 청크 텍스트 리스트 (Completeness 개선용)
                    None이면 요약 섹션만 context로 사용 (기존 동작 유지)
    """
    logger.info("Q&A 시작 — 질문: %r", question[:50])

    relevant_sections = _find_relevant_sections(question, summary)
    relevant_chunks   = _find_relevant_chunks(question, raw_chunks or [])

    if not relevant_sections and not relevant_chunks:
        logger.warning("관련 섹션/청크 없음 — 답변 불가")
        return QAResult(
            answer="문서에서 해당 질문에 대한 내용을 찾을 수 없습니다.",
            sources=[],
            is_answerable=False,
        )

    context = _build_context(relevant_sections, relevant_chunks)
    sources = [sec.section for sec in relevant_sections]

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
        logger.info("Q&A 완료 — 출처 섹션 %d개", len(sources))
        return QAResult(answer=raw, sources=sources, is_answerable=True)

    except Exception as e:
        logger.error("Q&A 생성 실패: %s", e)
        return QAResult(
            answer="[답변 생성 실패]",
            sources=sources,
            is_answerable=False,
        )


def generate_follow_ups(summary: SummaryResult) -> list[str]:
    """
    전체 요약을 바탕으로 자연스러운 추천 질문 3개를 생성한다.
    """
    defaults = ["재무지표 더 자세히 보여줘", "리스크 요인은 무엇인가요?", "향후 전망은?"]

    if not summary.overall or summary.overall.startswith("["):
        return defaults

    try:
        template    = FOLLOW_UP_PROMPT_PATH.read_text(encoding="utf-8")
        user_prompt = template.format(overall=summary.overall)

        raw = _call_api(
            messages=[
                {"role": "system", "content": "당신은 금융 문서 분석 전문가입니다."},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=200,
        )

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data      = json.loads(raw)
        questions = data.get("questions", [])

        if len(questions) >= 3:
            return questions[:3]

    except Exception as e:
        logger.warning("추천 질문 생성 실패: %s", e)

    return defaults
