import logging
from pathlib import Path

from summarizer.llm import SummaryResult, _call_api

logger = logging.getLogger(__name__)

PROMPTS_DIR    = Path(__file__).parent / "prompts"
QA_PROMPT_PATH = PROMPTS_DIR / "qa_v1.md"


class QAResult:
    def __init__(self, answer: str, sources: list[str], is_answerable: bool):
        self.answer       = answer
        self.sources      = sources   # 근거 섹션명 목록
        self.is_answerable = is_answerable


def _find_relevant_sections(question: str, summary: SummaryResult) -> list:
    """질문 키워드와 겹치는 섹션을 필터링한다."""
    keywords = set(question.replace("?", "").replace(".", "").split())

    scored = []
    for sec in summary.sections:
        text = sec.section + " " + " ".join(sec.bullets)
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scored.append((score, sec))

    # 가장 관련성 높은 섹션 최대 3개를 반환하는 Vector DB 없는 경량 검색 함수
    scored.sort(key=lambda x: x[0], reverse=True)
    return [sec for _, sec in scored[:3]]


def ask(question: str, summary: SummaryResult) -> QAResult:
    """
    질문과 요약 결과를 받아 관련 섹션 기반으로 답변을 생성한다.

    Args:
        question: 사용자 질문
        summary:  summarize() 결과 SummaryResult

    Returns:
        QAResult — answer / sources / is_answerable 필드 포함
    """
    logger.info("Q&A 시작 — 질문: %r", question[:50])

    relevant = _find_relevant_sections(question, summary)

    # 관련 섹션 없으면 답변 불가 처리
    if not relevant:
        logger.warning("관련 섹션 없음 — 답변 불가")
        return QAResult(
            answer="문서에서 해당 질문에 대한 내용을 찾을 수 없습니다.",
            sources=[],
            is_answerable=False,
        )

    # 관련 섹션 텍스트 조합
    context = "\n\n".join(
        f"[{sec.section}]\n" + "\n".join(f"- {b}" for b in sec.bullets)
        for sec in relevant
    )
    sources = [sec.section for sec in relevant]

    template    = QA_PROMPT_PATH.read_text(encoding="utf-8")
    user_prompt = template.format(question=question, context=context)

    try:
        raw = _call_api(
            messages=[
                {"role": "system", "content": (
                    "당신은 금융 문서 내용을 기반으로 질문에 답변하는 전문 AI입니다. "
                    "제공된 문서 내용에만 근거하여 답변하세요. "
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