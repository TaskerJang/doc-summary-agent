import json
import logging
from pathlib import Path

from summarizer.llm import SummaryResult, _call_api

logger = logging.getLogger(__name__)

PROMPTS_DIR         = Path(__file__).parent / "prompts"
QA_PROMPT_PATH      = PROMPTS_DIR / "qa_v1.md"
FOLLOW_UP_PROMPT_PATH = PROMPTS_DIR / "follow_up_v1.md"


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
    return [sec for _, sec in scored[:3]]


def ask(question: str, summary: SummaryResult) -> QAResult:
    logger.info("Q&A 시작 — 질문: %r", question[:50])

    relevant = _find_relevant_sections(question, summary)

    if not relevant:
        logger.warning("관련 섹션 없음 — 답변 불가")
        return QAResult(
            answer="문서에서 해당 질문에 대한 내용을 찾을 수 없습니다.",
            sources=[],
            is_answerable=False,
        )

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