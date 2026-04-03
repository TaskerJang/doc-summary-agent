"""
summarizer/llm.py
LLM 요약 생성 — REQ-08, REQ-10
GPT-5.2 API 호출 + tenacity exponential backoff 재시도
Pydantic BaseModel 기반 출력 스키마 검증
"""
import json
import logging
import os
from pathlib import Path

from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────────────────
MODEL       = "gpt-5.2"
TIMEOUT     = 30
PROMPTS_DIR         = Path(__file__).parent / "prompts"
SYSTEM_PROMPT_PATH  = PROMPTS_DIR / "system.md"
OCR_WARNING_PATH    = PROMPTS_DIR / "ocr_warning.md"
CHUNK_PROMPT_PATH   = PROMPTS_DIR / "chunk_summary.md"
OVERALL_PROMPT_PATH = PROMPTS_DIR / "overall_summary.md"

# ── OpenAI 클라이언트 ──────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── 출력 스키마 ────────────────────────────────────────────
class SectionSummary(BaseModel):
    section: str        # 섹션명
    bullets: list[str]  # 불릿 포인트
    source: str         # 출처 (섹션명)


class SummaryResult(BaseModel):
    overall: str                    # 전체 핵심 요약 3~5문장
    sections: list[SectionSummary]  # 섹션별 상세 요약
    is_image_based: bool            # OCR 문서 여부 (UI 경고 표시용)


# ── 프롬프트 로드 ──────────────────────────────────────────
def _load_system_prompt(is_image_based: bool) -> str:
    """system.md + (선택) ocr_warning.md 를 합쳐 시스템 프롬프트를 반환한다."""
    base = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    if is_image_based:
        base += OCR_WARNING_PATH.read_text(encoding="utf-8")
    return base


# ── 재시도 데코레이터 ──────────────────────────────────────
@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_api(messages: list[dict], max_tokens: int = 2000) -> str:
    """OpenAI API 호출 — 실패 시 exponential backoff 재시도 (최대 3회).

    재시도 대상: RateLimitError / APITimeoutError / APIConnectionError
    그 외 에러: 즉시 올림 (재시도 없음)
    backoff: 2s → 4s → 8s
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=max_tokens,
        timeout=TIMEOUT,
    )
    return response.choices[0].message.content or ""


# ── Map 단계: 청크 → 섹션 요약 ────────────────────────────
def _summarize_chunk(chunk: dict, system_prompt: str) -> SectionSummary | None:
    """단일 청크를 섹션 요약으로 변환한다."""
    section = chunk.get("section") or "(no section)"
    text    = chunk.get("text", "").strip()

    if not text:
        return None

    template    = CHUNK_PROMPT_PATH.read_text(encoding="utf-8")
    user_prompt = template.format(section=section, text=text)

    try:
        raw  = _call_api(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=500,
        )
        data = json.loads(raw)
        return SectionSummary(**data)
    except Exception as e:
        logger.warning("청크 요약 실패 (section=%r): %s", section, e)
        return None


# ── Reduce 단계: 섹션 요약 → 전체 요약 ────────────────────
def _summarize_overall(sections: list[SectionSummary], system_prompt: str) -> str:
    """섹션별 요약을 취합하여 전체 핵심 요약 3~5문장을 생성한다."""
    sections_text = "\n".join(
        f"[{s.section}]\n" + "\n".join(f"- {b}" for b in s.bullets)
        for s in sections
    )

    template    = OVERALL_PROMPT_PATH.read_text(encoding="utf-8")
    user_prompt = template.format(sections_text=sections_text)

    try:
        return _call_api(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=500,
        )
    except Exception as e:
        logger.error("전체 요약 생성 실패: %s", e)
        return "[전체 요약 생성 실패]"


# ── 공개 인터페이스 ────────────────────────────────────────
def summarize(chunks: list[dict], is_image_based: bool = False) -> SummaryResult:
    """
    청크 배열을 받아 전체 요약 및 섹션별 요약을 반환한다.

    Args:
        chunks:         chunker.chunk() 결과 List[Chunk]
        is_image_based: 이미지 기반 PDF 여부 (OCR 경고 프롬프트 추가)

    Returns:
        SummaryResult — overall / sections / is_image_based 필드 포함
    """
    logger.info("요약 시작 — 청크 %d개  is_image_based=%s", len(chunks), is_image_based)

    system_prompt = _load_system_prompt(is_image_based)

    # Map: 청크별 섹션 요약
    sections: list[SectionSummary] = []
    for i, c in enumerate(chunks):
        logger.debug("청크 요약 중 [%d/%d]", i + 1, len(chunks))
        result = _summarize_chunk(c, system_prompt)
        if result:
            sections.append(result)

    if not sections:
        logger.warning("섹션 요약 결과 없음 — 유효한 청크 없음")
        return SummaryResult(
            overall="[요약 생성 실패 — 유효한 청크 없음]",
            sections=[],
            is_image_based=is_image_based,
        )

    # Reduce: 전체 핵심 요약
    overall = _summarize_overall(sections, system_prompt)

    logger.info("요약 완료 — 섹션 %d개", len(sections))
    return SummaryResult(
        overall=overall,
        sections=sections,
        is_image_based=is_image_based,
    )