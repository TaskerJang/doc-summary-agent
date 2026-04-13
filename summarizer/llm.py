import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
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
MAX_WORKERS = 5
PROMPTS_DIR         = Path(__file__).parent / "prompts"
SYSTEM_PROMPT_PATH  = PROMPTS_DIR / "system_v1.md"
OCR_WARNING_PATH    = PROMPTS_DIR / "ocr_warning_v1.md"
CHUNK_PROMPT_PATH   = PROMPTS_DIR / "chunk_summary_v1.md"
OVERALL_PROMPT_PATH = PROMPTS_DIR / "overall_summary_v1.md"

# ── OpenAI 클라이언트 ──────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── 출력 스키마 ────────────────────────────────────────────
class SectionSummary(BaseModel):
    section: str
    bullets: list[str]
    source: str


class SummaryResult(BaseModel):
    overall: str
    sections: list[SectionSummary]
    is_image_based: bool


# ── 청크 수 → 불릿 수 결정 ────────────────────────────────
def _resolve_bullet_count(chunk_count: int) -> int:
    """청크 수에 따라 overall 요약의 불릿 수를 동적으로 결정한다."""
    if chunk_count <= 10:
        return 5
    elif chunk_count <= 25:
        return 8
    else:
        return 12


# ── 프롬프트 로드 ──────────────────────────────────────────
def _load_system_prompt(is_image_based: bool) -> str:
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
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_completion_tokens=max_tokens,
        timeout=TIMEOUT,
    )
    return response.choices[0].message.content or ""


# ── Map 단계: 청크 → 섹션 요약 ────────────────────────────
def _summarize_chunk(chunk: dict, system_prompt: str) -> SectionSummary | None:
    section = chunk.get("section") or "(no section)"
    text    = chunk.get("text", "").strip()

    if not text:
        return None

    template    = CHUNK_PROMPT_PATH.read_text(encoding="utf-8")
    user_prompt = template.format(section=section, text=text)

    try:
        raw = _call_api(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=500,
        )

        raw = raw.strip()
        if not raw:
            logger.warning("청크 요약 빈 응답 (section=%r)", section)
            return None

        # 마크다운 코드블록 제거
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        # JSON 파싱 — 실패 시 작은따옴표 이스케이프 후 재시도
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raw_fixed = re.sub(
                r"(?<=: \")([^\"]*)'([^\"]*?)(?=\")", r"\1\'\2", raw
            )
            data = json.loads(raw_fixed)

        return SectionSummary(**data)
    except Exception as e:
        logger.warning("청크 요약 실패 (section=%r): %s", section, e)
        return None


# ── Reduce 단계: 섹션 요약 → 전체 요약 ────────────────────
def _summarize_overall(
    sections: list[SectionSummary],
    system_prompt: str,
    chunk_count: int,
) -> str:
    sections_text = "\n".join(
        f"[{s.section}]\n" + "\n".join(f"- {b}" for b in s.bullets)
        for s in sections
    )
    sections_text = re.sub(r'(\d+\.?\d*)~+(\d+\.?\d*)', r'\1에서 \2', sections_text)

    bullet_count = _resolve_bullet_count(chunk_count)
    template    = OVERALL_PROMPT_PATH.read_text(encoding="utf-8")
    user_prompt = template.format(
        sections_text=sections_text,
        bullet_count=bullet_count,
    )

    try:
        result = _call_api(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=1000,
        )

        if not result or not result.strip():
            logger.warning("전체 요약 빈 응답 — 첫 섹션 불릿으로 대체")
            return sections[0].bullets[0] if sections and sections[0].bullets else "[전체 요약 생성 실패]"

        # ~~/~ 모두 "에서"로 치환
        result = re.sub(r'(\d+\.?\d*)~+(\d+\.?\d*)', r'\1에서 \2', result)
        return result

    except Exception as e:
        logger.error("전체 요약 생성 실패: %s", e)
        return "[전체 요약 생성 실패]"


# ── 공개 인터페이스 ────────────────────────────────────────
def summarize(chunks: list[dict], is_image_based: bool = False) -> SummaryResult:
    """
    청크 배열을 받아 전체 요약 및 섹션별 요약을 반환한다.
    청크 요약은 ThreadPoolExecutor로 병렬 처리한다.
    """
    chunk_count = len(chunks)
    logger.info("요약 시작 — 청크 %d개  is_image_based=%s", chunk_count, is_image_based)

    system_prompt = _load_system_prompt(is_image_based)

    # Map: 병렬 청크 요약
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(_summarize_chunk, c, system_prompt)
            for c in chunks
        ]
        sections: list[SectionSummary] = [
            f.result() for f in futures if f.result() is not None
        ]

    if not sections:
        logger.warning("섹션 요약 결과 없음 — 유효한 청크 없음")
        return SummaryResult(
            overall="[요약 생성 실패 — 유효한 청크 없음]",
            sections=[],
            is_image_based=is_image_based,
        )

    # Reduce: 전체 핵심 요약 (청크 수 기반 불릿 수 동적 결정)
    overall = _summarize_overall(sections, system_prompt, chunk_count)

    logger.info("요약 완료 — 섹션 %d개  불릿 수 기준 %d개", len(sections), _resolve_bullet_count(chunk_count))
    return SummaryResult(
        overall=overall,
        sections=sections,
        is_image_based=is_image_based,
    )
