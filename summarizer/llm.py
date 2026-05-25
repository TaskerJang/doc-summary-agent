import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────────────────
MODEL          = "gpt-5.2"
TIMEOUT        = 60   # reasoning 모델 대응 — 30 → 60s 박음
# chunk_size=700 기준 청크 수 감소 → Semaphore 상한도 16으로 상향
# rate limit 여유가 있으면 더 올릴 수 있음
MAX_CONCURRENT = 16
PROMPTS_DIR         = Path(__file__).parent / "prompts"
SYSTEM_PROMPT_PATH  = PROMPTS_DIR / "system_v1.md"
OCR_WARNING_PATH    = PROMPTS_DIR / "ocr_warning_v1.md"
CHUNK_PROMPT_PATH   = PROMPTS_DIR / "chunk_summary_v1.md"
OVERALL_PROMPT_PATH = PROMPTS_DIR / "overall_summary_v1.md"

# ── LLM Runtime Config (#127) ────────────────────────────
# 평가 진입점에서 configure_llm()으로 토글 가능. 호출 안 하면 기본 GPT-5.2 + OpenAI 직결.
# prod 경로 (Chainlit UI)는 configure_llm()을 호출하지 않으므로 영향 0.
@dataclass
class LLMConfig:
    """LLM 호출 설정. OpenRouter 등 OpenAI-호환 endpoint 토글용."""
    model: str
    base_url: str | None = None  # None이면 OpenAI 기본
    api_key_env: str = "OPENAI_API_KEY"


# 모듈 레벨 상태 — configure_llm()으로 재설정 가능
_active_client: AsyncOpenAI = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_active_model: str = MODEL


def configure_llm(
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
) -> None:
    """평가 진입점에서 LLM 호출을 일회성으로 재설정한다 (#127).

    Args:
        model: 사용할 모델 ID (예: "openai/gpt-5-mini", "moonshotai/kimi-k2.5").
               None이면 기존 MODEL ("gpt-5.2") 유지.
        base_url: OpenAI-호환 endpoint base URL. OpenRouter면 "https://openrouter.ai/api/v1".
                  None이면 OpenAI 기본.
        api_key_env: API 키를 읽을 환경변수 이름. 기본 "OPENAI_API_KEY".

    호출 안 하면 모듈 import 시점의 기본값 (gpt-5.2 + OpenAI 직결) 유지 — prod 영향 0.
    """
    global _active_client, _active_model
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"환경변수 {api_key_env} 미설정 — LLM 재설정 불가")
    _active_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    if model is not None:
        _active_model = model
    logger.info(
        "LLM 재설정: model=%s base_url=%s api_key_env=%s",
        _active_model, base_url, api_key_env,
    )

# ── AsyncOpenAI 클라이언트 (하위 호환 alias) ──────────────
# 외부에서 `from summarizer.llm import client` 하는 경로가 있을 수 있어 유지.
# 단, 실제 _call_api는 _active_client를 사용 — configure_llm() 시 토글됨.
client = _active_client

# ── Semaphore (모듈 레벨 — 이벤트 루프와 생명주기 공유) ────
_sem: asyncio.Semaphore | None = None


def _get_sem() -> asyncio.Semaphore:
    """실행 중인 이벤트 루프에서 Semaphore를 지연 초기화한다."""
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(MAX_CONCURRENT)
    return _sem


# ── 출력 스키마 ────────────────────────────────────────────
class SectionSummary(BaseModel):
    section:    str
    bullets:    list[str]
    source:     str
    chart_spec: dict[str, Any] | None = None  # 시각화 라우팅용 (이슈 #60)


class SummaryResult(BaseModel):
    overall: str
    sections: list[SectionSummary]
    is_image_based: bool


# ── 청크 수 → 불릿 수 결정 ────────────────────────────────
def _resolve_bullet_count(chunk_count: int) -> int:
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


# ── Reasoning OFF for OpenRouter reasoning models ──────────
# Kimi K2.5, DeepSeek V3.2, GPT-5 등 reasoning 모델 박힌 케이스에서
# thinking tokens 가 max_tokens 다 박아버리고 실제 출력은 빈 문자열 박힘.
# OpenRouter 의 reasoning 파라미터로 reasoning 자체를 OFF 박아 출력 보장.
# OpenAI 직결 시에는 extra_body 무시되므로 prod 경로 영향 0.
_REASONING_OFF_BODY = {
    "reasoning": {
        "enabled": False,
    },
}


# ── 재시도 데코레이터 ──────────────────────────────────────
@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def _call_api(messages: list[dict], max_tokens: int = 2000) -> str:
    """LLM 호출 — _active_client / _active_model 사용 (#127).

    configure_llm() 호출 안 한 상태면 기본 GPT-5.2 + OpenAI 직결.
    호출했다면 그 설정대로 동작.

    extra_body 의 reasoning OFF 박힘 — OpenRouter 의 reasoning model
    (Kimi K2.5, DeepSeek V3.2 등) 에서 thinking tokens 가 max_tokens
    다 박아버리고 출력 빈 문자열 박는 문제 방지. OpenAI 직결은 무시.
    """
    response = await _active_client.chat.completions.create(
        model=_active_model,
        messages=messages,
        temperature=0.3,
        max_completion_tokens=max_tokens,
        timeout=TIMEOUT,
        extra_body=_REASONING_OFF_BODY,
    )
    return response.choices[0].message.content or ""


# ── Map 단계: 청크 → 섹션 요약 (Semaphore 보호) ───────────
async def _summarize_chunk(chunk: dict, system_prompt: str) -> SectionSummary | None:
    section = chunk.get("section") or "(no section)"
    text    = chunk.get("text", "").strip()

    if not text:
        return None

    template    = CHUNK_PROMPT_PATH.read_text(encoding="utf-8")
    user_prompt = template.format(section=section, text=text)

    async with _get_sem():
        try:
            raw = await _call_api(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=600,
            )

            raw = raw.strip()
            if not raw:
                logger.warning("청크 요약 빈 응답 (section=%r)", section)
                return None

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                raw_fixed = re.sub(
                    r"(?<=: \")([^\"]*)'([^\"]*?)(?=\")", r"\1\'\2", raw
                )
                data = json.loads(raw_fixed)

            chart_spec = data.get("chart_spec")
            if isinstance(chart_spec, dict):
                if str(chart_spec.get("chart_type", "none")).lower() == "none":
                    chart_spec = None
            else:
                chart_spec = None

            data["chart_spec"] = chart_spec
            return SectionSummary(**data)

        except Exception as e:
            logger.warning("청크 요약 실패 (section=%r): %s", section, e)
            return None


# ── Reduce 단계: 섹션 요약 → 전체 요약 ────────────────────
async def _summarize_overall(
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
        result = await _call_api(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=1000,
        )

        if not result or not result.strip():
            logger.warning("전체 요약 빈 응답 — 첫 섹션 불릿으로 대체")
            return sections[0].bullets[0] if sections and sections[0].bullets else "[전체 요약 생성 실패]"

        result = re.sub(r'(\d+\.?\d*)~+(\d+\.?\d*)', r'\1에서 \2', result)
        return result

    except Exception as e:
        logger.error("전체 요약 생성 실패: %s", e)
        return "[전체 요약 생성 실패]"


# ── 공개 인터페이스 ────────────────────────────────────────
async def summarize(chunks: list[dict], is_image_based: bool = False) -> SummaryResult:
    chunk_count = len(chunks)
    logger.info("요약 시작 — 청크 %d개  is_image_based=%s", chunk_count, is_image_based)

    system_prompt = _load_system_prompt(is_image_based)

    raw_results: list[SectionSummary | None] = await asyncio.gather(
        *[_summarize_chunk(c, system_prompt) for c in chunks]
    )
    sections: list[SectionSummary] = [r for r in raw_results if r is not None]

    if not sections:
        logger.warning("섹션 요약 결과 없음 — 유효한 청크 없음")
        return SummaryResult(
            overall="[요약 생성 실패 — 유효한 청크 없음]",
            sections=[],
            is_image_based=is_image_based,
        )

    overall = await _summarize_overall(sections, system_prompt, chunk_count)

    chart_count = sum(1 for s in sections if s.chart_spec is not None)
    logger.info(
        "요약 완료 — 섹션 %d개  불릿 수 기준 %d개  차트 생성 대상 %d개",
        len(sections), _resolve_bullet_count(chunk_count), chart_count,
    )
    return SummaryResult(
        overall=overall,
        sections=sections,
        is_image_based=is_image_based,
    )
