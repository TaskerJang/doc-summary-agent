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

# ── 상수 ──────────────────────────────────────────
MODEL          = "gpt-5.2"
TIMEOUT        = 60   # reasoning 모델 대응 — 30 → 60s
# chunk_size=700 기준 청크 수 감소 → Semaphore 상한도 16으로 상향
MAX_CONCURRENT = 16
PROMPTS_DIR         = Path(__file__).parent / "prompts"
SYSTEM_PROMPT_PATH  = PROMPTS_DIR / "system_v1.md"
OCR_WARNING_PATH    = PROMPTS_DIR / "ocr_warning_v1.md"
CHUNK_PROMPT_PATH   = PROMPTS_DIR / "chunk_summary_v1.md"
OVERALL_PROMPT_PATH = PROMPTS_DIR / "overall_summary_v1.md"

# ── LLM Runtime Config (#127) ───────────────────────────
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
# OpenRouter 경유 여부 — extra_body 전달 분기용
_use_openrouter_extras: bool = False


def configure_llm(
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
) -> None:
    """평가 진입점에서 LLM 호출을 일회성으로 재설정 (#127)."""
    global _active_client, _active_model, _use_openrouter_extras
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"환경변수 {api_key_env} 미설정 — LLM 재설정 불가")
    _active_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    if model is not None:
        _active_model = model
    # OpenRouter 경유면 extra_body (reasoning OFF) 박기
    _use_openrouter_extras = base_url is not None and "openrouter" in base_url.lower()
    logger.info(
        "LLM 재설정: model=%s base_url=%s api_key_env=%s openrouter_extras=%s",
        _active_model, base_url, api_key_env, _use_openrouter_extras,
    )

# ── AsyncOpenAI 클라이언트 (하위 호환 alias) ──────────────
client = _active_client

# ── Semaphore ──────────────────────────────────────────
_sem: asyncio.Semaphore | None = None


def _get_sem() -> asyncio.Semaphore:
    """실행 중인 이벤트 루프에서 Semaphore를 지연 초기화한다."""
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(MAX_CONCURRENT)
    return _sem


# ── 출력 스키마 ─────────────────────────────────────────
class SectionSummary(BaseModel):
    section:    str
    bullets:    list[str]
    source:     str
    chart_spec: dict[str, Any] | None = None


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


# ── OpenAI 네이티브 모델 감지 ───────────────────────────
def _is_openai_native_model(model: str) -> bool:
    """OpenAI 네이티브 모델인지 판단 — OpenRouter 경유 OpenAI 도 포함.

    OpenAI 모델은 reasoning 파라미터를 자체적으로 처리하므로 OpenRouter의
    extra_body reasoning OFF 바디는 적용하지 않는다 (BadRequest 위험 회피).
    """
    m = model.lower()
    if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return True
    if m.startswith("openai/"):
        return True
    return False


# ── Reasoning OFF for OpenRouter reasoning models ──────────
# OpenRouter 공식 문서 (https://openrouter.ai/docs/use-cases/reasoning-tokens):
#   - "max_tokens must be strictly higher than the reasoning budget"
#   - "exclude: false. Default: false. All models support this."
#   - "enabled": Default: inferred from `effort` or `max_tokens`
#
# 3중 안전 (공식 권장값 조합):
#   - enabled: false  → reasoning 자체 OFF (Anthropic 일부 모델)
#   - max_tokens: 1   → reasoning tokens 최소화 (모든 모델 호환, 공식 권장)
#   - exclude: true   → reasoning 응답 전달 X ("All models support this")
#
# provider routing 제거 (이전 코드의 ["moonshot", "deepinfra", "together"]):
#   - Kimi K2.5 전용 hack 이었고, OpenAI/Anthropic 으로 토글 시 fallback 더러웠음.
#   - OpenRouter 의 자동 provider 선택이 더 안전.
_REASONING_OFF_BODY = {
    "reasoning": {
        "enabled": False,
        "max_tokens": 1,
        "exclude": True,
    },
}


def _build_call_kwargs(messages: list[dict], max_tokens: int) -> dict:
    """LLM 호출 kwargs. 모델 종류에 따라 reasoning 토큰을 최소화한다.

    gpt-5.x / o-series 는 reasoning 모델이라 제어 없이 호출하면 추론 토큰이
    max_completion_tokens 예산을 잡아먹어 content 가 비거나, OpenRouter 가
    message.reasoning 으로 실어보낸 CoT 가 답변으로 누출된다. 따라서:
    - OpenAI reasoning 모델 (OpenRouter 경유): extra_body 의 reasoning.effort=minimal
    - OpenAI reasoning 모델 (OpenAI 직결): 표준 reasoning_effort=minimal
    - Kimi/DeepSeek/Claude 등 OpenRouter 경유 reasoning 모델: reasoning OFF body
    """
    kwargs: dict = {
        "model": _active_model,
        "messages": messages,
        "temperature": 0.3,
        "max_completion_tokens": max_tokens,
        "timeout": TIMEOUT,
    }
    if _is_openai_native_model(_active_model):
        if _use_openrouter_extras:                  # openai/gpt-5.2 등 OpenRouter 경유
            kwargs["extra_body"] = {"reasoning": {"effort": "minimal"}}
        else:                                       # OpenAI 직결
            kwargs["reasoning_effort"] = "minimal"
    elif _use_openrouter_extras:                    # Kimi / DeepSeek / Claude
        kwargs["extra_body"] = _REASONING_OFF_BODY
    return kwargs


# ── 재시도 데코레이터 ─────────────────────────────────────
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

    OpenRouter 경유 + 비-OpenAI 모델이면 extra_body 박혀서 reasoning OFF 적용.
    OpenAI 모델 (openai/gpt-*, gpt-*, o-series) 은 extra_body 미적용 —
    reasoning_effort 등은 OpenAI SDK 가 자체 처리.

    빈 응답 진단:
    - content 가 비어있으면 finish_reason 박혀서 logger.warning 박기
    - reasoning_content / reasoning 필드 fallback 시도
    """
    call_kwargs = _build_call_kwargs(messages, max_tokens)
    response = await _active_client.chat.completions.create(**call_kwargs)

    msg = response.choices[0].message if response.choices else None
    raw = (msg.content if msg else "") or ""

    if not raw.strip():
        finish_reason = response.choices[0].finish_reason if response.choices else "?"
        logger.warning(
            "LLM 빈 content (model=%s finish_reason=%s) — '[답변 불가]' 처리",
            _active_model, finish_reason,
        )
        # reasoning_content / reasoning 을 답으로 반환하지 않는다 — CoT 누출 차단.
        # 빈 문자열은 judge JSON 파싱 Error 를 유발하므로 '[답변 불가]' 로 정규화.
        return "[답변 불가]"

    return raw


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
                max_tokens=2000,
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


# ── Reduce 단계: 섹션 요약 → 전체 요약 ───────────────────────
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
            max_tokens=2500,
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
