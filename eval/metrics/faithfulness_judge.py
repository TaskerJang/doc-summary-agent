"""
eval/metrics/faithfulness_judge.py
LLM-as-Judge — Faithfulness / Completeness / Conciseness
참고: FineSurE (ACL 2024)

Reasoning OFF (#54 추가 정리):
- max_tokens: 0 → 1 (OpenRouter 공식 권장값, "max_tokens must be strictly higher than reasoning budget")
- OpenAI 직결 → reasoning_effort="low"
- OpenRouter 경유 비-OpenAI 모델 → extra_body 로 reasoning OFF 3중
"""
import logging
import os
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APITimeoutError, APIConnectionError, BadRequestError

logger = logging.getLogger(__name__)

# ── 기본 설정 (OpenAI GPT-5.2) ───────────────────────────
DEFAULT_MODEL = "gpt-5.2"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = DEFAULT_MODEL

_active_judge_client: OpenAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_active_judge_model: str = DEFAULT_MODEL
_judge_strict_schema_supported: bool = True
_use_openrouter_extras: bool = False


def configure_judge_llm(
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
) -> None:
    global _active_judge_client, _active_judge_model, _judge_strict_schema_supported, _use_openrouter_extras
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"환경변수 {api_key_env} 미설정 — Judge LLM 재설정 불가")
    _active_judge_client = OpenAI(api_key=api_key, base_url=base_url)
    if model is not None:
        _active_judge_model = model
    _judge_strict_schema_supported = (base_url is None)
    _use_openrouter_extras = base_url is not None and "openrouter" in base_url.lower()
    logger.info(
        "Judge LLM 재설정: model=%s base_url=%s api_key_env=%s strict_schema=%s openrouter=%s",
        _active_judge_model, base_url, api_key_env,
        _judge_strict_schema_supported, _use_openrouter_extras,
    )


_PROMPTS_DIR = Path(__file__).parent / "prompts"
FAITHFULNESS_PROMPT           = (_PROMPTS_DIR / "faithfulness_v1.md").read_text(encoding="utf-8")
NUMERICAL_FAITHFULNESS_PROMPT = (_PROMPTS_DIR / "numerical_faithfulness_v1.md").read_text(encoding="utf-8")


# ── Reasoning OFF for OpenRouter reasoning models ──────────
# 공식 문서: https://openrouter.ai/docs/use-cases/reasoning-tokens
# 3중 안전:
#   - enabled: false  → Anthropic 일부 모델
#   - max_tokens: 1   → 모든 모델 호환 (공식 권장)
#   - exclude: true   → "All models support this"
_REASONING_OFF_BODY = {
    "reasoning": {
        "enabled": False,
        "max_tokens": 1,
        "exclude": True,
    },
}


def _relax_schema(response_format: dict) -> dict:
    """strict json_schema → json_object 로 완화."""
    return {"type": "json_object"}


def _is_openai_native_model(model: str) -> bool:
    """OpenAI 네이티브 모델인지 판단 — OpenRouter 경유 OpenAI 도 포함.

    OpenAI 모델은 reasoning 파라미터를 자체 처리 → reasoning_effort 전달.
    비-OpenAI 모델 (Anthropic/DeepSeek/Moonshot/xAI) 은 OpenRouter extra_body 로 reasoning OFF.
    """
    m = model.lower()
    if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return True
    if m.startswith("openai/"):
        return True
    return False


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_api(prompt: str, response_format: dict) -> dict:
    """Judge LLM 호출 — _active_judge_client / _active_judge_model 사용.

    Model-specific 분기:
    - OpenAI 네이티브 (gpt-*, o-series, openai/*) → reasoning_effort="low" 전달
    - OpenRouter 경유 비-OpenAI (Anthropic/DeepSeek/Moonshot/xAI) → extra_body 로 reasoning OFF 3중
    - max_completion_tokens: 2000 (thinking 토큰 여유)

    strict json_schema 비호환 endpoint에서 첫 BadRequestError 발생 시
    _judge_strict_schema_supported를 False로 set → 이후 호출은 json_object로 fallback.
    """
    global _judge_strict_schema_supported
    import json

    rf = response_format if _judge_strict_schema_supported else _relax_schema(response_format)

    call_kwargs = {
        "model": _active_judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 2000,
        "response_format": rf,
        "timeout": 60,
    }
    if _is_openai_native_model(_active_judge_model):
        # OpenAI 네이티브만 reasoning_effort 지원
        call_kwargs["reasoning_effort"] = "low"
    elif _use_openrouter_extras:
        # OpenRouter 경유 비-OpenAI — reasoning 3중 OFF 박기
        call_kwargs["extra_body"] = _REASONING_OFF_BODY

    try:
        response = _active_judge_client.chat.completions.create(**call_kwargs)
    except BadRequestError as e:
        err_msg = str(e).lower()
        if "json_schema" in err_msg or "response_format" in err_msg or "strict" in err_msg:
            if _judge_strict_schema_supported:
                logger.warning(
                    "Judge model가 strict json_schema 미지원 → json_object로 fallback: %s", e
                )
                _judge_strict_schema_supported = False
                call_kwargs["response_format"] = _relax_schema(response_format)
                response = _active_judge_client.chat.completions.create(**call_kwargs)
            else:
                raise
        else:
            raise

    raw_content = response.choices[0].message.content if response.choices else ""
    raw_content = (raw_content or "").strip()

    if not raw_content:
        finish_reason = response.choices[0].finish_reason if response.choices else "?"
        msg = response.choices[0].message if response.choices else None
        reasoning_content = getattr(msg, "reasoning_content", None) if msg else None
        reasoning = getattr(msg, "reasoning", None) if msg else None
        logger.warning(
            "Judge 응답 빈 content (model=%s finish_reason=%s) — "
            "reasoning_content=%s reasoning=%s",
            _active_judge_model, finish_reason,
            (reasoning_content or "")[:200] if reasoning_content else None,
            (reasoning or "")[:200] if reasoning else None,
        )
        for candidate_field in (reasoning_content, reasoning):
            if candidate_field and "{" in candidate_field:
                try:
                    start = candidate_field.find("{")
                    end = candidate_field.rfind("}")
                    if start >= 0 and end > start:
                        candidate = candidate_field[start:end+1]
                        return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    continue
        return {}

    if raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
            raw_content = raw_content[4:]
        raw_content = raw_content.strip()

    return json.loads(raw_content)


def judge_faithfulness(source: str, summary: str) -> dict:
    prompt = FAITHFULNESS_PROMPT.format(source=source[:3000], summary=summary)
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "faithfulness_result",
            "schema": {
                "type": "object",
                "properties": {
                    "faithfulness":          {"type": "string", "enum": ["Faithful", "Not Faithful"]},
                    "faithfulness_reason":   {"type": "string"},
                    "completeness":          {"type": "integer", "minimum": 1, "maximum": 5},
                    "completeness_reason":   {"type": "string"},
                    "conciseness":           {"type": "integer", "minimum": 1, "maximum": 5},
                    "conciseness_reason":    {"type": "string"},
                },
                "required": ["faithfulness", "faithfulness_reason",
                             "completeness", "completeness_reason",
                             "conciseness", "conciseness_reason"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    try:
        return _call_api(prompt, schema)
    except Exception as e:
        logger.error("Faithfulness Judge 실패: %s", e)
        return {
            "faithfulness": "Error",
            "faithfulness_reason": str(e),
            "completeness": None,
            "completeness_reason": "",
            "conciseness": None,
            "conciseness_reason": "",
        }


def judge_numerical_faithfulness(source: str, summary: str) -> dict:
    prompt = NUMERICAL_FAITHFULNESS_PROMPT.format(source=source[:3000], summary=summary)
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "numerical_faithfulness_result",
            "schema": {
                "type": "object",
                "properties": {
                    "numerical_faithfulness": {"type": "string", "enum": ["Correct", "Incorrect"]},
                    "reason":                 {"type": "string"},
                },
                "required": ["numerical_faithfulness", "reason"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    try:
        return _call_api(prompt, schema)
    except Exception as e:
        logger.error("Numerical Faithfulness Judge 실패: %s", e)
        return {"numerical_faithfulness": "Error", "reason": str(e)}
