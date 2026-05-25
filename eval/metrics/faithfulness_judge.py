"""
eval/metrics/faithfulness_judge.py
LLM-as-Judge — Faithfulness / Completeness / Conciseness
참고: FineSurE (ACL 2024)
"""
import logging
import os
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APITimeoutError, APIConnectionError, BadRequestError

logger = logging.getLogger(__name__)

# ── 기본 설정 (OpenAI GPT-5.2) ────────────────────────────
DEFAULT_MODEL = "gpt-5.2"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = DEFAULT_MODEL  # 하위 호환 alias

# ── Judge LLM Runtime Config (#127) ──────────────────────
# 평가 진입점에서 configure_judge_llm()으로 토글 가능.
# 측정 대상 LLM과 분리 보장 — measurement bias 방지.
# 호출 안 하면 기본 GPT-5.2 + OpenAI 직결.
_active_judge_client: OpenAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_active_judge_model: str = DEFAULT_MODEL
# OpenAI 외 endpoint (OpenRouter 등) 사용 시 strict JSON schema 비호환 가능 → fallback 플래그
_judge_strict_schema_supported: bool = True


def configure_judge_llm(
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
) -> None:
    """평가 진입점에서 Judge LLM을 일회성으로 재설정한다 (#127).

    측정 대상 LLM (summarizer.llm.configure_llm)과 *독립적으로* 설정 → judge bias 방지.

    Args:
        model: Judge 모델 ID (예: "anthropic/claude-haiku-4.5"). None이면 기존 GPT-5.2 유지.
        base_url: OpenAI-호환 endpoint URL (예: OpenRouter면 "https://openrouter.ai/api/v1").
        api_key_env: API 키 환경변수 이름 (기본 "OPENAI_API_KEY", OpenRouter면 "OPENROUTER_API_KEY").

    호출 안 하면 기본값 (GPT-5.2 + OpenAI 직결) 유지 — 기존 leaderboard 결과와 호환.

    OpenAI 외 endpoint (OpenRouter Claude 등) 사용 시 strict JSON schema 비호환 가능 →
    자동으로 fallback 플래그 set (첫 호출에서 BadRequestError 잡으면 schema 완화).
    """
    global _active_judge_client, _active_judge_model, _judge_strict_schema_supported
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"환경변수 {api_key_env} 미설정 — Judge LLM 재설정 불가")
    _active_judge_client = OpenAI(api_key=api_key, base_url=base_url)
    if model is not None:
        _active_judge_model = model
    # base_url 명시 시 — OpenAI 외 endpoint → strict schema 위험. 첫 호출에서 동적 fallback.
    _judge_strict_schema_supported = (base_url is None)
    logger.info(
        "Judge LLM 재설정: model=%s base_url=%s api_key_env=%s strict_schema=%s",
        _active_judge_model, base_url, api_key_env, _judge_strict_schema_supported,
    )


_PROMPTS_DIR = Path(__file__).parent / "prompts"
FAITHFULNESS_PROMPT           = (_PROMPTS_DIR / "faithfulness_v1.md").read_text(encoding="utf-8")
NUMERICAL_FAITHFULNESS_PROMPT = (_PROMPTS_DIR / "numerical_faithfulness_v1.md").read_text(encoding="utf-8")


def _relax_schema(response_format: dict) -> dict:
    """strict json_schema → json_object 로 완화 (#127).

    OpenRouter 경유 Claude/Kimi 등 일부 모델은 OpenAI의 strict json_schema를 미지원.
    이 경우 자동으로 json_object 모드로 fallback하여 자유 JSON 응답을 받는다.
    프롬프트에 이미 JSON 형식 안내가 박혀있어 응답 형식은 유지됨.
    """
    return {"type": "json_object"}


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_api(prompt: str, response_format: dict) -> dict:
    """Judge LLM 호출 — _active_judge_client / _active_judge_model 사용 (#127).

    strict json_schema 비호환 endpoint에서 첫 BadRequestError 발생 시
    _judge_strict_schema_supported를 False로 set → 이후 호출은 json_object로 fallback.
    """
    global _judge_strict_schema_supported
    import json

    # 비호환 endpoint면 처음부터 json_object 모드
    rf = response_format if _judge_strict_schema_supported else _relax_schema(response_format)

    try:
        response = _active_judge_client.chat.completions.create(
            model=_active_judge_model,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="low",
            max_completion_tokens=500,
            response_format=rf,
            timeout=30,
        )
    except BadRequestError as e:
        # strict json_schema 미지원 endpoint 감지 → fallback + 1회 재시도
        err_msg = str(e).lower()
        if "json_schema" in err_msg or "response_format" in err_msg or "strict" in err_msg:
            if _judge_strict_schema_supported:
                logger.warning(
                    "Judge model가 strict json_schema 미지원 → json_object로 fallback: %s", e
                )
                _judge_strict_schema_supported = False
                response = _active_judge_client.chat.completions.create(
                    model=_active_judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    reasoning_effort="low",
                    max_completion_tokens=500,
                    response_format=_relax_schema(response_format),
                    timeout=30,
                )
            else:
                raise
        else:
            raise

    return json.loads(response.choices[0].message.content or "{}")


def judge_faithfulness(source: str, summary: str) -> dict:
    """
    Faithfulness / Completeness / Conciseness 평가
    Returns:
        {'faithfulness': str, 'completeness': int, 'conciseness': int, ...}
    """
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
    """
    수치 충실도 LLM 판정
    Returns:
        {'numerical_faithfulness': str, 'reason': str}
    """
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
