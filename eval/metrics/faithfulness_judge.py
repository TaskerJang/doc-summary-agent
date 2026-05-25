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
# OpenRouter 경유 여부 — extra_body 분기용
_use_openrouter_extras: bool = False


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
    global _active_judge_client, _active_judge_model, _judge_strict_schema_supported, _use_openrouter_extras
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"환경변수 {api_key_env} 미설정 — Judge LLM 재설정 불가")
    _active_judge_client = OpenAI(api_key=api_key, base_url=base_url)
    if model is not None:
        _active_judge_model = model
    # base_url 명시 시 — OpenAI 외 endpoint → strict schema 위험. 첫 호출에서 동적 fallback.
    _judge_strict_schema_supported = (base_url is None)
    # OpenRouter 경유면 extra_body (reasoning OFF) 박기
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
# Kimi K2.5 / claude-haiku-4.5 등 reasoning model 박힐 때
# enabled: false 박은 게 안 박힌 케이스 발견 — max_tokens: 0 + exclude: true
# 박아서 reasoning tokens 자체 0 박기.
_REASONING_OFF_BODY = {
    "reasoning": {
        "enabled": False,
        "max_tokens": 0,
        "exclude": True,
    },
}


def _relax_schema(response_format: dict) -> dict:
    """strict json_schema → json_object 로 완화 (#127).

    OpenRouter 경유 Claude/Kimi 등 일부 모델은 OpenAI의 strict json_schema를 미지원.
    이 경우 자동으로 json_object 모드로 fallback하여 자유 JSON 응답을 받는다.
    프롬프트에 이미 JSON 형식 안내가 박혀있어 응답 형식은 유지됨.
    """
    return {"type": "json_object"}


def _is_openai_native_model(model: str) -> bool:
    """OpenAI 네이티브 모델인지 판단 (#127 reasoning_effort 분기용).

    OpenAI 네이티브: gpt-*, o1, o3, o4 등 → reasoning_effort 지원
    OpenRouter 경유 OpenAI: openai/gpt-* → reasoning_effort 지원

    OpenRouter 경유 Anthropic/DeepSeek/Moonshot/xAI 등 → reasoning_effort 미지원 또는
    무시되어 content가 빈 문자열로 반환되는 케이스 다수 (LibreChat #9739 등 참고).
    """
    m = model.lower()
    # OpenAI 직결 또는 OpenRouter 경유 OpenAI
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
    """Judge LLM 호출 — _active_judge_client / _active_judge_model 사용 (#127).

    Model-specific 분기:
    - OpenAI 네이티브 (gpt-*, o-series) → reasoning_effort="low" 전달
    - OpenRouter 경유 Anthropic/DeepSeek/Moonshot/xAI → reasoning_effort 제거 +
      `extra_body={"reasoning": {"enabled": False, "max_tokens": 0, "exclude": True}}`
      로 thinking 모드 명시적 OFF (3중 안전)
      (참고: Claude OpenRouter는 reasoning_effort 무시 + thinking ON 시 content 빈 문자열 반환)
    - max_completion_tokens: 2000 (기존 500 → thinking 토큰 여유 확보)

    strict json_schema 비호환 endpoint에서 첫 BadRequestError 발생 시
    _judge_strict_schema_supported를 False로 set → 이후 호출은 json_object로 fallback.

    빈 응답 진단: content가 비어있으면 finish_reason과 함께 raw response 일부를 logger.warning으로 출력.
    """
    global _judge_strict_schema_supported
    import json

    # 비호환 endpoint면 처음부터 json_object 모드
    rf = response_format if _judge_strict_schema_supported else _relax_schema(response_format)

    # Model-specific 파라미터 분기 (#127 C7)
    call_kwargs = {
        "model": _active_judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 2000,  # 500 → 2000 (thinking 토큰 여유)
        "response_format": rf,
        "timeout": 60,  # 30 → 60s (reasoning 모델 대응)
    }
    if _is_openai_native_model(_active_judge_model):
        # OpenAI 네이티브만 reasoning_effort 지원
        call_kwargs["reasoning_effort"] = "low"
    elif _use_openrouter_extras:
        # OpenRouter 경유 — reasoning 3중 OFF 박기
        # enabled: false (Anthropic) + max_tokens: 0 (모든 모델) + exclude: true (응답 전달 X)
        call_kwargs["extra_body"] = _REASONING_OFF_BODY

    try:
        response = _active_judge_client.chat.completions.create(**call_kwargs)
    except BadRequestError as e:
        # strict json_schema 미지원 endpoint 감지 → fallback + 1회 재시도
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

    # 응답 content 추출 + 빈 응답 진단 (#127 C7)
    raw_content = response.choices[0].message.content if response.choices else ""
    raw_content = (raw_content or "").strip()

    if not raw_content:
        # 빈 응답 — 진단 정보 출력
        finish_reason = response.choices[0].finish_reason if response.choices else "?"
        # 일부 모델은 reasoning을 별도 필드로 반환 → 그 경우 fallback으로 추출
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
        # reasoning_content 안에 JSON이 들어있을 수도 있어 fallback 시도
        for candidate_field in (reasoning_content, reasoning):
            if candidate_field and "{" in candidate_field:
                try:
                    # 첫 { 부터 마지막 } 까지 추출 시도
                    start = candidate_field.find("{")
                    end = candidate_field.rfind("}")
                    if start >= 0 and end > start:
                        candidate = candidate_field[start:end+1]
                        return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    continue
        return {}

    # 마크다운 코드 블록 감싸기 (```json ... ```) 자동 제거
    if raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
            raw_content = raw_content[4:]
        raw_content = raw_content.strip()

    return json.loads(raw_content)


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
