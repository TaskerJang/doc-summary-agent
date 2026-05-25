"""eval/metrics/answer_correctness.py
RAGAS Answer Correctness + GraphRAG-Bench Accuracy 결합 — 의미적 일치 LLM Judge.

## 원천
- RAGAS Answer Correctness (https://docs.ragas.io)
- GraphRAG-Bench (arxiv:2506.02404, ICLR 2026) Accuracy 메트릭

doc-graph-agent 와 동일 코드 — #127 대칭 메트릭 분리.
"""
import json
import logging
from pathlib import Path

from openai import BadRequestError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APITimeoutError, APIConnectionError

from eval.metrics.faithfulness_judge import (
    _relax_schema,
    _is_openai_native_model,
)

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
ANSWER_CORRECTNESS_PROMPT = (_PROMPTS_DIR / "answer_correctness_v1.md").read_text(encoding="utf-8")


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_correctness_api(prompt: str, response_format: dict) -> dict:
    from eval.metrics import faithfulness_judge as fj

    client = fj._active_judge_client
    model = fj._active_judge_model
    strict_supported = fj._judge_strict_schema_supported

    rf = response_format if strict_supported else _relax_schema(response_format)

    call_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 1500,
        "response_format": rf,
        "timeout": 30,
    }
    if _is_openai_native_model(model):
        call_kwargs["reasoning_effort"] = "low"
    else:
        call_kwargs["extra_body"] = {"reasoning": {"enabled": False}}

    try:
        response = client.chat.completions.create(**call_kwargs)
    except BadRequestError as e:
        err_msg = str(e).lower()
        if "json_schema" in err_msg or "response_format" in err_msg or "strict" in err_msg:
            if fj._judge_strict_schema_supported:
                logger.warning("Answer Correctness strict json_schema 미지원 → json_object fallback")
                fj._judge_strict_schema_supported = False
                call_kwargs["response_format"] = _relax_schema(response_format)
                response = client.chat.completions.create(**call_kwargs)
            else:
                raise
        else:
            raise

    raw_content = response.choices[0].message.content if response.choices else ""
    raw_content = (raw_content or "").strip()

    if not raw_content:
        msg = response.choices[0].message if response.choices else None
        reasoning_content = getattr(msg, "reasoning_content", None) if msg else None
        if reasoning_content and "{" in reasoning_content:
            try:
                start = reasoning_content.find("{")
                end = reasoning_content.rfind("}")
                if start >= 0 and end > start:
                    return json.loads(reasoning_content[start:end+1])
            except (json.JSONDecodeError, ValueError):
                pass
        return {}

    if raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
            raw_content = raw_content[4:]
        raw_content = raw_content.strip()

    return json.loads(raw_content)


def judge_answer_correctness(question: str, reference: str, prediction: str) -> dict:
    prompt = ANSWER_CORRECTNESS_PROMPT.format(
        question=question[:1000],
        reference=reference[:2000],
        prediction=prediction[:2000],
    )
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_correctness_result",
            "schema": {
                "type": "object",
                "properties": {
                    "answer_correctness": {"type": "integer", "minimum": 1, "maximum": 5},
                    "reason":             {"type": "string"},
                },
                "required": ["answer_correctness", "reason"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    try:
        return _call_correctness_api(prompt, schema)
    except Exception as e:
        logger.error("Answer Correctness Judge 실패: %s", e)
        return {"answer_correctness": None, "reason": f"Error: {e}"}
