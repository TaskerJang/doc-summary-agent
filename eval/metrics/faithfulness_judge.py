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
from openai import RateLimitError, APITimeoutError, APIConnectionError

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-5.2"

_PROMPTS_DIR = Path(__file__).parent / "prompts"
FAITHFULNESS_PROMPT           = (_PROMPTS_DIR / "faithfulness_v1.md").read_text(encoding="utf-8")
NUMERICAL_FAITHFULNESS_PROMPT = (_PROMPTS_DIR / "numerical_faithfulness_v1.md").read_text(encoding="utf-8")


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_api(prompt: str, response_format: dict) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=500,
        response_format=response_format,
        timeout=30,
    )
    import json
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
            "completeness": 0,
            "completeness_reason": "",
            "conciseness": 0,
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