"""
eval/metrics/faithfulness_judge.py
LLM-as-Judge — Faithfulness / Completeness / Conciseness
참고: FineSurE (ACL 2024)
"""
import json
import logging
import os
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APITimeoutError, APIConnectionError

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL  = "gpt-5.2"

FAITHFULNESS_PROMPT = """당신은 금융 문서 요약의 품질을 평가하는 전문가입니다.
아래 원문과 요약을 읽고, 다음 세 가지 기준으로 평가하세요.

# 원문
{source}

# 요약
{summary}

# 평가 기준
1. Faithfulness: 요약이 원문에만 근거하는가? (원문에 없는 내용 생성 시 Not Faithful)
2. Completeness: 핵심 정보가 누락되지 않았는가? (1~5점)
3. Conciseness: 불필요한 내용이 없는가? (1~5점)

# 출력 형식 (JSON만 반환, 다른 텍스트 포함 금지)
{{"faithfulness": "Faithful" or "Not Faithful",
  "faithfulness_reason": "한 줄 근거",
  "completeness": 1~5,
  "completeness_reason": "한 줄 근거",
  "conciseness": 1~5,
  "conciseness_reason": "한 줄 근거"}}
"""

NUMERICAL_FAITHFULNESS_PROMPT = """당신은 금융 문서의 수치 정확도를 평가하는 전문가입니다.
아래 원문과 요약을 읽고, 요약에 등장하는 모든 수치가 원문과 일치하는지 판단하세요.

# 원문
{source}

# 요약
{summary}

# 출력 형식 (JSON만 반환)
{{"numerical_faithfulness": "Correct" or "Incorrect",
  "reason": "불일치 수치가 있다면 명시, 없으면 '모든 수치 일치'"}}
"""


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_api(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_completion_tokens=500,
        timeout=30,
    )
    return response.choices[0].message.content or ""


def judge_faithfulness(source: str, summary: str) -> dict:
    """
    Faithfulness / Completeness / Conciseness 평가
    Returns:
        {'faithfulness': str, 'completeness': int, 'conciseness': int, ...}
    """
    prompt = FAITHFULNESS_PROMPT.format(source=source[:3000], summary=summary)
    try:
        raw = _call_api(prompt).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
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
    try:
        raw = _call_api(prompt).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        logger.error("Numerical Faithfulness Judge 실패: %s", e)
        return {"numerical_faithfulness": "Error", "reason": str(e)}
