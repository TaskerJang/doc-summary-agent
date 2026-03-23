"""
LLM 요약 생성 — REQ-08, REQ-10
GPT-5.2 API 호출 + tenacity exponential backoff 재시도
pydantic BaseModel 기반 출력 구조 검증
"""
from typing import List


def summarize(chunks: List[dict]) -> dict:
    """청크 배열을 받아 전체 요약 및 섹션별 요약을 반환한다."""
    raise NotImplementedError
