"""
청킹 — REQ-07
섹션/단락 단위 분할, 표·이미지는 독립 청크 처리
⚠️  구현 방식 미정: 직접 구현 vs LangChain TextSplitter
"""
from typing import List


def chunk(text: str) -> List[dict]:
    """파싱된 텍스트를 청크 배열로 분할하여 반환한다."""
    raise NotImplementedError
