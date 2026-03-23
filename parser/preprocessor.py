"""
전처리 (노이즈 제거) — REQ-05
① Regex 기반 반복 텍스트 제거
② PyMuPDF clip 기반 상단·하단 영역 필터링
"""


def clean(text: str) -> str:
    """헤더·푸터·페이지 번호·워터마크 등 노이즈를 제거한다."""
    raise NotImplementedError
