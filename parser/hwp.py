"""
HWP 파서 — REQ-03
⚠️  구현 방식 미정 (Open Issue)
현재 후보: LibreOffice CLI .hwp → .docx 변환 후 DOCX 파서 적용

⚠️  LLM·VLM 사용 금지 (REQ-01)
"""
from pathlib import Path


def parse(hwp_path: Path) -> str:
    """HWP 파일을 파싱하여 텍스트를 반환한다."""
    raise NotImplementedError
