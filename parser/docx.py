"""
DOCX 파서 — REQ-02
주 파서: python-docx (텍스트·표·단락 추출)
fallback: docx2python
.doc 입력 시 LibreOffice CLI → .docx 변환 후 처리

⚠️  LLM·VLM 사용 금지 (REQ-01)
"""
from pathlib import Path


def parse(docx_path: Path) -> str:
    """DOCX 파일을 파싱하여 텍스트를 반환한다."""
    raise NotImplementedError
