"""
PDF 파서 — REQ-01
주 파서: pymupdf4llm (텍스트·Markdown 추출)
fallback: pdfplumber (복잡한 표 추출)

⚠️  LLM·VLM 사용 금지 (REQ-01)
"""
from pathlib import Path


def parse(pdf_path: Path) -> str:
    """PDF 파일을 파싱하여 Markdown 텍스트를 반환한다."""
    raise NotImplementedError
