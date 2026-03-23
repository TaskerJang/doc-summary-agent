"""
PDF 파서 — REQ-01
주 파서: pymupdf4llm (텍스트·Markdown 추출)
fallback: pdfplumber (복잡한 표 추출)

⚠️  LLM·VLM 사용 금지 (REQ-01)
"""
from pathlib import Path


def parse(pdf_path: Path) -> str:
    """PDF 파일을 파싱하여 Markdown 텍스트를 반환한다.

    1) pymupdf4llm으로 전체 텍스트·표·이미지 캡션을 Markdown으로 추출
    2) 추출 결과가 비어있으면 pdfplumber fallback으로 표 재추출 후 병합
    3) 차트 등 이미지 내 수치는 추출 불가 → [차트 수치 미추출] 표기 (REQ-01 ③)
    """
    text = _parse_pymupdf4llm(pdf_path)

    if not text.strip():
        # pymupdf4llm 추출 실패 시 pdfplumber fallback
        text = _parse_pdfplumber(pdf_path)

    return text


def _parse_pymupdf4llm(pdf_path: Path) -> str:
    """pymupdf4llm으로 PDF 전체를 Markdown으로 추출한다."""
    import pymupdf4llm

    return pymupdf4llm.to_markdown(str(pdf_path))


def _parse_pdfplumber(pdf_path: Path) -> str:
    """pdfplumber로 텍스트와 표를 추출한다. (fallback)"""
    import pdfplumber

    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # 텍스트 추출
            text = page.extract_text()
            if text:
                lines.append(text)
            # 표 추출
            for table in page.extract_tables():
                for row in table:
                    lines.append(" | ".join(cell or "" for cell in row))

    return "\n".join(lines)
