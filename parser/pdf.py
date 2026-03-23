"""
PDF 파서 — REQ-01
주 파서: pymupdf4llm (텍스트·Markdown 추출)
fallback: pdfplumber (복잡한 표 추출)

⚠️  LLM·VLM 사용 금지 (REQ-01)
⚠️  이미지 기반 PDF(미래에셋 1~3Q 등) OCR 처리 전략은 3/26 1차 보고 후 결정
"""
from pathlib import Path


# 이미지 기반 PDF 판정 기준: 유효 텍스트 비율이 이 값 미만이면 이미지 기반으로 간주
_IMAGE_BASED_THRESHOLD = 0.1


def parse(pdf_path: Path) -> str:
    """PDF 파일을 파싱하여 Markdown 텍스트를 반환한다.

    1) pymupdf4llm으로 전체 텍스트·표·이미지 캡션을 Markdown으로 추출
    2) 이미지 기반 PDF로 감지되면 [이미지 기반 PDF — 텍스트 추출 불가] 반환
    3) 복잡한 표가 깨지면 pdfplumber fallback으로 재추출
    4) 차트 등 이미지 내 수치는 추출 불가 → [차트 수치 미추출] 표기 (REQ-01 ③)
    """
    if _is_image_based(pdf_path):
        # TODO: OCR 전략 3/26 보고 후 결정. 현재는 명시적 메시지 반환.
        return f"[이미지 기반 PDF — 텍스트 추출 불가: {pdf_path.name}]"

    text = _parse_pymupdf4llm(pdf_path)

    if not text.strip():
        text = _parse_pdfplumber(pdf_path)

    return text


def _is_image_based(pdf_path: Path, sample_pages: int = 3) -> bool:
    """앞 페이지를 샘플링하여 이미지 기반 PDF 여부를 판별한다.

    fitz.get_text()로 텍스트 레이어가 없는 페이지 비율이 threshold 이상이면
    이미지 기반으로 간주한다.
    """
    import fitz

    doc = fitz.open(str(pdf_path))
    pages_to_check = min(sample_pages, len(doc))
    empty_count = 0

    for i in range(pages_to_check):
        if not doc[i].get_text().strip():
            empty_count += 1

    return (empty_count / pages_to_check) >= (1 - _IMAGE_BASED_THRESHOLD)


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
            text = page.extract_text()
            if text:
                lines.append(text)
            for table in page.extract_tables():
                for row in table:
                    lines.append(" | ".join(cell or "" for cell in row))

    return "\n".join(lines)
