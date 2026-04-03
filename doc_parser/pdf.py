from pathlib import Path

from doc_parser.ocr_cache import extract_text_with_cache


def is_image_based_pdf(pdf_path: Path) -> bool:
    """
    페이지당 평균 텍스트가 500자 미만이면 이미지 기반 PDF로 판단.
    (구 _is_image_based_pdf — public으로 승격)
    """
    import fitz
    doc = fitz.open(str(pdf_path))
    total_text = sum(len(page.get_text().strip()) for page in doc)
    avg_text_per_page = total_text / len(doc)
    return avg_text_per_page < 500


# 하위 호환 alias (내부 호출용)
_is_image_based_pdf = is_image_based_pdf


def parse(pdf_path: Path) -> str:
    """
    진입점 — 이미지 기반 PDF 여부를 먼저 판단 후 파싱 전략 결정
    이미지 기반 (80% 이상) → OCR
    텍스트 기반 → pymupdf4llm → pdfplumber fallback
    """
    if is_image_based_pdf(pdf_path):
        return _parse_ocr(pdf_path)

    text = _parse_pymupdf4llm(pdf_path)

    if not text.strip():
        text = _parse_pdfplumber(pdf_path)

    return text


def parse_bytes(data: bytes, filename: str = "upload.pdf") -> str:
    """
    parse()의 바이트 버전 — FastAPI UploadFile 대응
    임시 파일로 저장 후 parse() 위임
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        return parse(Path(tmp.name))


def _is_image_based_page(page) -> bool:
    """페이지 하나를 받아 이미지 기반인지 True/False 반환"""
    return not page.get_text().strip()


def _parse_pymupdf4llm(pdf_path: Path) -> str:
    """
    텍스트 페이지만 걸러서 pymupdf4llm으로 Markdown 변환
    이미지 페이지는 건너뜀
    """
    import fitz
    import pymupdf4llm

    doc = fitz.open(str(pdf_path))

    text_page_indices = [
        i for i, page in enumerate(doc)
        if not _is_image_based_page(page)
    ]

    if not text_page_indices:
        return ""

    return pymupdf4llm.to_markdown(
        doc,
        pages=text_page_indices,
        show_progress=False,
        table_strategy="lines_strict",
    )


def _parse_ocr(pdf_path: Path) -> str:
    """이미지 기반 PDF → EasyOCR + 캐시"""
    pages = extract_text_with_cache(pdf_path)
    return "\n\n".join(pages)


def _parse_pdfplumber(pdf_path: Path) -> str:
    """
    pymupdf4llm 결과가 비어있을 때 fallback
    텍스트 + 선 기반 표 추출
    """
    import pdfplumber

    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines.append(text)
            table_settings = {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
            }
            for table in page.extract_tables(table_settings):
                for row in table:
                    lines.append(" | ".join(cell or "" for cell in row))
    return "\n".join(lines)
