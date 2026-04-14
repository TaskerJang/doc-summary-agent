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
    텍스트 기반 → pymupdf4llm (텍스트 없는 페이지는 pdfplumber fallback) → overall fallback
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


def _extract_page_text_pdfplumber(pdf_path: Path, page_index: int) -> str:
    """
    pdfplumber로 특정 페이지(0-based index)의 텍스트 + 선 기반 표를 추출.
    pymupdf4llm이 빈 문자열을 반환한 페이지에 대한 페이지 단위 fallback용.
    """
    import pdfplumber

    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        if page_index >= len(pdf.pages):
            return ""
        page = pdf.pages[page_index]

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


def _parse_pymupdf4llm(pdf_path: Path) -> str:
    """
    텍스트 페이지는 pymupdf4llm으로 Markdown 변환.
    텍스트가 없는 페이지(벡터 그래픽·표 등)는 pdfplumber로 fallback 처리하여
    내용 유실 없이 병합.
    """
    import fitz
    import pymupdf4llm

    doc = fitz.open(str(pdf_path))
    page_texts: list[str] = []

    text_page_indices = []
    fallback_page_indices = []

    for i, page in enumerate(doc):
        if _is_image_based_page(page):
            fallback_page_indices.append(i)
        else:
            text_page_indices.append(i)

    # pymupdf4llm으로 텍스트 페이지 일괄 변환
    pymupdf_result: dict[int, str] = {}
    if text_page_indices:
        md = pymupdf4llm.to_markdown(
            doc,
            pages=text_page_indices,
            show_progress=False,
            table_strategy="lines_strict",
        )
        # pymupdf4llm은 pages 순서대로 결과를 반환 → 인덱스 매핑
        md_pages = md.split("\f") if "\f" in md else [md]
        for idx, content in zip(text_page_indices, md_pages):
            pymupdf_result[idx] = content

    # pdfplumber로 fallback 페이지 추출
    pdfplumber_result: dict[int, str] = {}
    for i in fallback_page_indices:
        fb_text = _extract_page_text_pdfplumber(pdf_path, i)
        if fb_text.strip():
            pdfplumber_result[i] = fb_text

    # 원래 페이지 순서대로 병합
    for i in range(len(doc)):
        if i in pymupdf_result and pymupdf_result[i].strip():
            page_texts.append(pymupdf_result[i])
        elif i in pdfplumber_result:
            page_texts.append(pdfplumber_result[i])

    return "\n\n".join(page_texts)


def _parse_ocr(pdf_path: Path) -> str:
    """이미지 기반 PDF → EasyOCR + 캐시"""
    pages = extract_text_with_cache(pdf_path)
    return "\n\n".join(pages)


def _parse_pdfplumber(pdf_path: Path) -> str:
    """
    pymupdf4llm 결과가 전체 공백일 때 overall fallback.
    텍스트 + 선 기반 표 추출.
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
