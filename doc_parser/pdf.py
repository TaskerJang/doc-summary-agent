import re
from pathlib import Path

from doc_parser.ocr_cache import extract_text_with_cache, extract_page_with_cache

# pymupdf4llm이 이미지를 생략할 때 삽입하는 마커 패턴
_OMITTED_MARKER = "intentionally omitted"
_OMITTED_RE = re.compile(r"==> picture \[.*?\] intentionally omitted <==")


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
    이미지 기반 (페이지당 평균 500자 미만) → 전체 EasyOCR
    텍스트 기반 → pymupdf4llm + 페이지별 선택 OCR + pdfplumber fallback
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


def _has_embedded_images(md_text: str) -> bool:
    """pymupdf4llm 변환 결과에 생략된 이미지 마커가 있는지 확인"""
    return _OMITTED_MARKER in md_text


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
    페이지별 3단계 처리:
      1) get_text() 있음  → pymupdf4llm Markdown 변환
         1-1) 변환 결과에 이미지 마커 있음
              → EasyOCR로 OCR 후 'intentionally omitted' 마커 자리에 직접 치환
      2) get_text() 없음  → pdfplumber fallback
      3) pdfplumber도 빈 값 → EasyOCR 페이지 단위 OCR
    """
    import fitz
    import pymupdf4llm

    doc = fitz.open(str(pdf_path))
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
        md_pages = md.split("\f") if "\f" in md else [md]
        for idx, content in zip(text_page_indices, md_pages):
            pymupdf_result[idx] = content

    # pdfplumber fallback (get_text() 빈 페이지)
    pdfplumber_result: dict[int, str] = {}
    for i in fallback_page_indices:
        fb_text = _extract_page_text_pdfplumber(pdf_path, i)
        if fb_text.strip():
            pdfplumber_result[i] = fb_text

    # 원래 페이지 순서대로 병합 + 이미지 임베딩 페이지 EasyOCR 치환
    page_texts: list[str] = []
    for i in range(len(doc)):
        if i in pymupdf_result and pymupdf_result[i].strip():
            content = pymupdf_result[i]
            # intentionally omitted 마커를 EasyOCR 결과로 직접 치환
            if _has_embedded_images(content):
                ocr_text = extract_page_with_cache(pdf_path, i)
                if ocr_text.strip():
                    content = _OMITTED_RE.sub(ocr_text, content)
            page_texts.append(content)
        elif i in pdfplumber_result:
            page_texts.append(pdfplumber_result[i])
        else:
            # pdfplumber도 빈 경우 → EasyOCR 최후 수단
            ocr_text = extract_page_with_cache(pdf_path, i)
            if ocr_text.strip():
                page_texts.append(ocr_text)

    return "\n\n".join(page_texts)


def _parse_ocr(pdf_path: Path) -> str:
    """이미지 기반 PDF 전체 → EasyOCR + 캐시"""
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
