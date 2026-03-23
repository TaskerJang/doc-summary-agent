"""
PDF 파서 — REQ-01
주 파서: pymupdf4llm (텍스트·Markdown 추출)
fallback: pdfplumber (복잡한 표 추출)

⚠️  LLM·VLM 사용 금지 (REQ-01)
⚠️  이미지 기반 PDF(미래에셋 1~3Q 등) OCR 처리 전략은 3/26 1차 보고 후 결정

실행:
    python parser/pdf.py
"""
import time
from pathlib import Path

# 이미지 기반 PDF 판정 기준: 텍스트 없는 페이지 비율이 이 값 이상이면 이미지 기반으로 간주
_IMAGE_BASED_THRESHOLD = 0.1

# 테스트 문서 경로 (로컬 sample_docs 기준)
_SAMPLE_DIR = Path(__file__).parent.parent / "tests" / "step1_parser" / "sample_docs"
_OUTPUT_DIR = Path(__file__).parent.parent / "tests" / "step1_parser" / "output"
_TEST_DOCS = {
    "pdf_miraeasset_1q": _SAMPLE_DIR / "미래에셋증권 1분기 실적보고서.pdf",
    "pdf_miraeasset_4q": _SAMPLE_DIR / "미래에셋증권 4분기 실적보고서.pdf",
    "pdf_ds":            _SAMPLE_DIR / "DS투자증권 시황분석 리포트.pdf",
    "pdf_hanwha":        _SAMPLE_DIR / "한화투자증권 두산밥캣 기업분석 리포트.pdf",
}


def parse(pdf_path: Path) -> str:
    """PDF 파일을 파싱하여 Markdown 텍스트를 반환한다.

    1) 이미지 기반 PDF면 [이미지 기반 PDF — 텍스트 추출 불가] 반환
    2) pymupdf4llm으로 전체 텍스트·표를 Markdown으로 추출
    3) 결과가 비어있으면 pdfplumber fallback
    4) 차트 이미지 내 수치 → [차트 수치 미추출] 표기 (REQ-01 ③)
    """
    if _is_image_based(pdf_path):
        # TODO: OCR 전략 3/26 보고 후 결정. 현재는 명시적 메시지 반환.
        return f"[이미지 기반 PDF — 텍스트 추출 불가: {pdf_path.name}]"

    text = _parse_pymupdf4llm(pdf_path)

    if not text.strip():
        text = _parse_pdfplumber(pdf_path)

    return text


def _is_image_based(pdf_path: Path, sample_pages: int = 3) -> bool:
    """앞 페이지를 샘플링하여 이미지 기반 PDF 여부를 판별한다."""
    import fitz

    doc = fitz.open(str(pdf_path))
    pages_to_check = min(sample_pages, len(doc))
    empty_count = sum(
        1 for i in range(pages_to_check)
        if not doc[i].get_text().strip()
    )
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


if __name__ == "__main__":
    _OUTPUT_DIR.mkdir(exist_ok=True)

    for doc_key, doc_path in _TEST_DOCS.items():
        if not doc_path.exists():
            print(f"[SKIP] 문서 없음: {doc_path.name}")
            continue

        print(f"\n{'='*60}")
        print(f"📄 {doc_path.name}")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        result = parse(doc_path)
        elapsed = time.perf_counter() - t0

        out = _OUTPUT_DIR / f"{doc_key}__pdf_parse.md"
        out.write_text(result, encoding="utf-8")

        kor_chars = sum(1 for c in result if '가' <= c <= '힣')
        print(f"✅ {elapsed:.2f}s | 전체 {len(result):,}자 | 한글 {kor_chars:,}자 → {out.name}")
        print(f"미리보기: {result[:300].replace(chr(10), ' ')}...")
