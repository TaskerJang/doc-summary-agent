"""
PDF 파서 비교 테스트
대상 라이브러리: pymupdf4llm / pdfplumber / PyPDF2 (pypdf)

⚠️  LLM·VLM 사용 금지 — 순수 Python 파싱만 사용 (REQ-01)
"""
import time
import traceback
from pathlib import Path
from config import DOCS, output_path

PDF_DOCS = {
    k: v for k, v in DOCS.items() if k.startswith("pdf_")
}


# ── 1. pymupdf4llm ────────────────────────────────────────────
def parse_pymupdf4llm(pdf_path: Path) -> str:
    import pymupdf4llm
    return pymupdf4llm.to_markdown(str(pdf_path))


# ── 2. pdfplumber ─────────────────────────────────────────────
def parse_pdfplumber(pdf_path: Path) -> str:
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


# ── 3. PyPDF2 (pypdf) ─────────────────────────────────────────
def parse_pypdf2(pdf_path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    return "\n".join(
        page.extract_text() or "" for page in reader.pages
    )


# ── 실행 ─────────────────────────────────────────────────────
def run():
    parsers = {
        "pymupdf4llm": parse_pymupdf4llm,
        "pdfplumber":  parse_pdfplumber,
        "PyPDF2":      parse_pypdf2,
    }

    for doc_key, doc_path in PDF_DOCS.items():
        if not doc_path.exists():
            print(f"[SKIP] 문서 없음: {doc_path.name}")
            continue

        print(f"\n{'='*60}")
        print(f"📄 {doc_path.name}")
        print(f"{'='*60}")

        for lib_name, parse_fn in parsers.items():
            print(f"  ▶ {lib_name} ... ", end="", flush=True)
            try:
                t0 = time.perf_counter()
                result = parse_fn(doc_path)
                elapsed = time.perf_counter() - t0

                out = output_path(lib_name, doc_key)
                out.write_text(result, encoding="utf-8")
                print(f"✅ {elapsed:.2f}s | {len(result):,}자 → {out.name}")

            except Exception as e:
                print(f"❌ 실패: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    run()
