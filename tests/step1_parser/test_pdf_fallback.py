"""
pdfplumber 페이지 단위 fallback — before/after 비교 테스트

[검증 포인트]
1. 페이지 커버리지  : fallback 전 스킵된 페이지 수 vs fallback 후 복구된 페이지 수
2. 텍스트 길이      : after >= before (fallback은 추가 방향이므로 줄어들면 이상)
3. 키워드 커버리지  : config.FALLBACK_KEYWORDS 중 after에서 추가로 발견된 키워드

실행:
    uv run python tests/step1_parser/test_pdf_fallback.py
    uv run python tests/step1_parser/test_pdf_fallback.py --doc pdf_miraeasset_1q
"""
import re
import sys
import time
import argparse
import traceback
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DOCS, OUTPUT_DIR

# 복구 여부를 확인할 키워드 (ESG 주요성과 등 벡터 그래픽 페이지에 자주 등장하는 패턴)
FALLBACK_KEYWORDS = [
    "ESG", "ROE", "ROA", "영업이익", "당기순이익", "BPS", "EPS",
    "자기자본", "배당", r"\d+\.\d+%", r"\d{1,3},\d{3}",  # 숫자 패턴
]

PDF_DOCS = {k: v for k, v in DOCS.items() if k.startswith("pdf_")}


# ── Before: 기존 방식 (텍스트 없는 페이지 스킵) ──────────────────────────────
def parse_before(pdf_path: Path) -> tuple[str, list[int]]:
    """기존 _parse_pymupdf4llm — 텍스트 없는 페이지를 단순 스킵"""
    import pymupdf4llm

    doc = fitz.open(str(pdf_path))
    text_page_indices = []
    skipped_indices = []

    for i, page in enumerate(doc):
        if page.get_text().strip():
            text_page_indices.append(i)
        else:
            skipped_indices.append(i)

    if not text_page_indices:
        return "", skipped_indices

    result = pymupdf4llm.to_markdown(
        doc,
        pages=text_page_indices,
        show_progress=False,
        table_strategy="lines_strict",
    )
    return result, skipped_indices


# ── After: 개선 방식 (텍스트 없는 페이지 → pdfplumber fallback) ──────────────
def parse_after(pdf_path: Path) -> tuple[str, list[int], list[int]]:
    """개선된 _parse_pymupdf4llm — 스킵 페이지에 pdfplumber fallback 적용"""
    import pymupdf4llm
    import pdfplumber

    doc = fitz.open(str(pdf_path))
    text_page_indices = []
    fallback_indices = []

    for i, page in enumerate(doc):
        if page.get_text().strip():
            text_page_indices.append(i)
        else:
            fallback_indices.append(i)

    page_texts: dict[int, str] = {}

    if text_page_indices:
        md = pymupdf4llm.to_markdown(
            doc,
            pages=text_page_indices,
            show_progress=False,
            table_strategy="lines_strict",
        )
        md_pages = md.split("\f") if "\f" in md else [md]
        for idx, content in zip(text_page_indices, md_pages):
            page_texts[idx] = content

    recovered_indices = []
    with pdfplumber.open(pdf_path) as pdf:
        for i in fallback_indices:
            if i >= len(pdf.pages):
                continue
            page = pdf.pages[i]
            lines = []
            text = page.extract_text()
            if text:
                lines.append(text)
            table_settings = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
            for table in page.extract_tables(table_settings):
                for row in table:
                    lines.append(" | ".join(cell or "" for cell in row))
            if lines:
                page_texts[i] = "\n".join(lines)
                recovered_indices.append(i)

    result = "\n\n".join(page_texts[i] for i in sorted(page_texts))
    return result, fallback_indices, recovered_indices


# ── 키워드 매칭 ───────────────────────────────────────────────────────────────
def find_keywords(text: str, keywords: list[str]) -> set[str]:
    found = set()
    for kw in keywords:
        if re.search(kw, text):
            found.add(kw)
    return found


# ── 리포트 출력 ───────────────────────────────────────────────────────────────
def report(doc_key: str, doc_path: Path):
    print(f"\n{'='*65}")
    print(f"📄 {doc_path.name}")
    print(f"{'='*65}")

    # Before
    t0 = time.perf_counter()
    before_text, skipped = parse_before(doc_path)
    before_time = time.perf_counter() - t0

    # After
    t0 = time.perf_counter()
    after_text, fallback_targets, recovered = parse_after(doc_path)
    after_time = time.perf_counter() - t0

    # 키워드 비교
    before_kw = find_keywords(before_text, FALLBACK_KEYWORDS)
    after_kw  = find_keywords(after_text,  FALLBACK_KEYWORDS)
    newly_found = after_kw - before_kw

    # 결과 저장
    (OUTPUT_DIR / f"{doc_key}__before_fallback.md").write_text(before_text, encoding="utf-8")
    (OUTPUT_DIR / f"{doc_key}__after_fallback.md").write_text(after_text,  encoding="utf-8")

    # 출력
    len_before, len_after = len(before_text), len(after_text)
    length_ok = "✅" if len_after >= len_before else "⚠️ "

    print(f"  [페이지 커버리지]")
    print(f"    스킵된 페이지 수  : {len(skipped)}페이지  {skipped}")
    print(f"    fallback 대상     : {len(fallback_targets)}페이지")
    print(f"    pdfplumber 복구   : {len(recovered)}페이지  {recovered}")

    print(f"\n  [텍스트 길이]")
    print(f"    before : {len_before:,}자  ({before_time:.2f}s)")
    print(f"    after  : {len_after:,}자  ({after_time:.2f}s)  {length_ok}")
    print(f"    증감   : {len_after - len_before:+,}자")

    print(f"\n  [키워드 커버리지]")
    print(f"    before 발견 : {sorted(before_kw)}")
    print(f"    after  발견 : {sorted(after_kw)}")
    if newly_found:
        print(f"    ✅ fallback으로 새로 발견 : {sorted(newly_found)}")
    else:
        print(f"    ➖ 추가 발견 키워드 없음 (이미 모두 커버되거나 해당 없음)")

    print(f"\n  결과 저장 → output/{doc_key}__before_fallback.md")
    print(f"             output/{doc_key}__after_fallback.md")


# ── 진입점 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", help="특정 doc_key만 실행 (예: pdf_miraeasset_1q)")
    args = parser.parse_args()

    targets = (
        {args.doc: PDF_DOCS[args.doc]} if args.doc and args.doc in PDF_DOCS
        else PDF_DOCS
    )

    for doc_key, doc_path in targets.items():
        if not doc_path.exists():
            print(f"\n[SKIP] 문서 없음: {doc_path.name}")
            continue
        try:
            report(doc_key, doc_path)
        except Exception as e:
            print(f"\n[ERROR] {doc_key}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
