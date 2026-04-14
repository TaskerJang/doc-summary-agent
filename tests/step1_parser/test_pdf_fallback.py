"""
pdfplumber fallback + EasyOCR 선택적 OCR — before/after 비교 테스트

[검증 포인트]
1. 이미지 마커 수   : pymupdf4llm이 생략한 이미지 페이지 수
2. 텍스트 길이      : after >= before (OCR 보강은 추가 방향)
3. 키워드 커버리지  : FALLBACK_KEYWORDS 중 after에서 추가로 발견된 키워드
4. OCR 캐시         : .ocr_cache/ 에 페이지 단위 캐시 생성 여부

실행:
    uv run python tests/step1_parser/test_pdf_fallback.py
    uv run python tests/step1_parser/test_pdf_fallback.py --doc pdf_miraeasset_4q
"""
import re
import sys
import time
import argparse
import traceback
from pathlib import Path

import fitz
import pymupdf4llm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DOCS, OUTPUT_DIR

FALLBACK_KEYWORDS = [
    "ESG", "ROE", "ROA", "영업이익", "당기순이익", "BPS", "EPS",
    "자기자본", "배당", r"\d+\.\d+%", r"\d{1,3},\d{3}",
]
_OMITTED_MARKER = "intentionally omitted"

PDF_DOCS = {k: v for k, v in DOCS.items() if k.startswith("pdf_")}


# ── Before: 기존 방식 ─────────────────────────────────────────────────────────
def parse_before(pdf_path: Path) -> tuple[str, int]:
    """기존 방식 — 텍스트 없는 페이지 스킵, 이미지 임베딩 그대로 방치"""
    doc = fitz.open(str(pdf_path))
    text_page_indices = [i for i, p in enumerate(doc) if p.get_text().strip()]

    if not text_page_indices:
        return "", 0

    md = pymupdf4llm.to_markdown(
        doc,
        pages=text_page_indices,
        show_progress=False,
        table_strategy="lines_strict",
    )
    omitted_count = md.count(_OMITTED_MARKER)
    return md, omitted_count


# ── After: 개선된 doc_parser.pdf.parse 직접 호출 ─────────────────────────────
def parse_after(pdf_path: Path) -> tuple[str, int]:
    """
    개선된 파서 — doc_parser.pdf.parse() 직접 호출:
    - 이미지 임베딩 페이지 → EasyOCR 보강
    - get_text() 빈 페이지 → pdfplumber fallback
    - 그래도 빈 경우 → EasyOCR 페이지 단위 OCR
    """
    from doc_parser.pdf import parse
    result = parse(pdf_path)
    omitted_count = result.count(_OMITTED_MARKER)
    return result, omitted_count


# ── 키워드 매칭 ───────────────────────────────────────────────────────────────
def find_keywords(text: str, keywords: list[str]) -> set[str]:
    found = set()
    for kw in keywords:
        if re.search(kw, text):
            found.add(kw)
    return found


# ── OCR 캐시 확인 ─────────────────────────────────────────────────────────────
def count_ocr_cache(pdf_path: Path) -> int:
    """해당 PDF에 대해 생성된 페이지 단위 OCR 캐시 파일 수"""
    import hashlib
    h = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]
    cache_dir = Path(".ocr_cache")
    return len(list(cache_dir.glob(f"{h}_p*.json"))) if cache_dir.exists() else 0


# ── 리포트 출력 ───────────────────────────────────────────────────────────────
def report(doc_key: str, doc_path: Path):
    print(f"\n{'='*65}")
    print(f"📄 {doc_path.name}")
    print(f"{'='*65}")

    # Before
    t0 = time.perf_counter()
    before_text, before_omitted = parse_before(doc_path)
    before_time = time.perf_counter() - t0

    # After
    t0 = time.perf_counter()
    after_text, after_omitted = parse_after(doc_path)
    after_time = time.perf_counter() - t0

    # 키워드 비교
    before_kw = find_keywords(before_text, FALLBACK_KEYWORDS)
    after_kw  = find_keywords(after_text,  FALLBACK_KEYWORDS)
    newly_found = after_kw - before_kw

    # 결과 저장
    (OUTPUT_DIR / f"{doc_key}__before_fallback.md").write_text(before_text, encoding="utf-8")
    (OUTPUT_DIR / f"{doc_key}__after_fallback.md").write_text(after_text,  encoding="utf-8")

    len_before, len_after = len(before_text), len(after_text)
    length_ok = "✅" if len_after >= len_before else "⚠️ "

    ocr_cache_count = count_ocr_cache(doc_path)

    print(f"  [이미지 처리]")
    print(f"    before 이미지 생략 수  : {before_omitted}개 (intentionally omitted)")
    print(f"    after  이미지 생략 수  : {after_omitted}개  {'✅ 감소' if after_omitted < before_omitted else '➖ 동일'}")
    print(f"    OCR 캐시 생성          : {ocr_cache_count}페이지")

    print(f"\n  [텍스트 길이]")
    print(f"    before : {len_before:,}자  ({before_time:.2f}s)")
    print(f"    after  : {len_after:,}자  ({after_time:.2f}s)  {length_ok}")
    print(f"    증감   : {len_after - len_before:+,}자")

    print(f"\n  [키워드 커버리지]")
    print(f"    before 발견 : {sorted(before_kw)}")
    print(f"    after  발견 : {sorted(after_kw)}")
    if newly_found:
        print(f"    ✅ OCR로 새로 발견 : {sorted(newly_found)}")
    else:
        print(f"    ➖ 추가 발견 없음")

    print(f"\n  결과 저장 → output/{doc_key}__before_fallback.md")
    print(f"             output/{doc_key}__after_fallback.md")


# ── 진입점 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", help="특정 doc_key만 실행 (예: pdf_miraeasset_4q)")
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
