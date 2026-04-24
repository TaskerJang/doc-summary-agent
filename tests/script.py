import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# scripts/check_phase1.py (로컬 작업용, 커밋 안 해도 됨)
import re
from pathlib import Path
import doc_parser
from doc_parser.preprocessor import clean
from chunker.chunker import chunk

SAMPLES = [
    "tests/step1_parser/sample_docs/DS투자증권_시황분석_리포트.pdf",
    "tests/step1_parser/sample_docs/미래에셋증권_1분기_실적보고서.pdf",
    "tests/step1_parser/sample_docs/미래에셋증권_2분기_실적보고서.pdf",
    "tests/step1_parser/sample_docs/미래에셋증권_3분기_실적보고서.pdf",
    "tests/step1_parser/sample_docs/미래에셋증권_4분기_실적보고서.pdf",
    "tests/step1_parser/sample_docs/한화투자증권_두산밥캣_기업분석_리포트.pdf",
    "tests/step1_parser/sample_docs/금융감독원_251125__보도자료__25_10월중_기업의_직접금융_조달실적.docx",
    "tests/step1_parser/sample_docs/금융감독원 251125_(보도자료) 25.10월중 기업의 직접금융 조달실적.doc",
    "tests/step1_parser/sample_docs/농협_2022년_9월말_기준_사업보고서.hwp",
]

MARKER_RE = re.compile(r"<!--PAGE:(\d+)-->")

def source_from_path(p: Path) -> str:
    ext, stem = p.suffix.lower(), p.stem.lower()
    if ext == ".pdf":
        return "ir_report" if any(k in stem for k in ["실적", "분기", "연간", "ir"]) else "pdf"
    if ext in (".docx", ".doc"): return "docx"
    if ext == ".hwp": return "hwp"
    return "generic"

for path_str in SAMPLES:
    p = Path(path_str)
    if not p.exists():
        print(f"SKIP (없음): {p.name}")
        continue
    print(f"\n{'='*70}\n{p.name}\n{'='*70}")

    # Step 1: 파싱
    raw = doc_parser.parse(p)
    raw_markers = MARKER_RE.findall(raw)
    print(f"파싱 후     : {len(raw):>7,}자  마커 {len(raw_markers):>3}개  "
          f"페이지 {min(map(int, raw_markers)) if raw_markers else '-'}~"
          f"{max(map(int, raw_markers)) if raw_markers else '-'}")

    # Step 1.5: clean
    src = source_from_path(p)
    cleaned = clean(raw, source=src)
    clean_markers = MARKER_RE.findall(cleaned)
    loss = len(raw_markers) - len(clean_markers)
    print(f"clean 후    : {len(cleaned):>7,}자  마커 {len(clean_markers):>3}개  "
          f"source={src}  손실 {loss}개")

    # Step 2: 청킹
    chunks = chunk(cleaned)
    with_page = sum(1 for c in chunks if c["page"] is not None)
    with_range = sum(1 for c in chunks if c["page_end"] is not None)
    any_marker_left = any(MARKER_RE.search(c["text"]) for c in chunks)
    print(f"청킹 후     : {len(chunks):>3}개  "
          f"page 있음 {with_page}개  범위 청크 {with_range}개  "
          f"청크 내 마커 잔존: {'❌ YES' if any_marker_left else '✓ NO'}")

    # 샘플 청크 3개 출력
    for c in chunks[:3]:
        page_str = f"p.{c['page']}" + (f"-{c['page_end']}" if c['page_end'] else "")
        print(f"  [{c['chunk_index']:02d}] {page_str:>8}  {c['section'][:25]:25s}  "
              f"{c['text'][:40]!r}")