"""
테스트 공통 설정 — 문서 경로 및 출력 디렉토리
⚠️  sample_docs/ 는 .gitignore 처리되어 있음. 로컬에만 보관.
"""
from pathlib import Path

BASE_DIR    = Path(__file__).parent
SAMPLE_DIR  = BASE_DIR / "sample_docs"
OUTPUT_DIR  = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 테스트 문서 (기능정의서 Input 정의 기준) ─────────────────
DOCS = {
    "pdf_miraeasset":  SAMPLE_DIR / "미래에셋증권_실적보고서.pdf",
    "pdf_ds":          SAMPLE_DIR / "DS투자증권_시황분석리포트.pdf",
    "pdf_hanwha":      SAMPLE_DIR / "한화투자증권_두산밥캣_기업분석.pdf",
    "doc_fss":         SAMPLE_DIR / "금융감독원_보도자료.doc",
    "hwp_nonghyup":    SAMPLE_DIR / "농협_2022년_사업보고서.hwp",
}

# 결과 저장 경로
def output_path(lib_name: str, doc_key: str, ext: str = "md") -> Path:
    return OUTPUT_DIR / f"{doc_key}__{lib_name}.{ext}"
