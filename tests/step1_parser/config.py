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
    # PDF — 미래에셋증권 실적보고서 (1~4분기)
    "pdf_miraeasset_1q": SAMPLE_DIR / "미래에셋증권 1분기 실적보고서.pdf",
    "pdf_miraeasset_2q": SAMPLE_DIR / "미래에셋증권 2분기 실적보고서.pdf",
    "pdf_miraeasset_3q": SAMPLE_DIR / "미래에셋증권 3분기 실적보고서.pdf",
    "pdf_miraeasset_4q": SAMPLE_DIR / "미래에셋증권 4분기 실적보고서.pdf",
    # PDF — 리포트
    "pdf_ds":            SAMPLE_DIR / "DS투자증권 시황분석 리포트.pdf",
    "pdf_hanwha":        SAMPLE_DIR / "한화투자증권 두산밥캣 기업분석 리포트.pdf",
    # DOC
    "doc_fss":           SAMPLE_DIR / "금융감독원 251125_(보도자료) 25.10월중 기업의 직접금융 조달실적.doc",
    # HWP
    "hwp_nonghyup":      SAMPLE_DIR / "농협 2022년 9월말 기준 사업보고서.hwp",
}

# 결과 저장 경로
def output_path(lib_name: str, doc_key: str, ext: str = "md") -> Path:
    return OUTPUT_DIR / f"{doc_key}__{lib_name}.{ext}"
