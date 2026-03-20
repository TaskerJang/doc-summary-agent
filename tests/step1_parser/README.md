# Step1 파서 라이브러리 비교 테스트

> Issue #3 · [Step1] 파서 라이브러리 비교·선정

## 폴더 구조

```
tests/step1_parser/
├── README.md              # 이 파일
├── config.py              # 테스트 문서 경로 및 공통 설정
├── test_pdf_parsers.py    # PDF 파서 비교 (pymupdf4llm / pdfplumber / PyPDF2)
├── test_docx_parsers.py   # DOCX 파서 비교 (python-docx / docx2python)
├── test_hwp_parsers.py    # HWP 파서 비교 (libhwp / python-hwpx)
└── run_all.py             # 전체 실행 + 결과 요약 출력
```

## 실행 방법

```bash
# 1. 의존성 설치 (uv 사용)
uv add pymupdf4llm pdfplumber pypdf python-docx docx2python

# 2. 테스트 문서를 tests/step1_parser/sample_docs/ 에 위치시키기
#    (gitignore 처리 — 실제 문서는 로컬에만 보관)

# 3. 전체 실행
uv run python tests/step1_parser/run_all.py

# 또는 개별 실행
uv run python tests/step1_parser/test_pdf_parsers.py
```

## 평가 기준 (기능정의서 v0.1)

| 항목 | 가중치 | 관련 요건 |
|------|-------|---------|
| 표 구조 보존 정확도 | 20% | REQ-01 ① |
| 텍스트 블록 분리 정확도 | 15% | REQ-01 ① |
| 다단 레이아웃 읽기 순서 | 15% | REQ-01 ② |
| 헤더·푸터 노이즈 제거 | 10% | REQ-05 ①② |
| 이미지·캡션 추출 | 5% | REQ-01 ③ |
| 단락·제목 구조 보존 | 10% | REQ-02 ② |
| 표 추출 정확도 (DOCX) | 5% | REQ-02 ① |
| HWP 텍스트 추출 | 10% | REQ-03 ① |
| 처리 속도 및 안정성 | 5% | REQ-06 |
| 다중 포맷 지원 여부 | 5% | Issue #3 |

> ⚠️ 점수(0~5)는 파싱 결과를 **직접 육안 검토** 후 `파서_라이브러리_비교평가표_v1.0.xlsx` 에 기입
