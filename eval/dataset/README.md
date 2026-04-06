# 평가 데이터셋 구축 기준

## QA 쌍 구조

| 필드 | 설명 | 예시 |
|---|---|---|
| `id` | 고유 식별자 | `hanwha_001` |
| `doc` | 문서 파일명 | `한화투자증권_두산밥캣.pdf` |
| `source` | 포맷 | `pdf` / `hwp` / `doc` |
| `question` | 평가 질문 | `목표주가는?` |
| `answer` | 정답 (문서 직접 확인) | `80,000원` |
| `type` | 질문 유형 | `factual` / `numerical` / `summary` / `negative` |
| `section` | 답변 위치 섹션명 | `투자의견 유지` |

## 질문 유형 분류

| 유형 | 설명 |
|---|---|
| `factual` | 단순 사실 추출 |
| `numerical` | 수치 계산·비교 |
| `summary` | 핵심 내용 요약 |
| `multi_doc` | 복수 문서 비교 |
| `negative` | 문서에 없는 내용 질의 (할루시네이션 억제 확인) |

## 문서별 QA 수 계획 (총 ~40개)

| 문서 | 포맷 | 유형 | QA 수 |
|---|---|---|---|
| 한화투자증권 두산밥캣 | PDF (텍스트) | factual, numerical, summary, negative | 6 |
| DS투자증권 시황분석 | PDF (텍스트) | factual, summary, negative | 5 |
| 미래에셋 4Q | PDF (텍스트) | numerical, summary, multi_doc | 6 |
| 미래에셋 1Q | PDF (이미지/OCR) | factual, numerical | 4 |
| 미래에셋 2Q | PDF (이미지/OCR) | factual, numerical | 4 |
| 미래에셋 3Q | PDF (이미지/OCR) | factual, numerical | 4 |
| 농협 사업보고서 | HWP | factual, summary, negative | 5 |
| 금융감독원 보도자료 | DOC | factual, numerical, negative | 6 |

> `negative` 유형은 각 문서마다 1~2개 포함
