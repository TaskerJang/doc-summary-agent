# doc-summary-agent

기업공시, 사업보고서, 산업분석 리포트 등 복잡한 구조의 문서를 파싱·요약하는 AI 에이전트

## 주요 기능

- **다중 포맷 파싱** — PDF / DOCX / DOC / HWP 지원
- **이미지 기반 PDF OCR** — EasyOCR(dpi=300) 기반 자동 감지 및 처리
- **노이즈 제거 전처리** — 헤더·푸터·면책고지 등 자동 필터링
- **Markdown 구조 기반 청킹** — 섹션/단락/표 단위 분할
- **Map-Reduce 요약** — GPT-5.2 기반 청크 병렬 요약 → 전체 요약
- **BM25 기반 Q&A** — 원문 청크 BM25 검색 + 요약 섹션 결합 context 구성
- **ChatGPT 스타일 UI** — Chainlit 기반 채팅 인터페이스 (스트리밍, 소스 인용, 추천 질문)
- **평가 파이프라인** — ROUGE / 수치 정확도 / Faithfulness / Completeness 자동 평가

## 평가 결과 (prompt_v2_bm25, chunk_size=500, overlap=50, QA 40개)

| 지표 | 값 |
|------|---|
| ROUGE-L | 0.2929 |
| 수치 정확도 | 0.7102 |
| Faithfulness | 31/40 (77.5%) |
| Completeness | 1.25 / 5 |
| Conciseness | 3.90 / 5 |

## 기술 스택

| 구분 | 라이브러리 |
|------|-----------|
| 파싱 | pymupdf4llm, pdfplumber, python-docx, pyhwp, EasyOCR |
| 청킹 | LangChain MarkdownTextSplitter |
| 요약 / Q&A | OpenAI GPT-5.2, rank-bm25, tenacity |
| UI | Chainlit |
| 평가 | rouge-score, bert-score, python-mecab-ko |
| 데이터 검증 | Pydantic |
| 패키지 관리 | uv |

## 프로젝트 구조

```

doc-summary-agent/
├── doc_parser/         # Step 1: 파싱 + 전처리 + 메타데이터
│   ├── ocr_cache.py    # EasyOCR 캐시 기반 처리
│   └── preprocessor.py # 노이즈 제거 전처리
├── chunker/            # Step 2: Markdown 구조 기반 청킹
├── summarizer/         # Step 3: LLM 요약 + BM25 Q&A
│   ├── llm.py          # Map-Reduce 요약
│   ├── qa.py           # BM25 기반 Q&A
│   └── prompts/        # 프롬프트 템플릿 (.md)
├── ui/                 # Step 4: Chainlit UI 컴포넌트
├── eval/               # 평가 파이프라인
│   ├── dataset/        # QA 데이터셋 (qa_pairs.json, 40개)
│   ├── metrics/        # ROUGE / 수치 정확도 / Faithfulness
│   ├── results/        # 평가 결과 JSON
│   ├── run_eval.py     # 평가 실행 진입점
│   └── generate_leaderboard.py
├── tests/              # 테스트
├── main.py             # CLI 파이프라인 진입점
└── run_chainlit.py     # UI 실행 진입점

```

## 실행 방법

**환경 설정**
```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
```

**UI 실행**
```bash
uv run chainlit run run_chainlit.py
```

**CLI 실행**
```bash
# 전체 파이프라인
uv run python main.py "sample_docs/리포트.pdf"

# Step 1~2만 실행 (요약 생략)
uv run python main.py "sample_docs/리포트.pdf" --no-summary

# JSON 출력
uv run python main.py "sample_docs/리포트.pdf" --json
```

**평가 실행**
```bash
# 전체 평가 (chunk_size=500, overlap=50)
uv run python eval/run_eval.py --chunk-size 500 --chunk-overlap 50 --tag full

# 특정 문서만
uv run python eval/run_eval.py --doc 리포트.pdf --tag test

# 리더보드 생성
uv run python eval/generate_leaderboard.py --result eval/results/eval_results_*.json
```

## 브랜치 전략

| 브랜치 | 용도 |
|--------|------|
| `main` | 최종 제출용 |
| `dev` | 개발 통합 |
| `feature/step1-parser` | Step 1 파싱 구현 |
| `feature/step2-pipeline` | Step 2 파이프라인 |
| `feature/step3-llm` | Step 3 LLM 요약 |
| `feature/step4-chainlit-ui` | Step 4 Chainlit UI |
| `feature/step5-eval` | Step 5 평가 파이프라인 |

## 커밋 컨벤션

| 태그 | 용도 |
|------|------|
| `feat:` | 새 기능 |
| `fix:` | 버그 수정 |
| `chore:` | 환경설정, 패키지 |
| `docs:` | 문서 |
| `refactor:` | 리팩토링 |
| `test:` | 테스트 |