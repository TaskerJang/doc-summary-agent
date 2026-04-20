# doc-summary-agent

기업공시, 사업보고서, 산업분석 리포트 등 복잡한 구조의 문서를 파싱·요약하고 질의응답하는 AI 에이전트

## 주요 기능

- **다중 포맷 파싱** — PDF / DOCX / DOC / HWP 지원
- **이미지 기반 PDF OCR** — EasyOCR(dpi=300) 기반 자동 감지 및 처리
- **노이즈 제거 전처리** — 헤더·푸터·면책고지 등 자동 필터링
- **Markdown 구조 기반 청킹** — 섹션/단락/표 단위 분할
- **Map-Reduce 요약** — GPT-5.2 기반 청크 병렬 요약 → 전체 요약
- **하이브리드 검색 Q&A** — BM25(sparse) + bge-m3(dense) + RRF 융합 + bge-reranker-v2-m3 재정렬
- **섹션별 자동 차트** — 수치 데이터를 Plotly로 자동 렌더링 (`ChatSettings`로 on/off·개수 조정)
- **추천 질문 버튼** — 요약 결과 기반으로 후속 질문 3개 자동 생성 (`cl.Action`)
- **PDF 원문 사이드 패널** — 업로드한 PDF를 UI 오른쪽에서 바로 확인
- **세션 히스토리 & 인증** — Password 로그인 후 과거 대화를 사이드바에서 재진입
- **Human Feedback** — 메시지별 thumbs up/down 수집 (SQLite 저장)
- **ChatGPT 스타일 UI** — Chainlit 기반 채팅 인터페이스 (스트리밍, 소스 인용, TaskList 진행 표시)
- **평가 파이프라인** — ROUGE / 수치 정확도 / Faithfulness / Completeness 자동 평가

## 평가 결과 (prompt_v2_bm25, chunk_size=500, overlap=50, QA 40개)

| 지표 | 값 |
|------|---|
| ROUGE-L | 0.2929 |
| 수치 정확도 | 0.7102 |
| Faithfulness | 31/40 (77.5%) |
| Completeness | 1.25 / 5 |
| Conciseness | 3.90 / 5 |

> 평가 기준은 BM25 단독 시절의 수치. 하이브리드 검색 도입 이후 재측정 예정.

## 기술 스택

| 구분 | 라이브러리 |
|------|-----------|
| 파싱 | pymupdf4llm, pdfplumber, python-docx, pyhwp, EasyOCR |
| 청킹 | LangChain MarkdownTextSplitter |
| 검색 | rank-bm25 (sparse), bge-m3 (dense), Qdrant (벡터 DB), bge-reranker-v2-m3 |
| 요약 / Q&A | OpenAI GPT-5.2, tenacity |
| UI | Chainlit 2.11, Plotly |
| 세션 영속화 | Chainlit `SQLAlchemyDataLayer` + SQLite(aiosqlite) |
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
├── summarizer/         # Step 3: LLM 요약 + 하이브리드 검색 Q&A
│   ├── llm.py          # Map-Reduce 요약
│   ├── embedder.py     # bge-m3 + Qdrant 인덱싱
│   ├── qa.py           # BM25 + dense + RRF + reranker Q&A
│   ├── chart_router.py # chart_spec → Plotly figure 변환
│   └── prompts/        # 프롬프트 템플릿 (.md)
├── ui/                 # Step 4: Chainlit UI 컴포넌트
│   └── app.py          # 진입점 (업로드 → 요약 → 차트 → PDF → 추천질문 → Q&A)
├── eval/               # 평가 파이프라인
│   ├── dataset/        # QA 데이터셋 (qa_pairs.json, 40개)
│   ├── metrics/        # ROUGE / 수치 정확도 / Faithfulness
│   ├── results/        # 평가 결과 JSON
│   ├── run_eval.py     # 평가 실행 진입점
│   └── generate_leaderboard.py
├── tests/              # 테스트
├── main.py             # CLI 파이프라인 진입점
└── run_chainlit.py     # UI 실행 진입점 (SQLAlchemyDataLayer 주입 + DB 스키마 초기화)

```

## 실행 방법

### 1. 환경 설정
```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
```

### 2. 의존 서비스 실행 (Qdrant)

하이브리드 검색의 dense 인덱싱은 Qdrant 벡터 DB를 사용한다. Docker로 띄워두자.

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  --name qdrant qdrant/qdrant
```

Qdrant가 없으면 인덱싱은 백그라운드에서 실패로그를 남기고, BM25 단독으로 Q&A가 동작한다 (기능은 유지되지만 정확도 하락).

### 3. UI 실행

```bash
# 계정 환경변수 설정 (기본값: admin / 1234!)
export APP_USERNAME=admin
export APP_PASSWORD=your-password

uv run chainlit run run_chainlit.py
```

브라우저에서 `http://localhost:8000` 접속 → 로그인 → 문서 업로드.

> Windows CMD에서는 `export` 대신 `set APP_USERNAME=admin` / `set APP_PASSWORD=your-password` 사용.
> PowerShell에서는 `$env:APP_USERNAME = "admin"`.

### 4. CLI 실행 (UI 없이)
```bash
# 전체 파이프라인
uv run python main.py "sample_docs/리포트.pdf"

# Step 1~2만 실행 (요약 생략)
uv run python main.py "sample_docs/리포트.pdf" --no-summary

# JSON 출력
uv run python main.py "sample_docs/리포트.pdf" --json
```

### 5. 평가 실행
```bash
# 전체 평가 (chunk_size=500, overlap=50)
uv run python eval/run_eval.py --chunk-size 500 --chunk-overlap 50 --tag full

# 특정 문서만
uv run python eval/run_eval.py --doc 리포트.pdf --tag test

# 리더보드 생성
uv run python eval/generate_leaderboard.py --result eval/results/eval_results_*.json
```

## 세션 히스토리 / 인증

- 로그인 성공 시 `chainlit.db`(SQLite)에 user/thread/step이 기록된다.
- 왼쪽 사이드바에 과거 대화 목록이 표시되며, 클릭 시 thread 내용이 로드된다.
- **현재 한계**: 과거 thread 재진입 시 차트/PDF는 복원되지 않는다 (blob_storage 미설정). 자세한 내용은 이슈 #101 참고.
- **resume 시 세션 상태 복원**(`@cl.on_chat_resume`)은 이슈 #99에서 후속 작업 중.

## 브랜치 전략

| 브랜치 | 용도 |
|--------|------|
| `main` | 최종 제출용 |
| `dev` | 개발 통합 (기본 브랜치) |
| `feat/{이슈번호}-{kebab-case-설명}` | 기능 단위 브랜치 (예: `feat/70-chat-history`) |
| `fix/...` / `docs/...` / `refactor/...` | 용도별 접두어 |

모든 작업 브랜치는 `dev`를 base로 생성하고, PR로 머지한다.

## 커밋 컨벤션

[Conventional Commits](https://www.conventionalcommits.org/) 기반의 `태그(스코프): 설명` 형식.

| 태그 | 용도 |
|------|------|
| `feat` | 새 기능 |
| `fix` | 버그 수정 |
| `perf` | 성능 개선 |
| `refactor` | 리팩토링 |
| `revert` | 이전 커밋 되돌리기 |
| `docs` | 문서 |
| `test` | 테스트 |
| `chore` | 환경설정, 패키지 |

예시
```
feat(ui): #70 cl.Pdf(display="side") 제거 — SQLAlchemy + blob_storage 미설정 시 create_element 블로킹 방지
fix(run): list_threads PaginatedResponse 반환 — to_dict() 에러 수정
perf(ui): stream_token 한 글자 → 줄 단위로 전환
```
