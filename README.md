# doc-summary-agent

기업공시, 사업보고서, 산업분석 리포트 등 복잡한 구조의 문서를 파싱·요약하는 AI 에이전트

## 브랜치 전략

| 브랜치 | 용도 |
|--------|------|
| `main` | 최종 제출용 |
| `dev` | 개발 통합 |
| `feature/step1-parser` | Step1 파싱 구현 |
| `feature/step2-pipeline` | Step2 파이프라인 |
| `feature/step3-llm` | Step3 LLM 요약 |
| `feature/step4-ui` | Step4 UI 구현 |

## 커밋 컨벤션

| 태그 | 용도 |
|------|------|
| `feat:` | 새 기능 |
| `fix:` | 버그 수정 |
| `chore:` | 환경설정, 패키지 |
| `docs:` | 문서 |
| `refactor:` | 리팩토링 |
| `test:` | 테스트 |