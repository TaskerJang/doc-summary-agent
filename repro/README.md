# repro/ — Issue #117 재현 실험

`cl.Pdf(display="side")`가 링크 클릭 없이 자동으로 펼쳐지는 버그(#117)의
진짜 원인을 격리 검증하기 위한 최소 Chainlit 앱.

## 1. 실행 방법

```cmd
:: 1) 샘플 PDF 준비 (이 파일은 .gitignore로 제외)
copy "C:\path\to\any.pdf" repro\sample.pdf

:: 2) Chainlit 환경 확인
uv run pip show chainlit | findstr Version

:: 3) repro 앱 실행 (기존 앱과 포트 분리)
uv run chainlit run repro/repro_pdf_side.py -w --port 8001
```

브라우저에서 `http://localhost:8001` 접속 → 로그인 (기존 앱과 동일 계정) →
8개 케이스 버튼을 **한 번씩만** 클릭.

## 2. 관찰 포인트

각 케이스 버튼을 눌렀을 때 다음 중 어느 쪽인지 확인:

| 결과 | 의미 |
|------|------|
| **🔴 자동 열림** | 메시지가 표시되는 즉시 사이드/페이지 패널이 열린다 (링크 클릭 전) |
| **🟢 정상** | 공식 문서대로, 링크를 클릭해야 열린다 |
| **⚫ 미표시** | 엘리먼트 자체가 UI에 나타나지 않는다 (mention=False side 케이스 기대값) |

> ⚠️ 세션 간 캐시 영향을 피하려면 **각 케이스마다 페이지 새로고침 후 새 세션**으로
> 테스트하거나, 최소한 브라우저 localStorage를 초기화하세요.

## 3. 결과 기록표

실제 테스트 후 아래 표를 채워서 다음 단계 판단에 사용.

| 케이스 | display | page | mention | 관찰 결과 | 비고 |
|--------|---------|------|---------|----------|------|
| A1     | side    | None | True    | ?         |      |
| A2     | side    | None | False   | ?         |      |
| A3     | side    | 1    | True    | ?         | ← 현재 프로덕션 설정 |
| A4     | side    | 1    | False   | ?         |      |
| B1     | inline  | None | -       | ?         |      |
| B2     | inline  | 1    | -       | ?         |      |
| C1     | page    | None | -       | ?         |      |
| C2     | page    | 1    | -       | ?         |      |

## 4. 해석 시나리오

결과 조합에 따라 다음 결정 트리를 따른다:

| 시나리오 | 조건 | 결정 |
|---------|------|------|
| S1 | A3만 자동 열림, A1은 정상 | **page 파라미터가 범인** → ui/app.py에서 `page=1` 제거로 해결 (D안 변형) |
| S2 | A1, A3 모두 자동 열림 | `display="side"` 자체가 버그 → A/B/C안 중 선택 필요 |
| S3 | B1, B2 (inline) 정상 | inline 전환 고려 (D안 변형) — 단 사이드 패널 UX는 포기 |
| S4 | 모든 케이스 정상 | 자동 열림은 프로덕션 앱의 **다른 코드**와의 상호작용이 원인 → ui/app.py 재현 필요 |

## 5. Chainlit 버전 참고

```cmd
uv run pip show chainlit
```

`Version:` 줄을 기록해 두면 상류 이슈 검색 시 유용:
- https://github.com/Chainlit/chainlit/issues
- 검색 키워드: `pdf side auto open`, `element display side click`

> **한 줄 요약**: 8개 케이스 모두 돌려 결과표를 채운 뒤, 시나리오 S1~S4 중 어느
> 쪽인지 확정되면 그에 맞는 수정안(A~D)으로 넘어간다.
