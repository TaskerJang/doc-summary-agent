# repro/ — Issue #117 재현 실험

`cl.Pdf(display="side")`가 링크 클릭 없이 자동으로 펼쳐지는 버그(#117)의
진짜 원인을 격리 검증하기 위한 최소 Chainlit 앱.

## 0. 환경 정보 (확인 완료)

| 항목 | 값 |
|------|-----|
| Chainlit 버전 | **2.11.0** |
| OS | Windows (CMD) |
| 런처 | uv |
| 버전 확인 명령 | `uv pip show chainlit` (NOT `uv run pip show` — uv venv는 pip 실행파일 없음) |

### 2.11.0에서 주목할 변경사항

릴리스 노트 기준 (PR 번호는 Chainlit 본체 리포):

- **PR #2833** `feat: added improved pdf viewer` — iframe을 제거하고 built-in
  pdfjs-dist 기반 뷰어로 **완전 교체**. 이번 이슈의 직접 원인 후보.
- **PR #2876** `declare pdfjs-dist as explicit frontend dependency` — 위 교체의
  종속성 선언.
- **PR #2891** `fix(socket): guard on_chat_start against duplicate dispatch on
  reconnect` — "메시지 3번 반복" 증상의 원인이었을 가능성. 2.11.0에 포함되어
  해결됐을 수 있으나 잔여 케이스 재현 여부 확인 필요.

### 기존에 확인된 관련 버그 (참고용)

- **Chainlit issue #846** — 대화 전환 시 사이드 패널이 닫히지 않음
- **Chainlit issue #1559** — display="side"/"page" 엘리먼트가 아예 표시 안 됨
- **Chainlit issue #1827** — CustomElement의 display="side"/"page" 접근 불가

## 1. 실행 방법

```cmd
:: 1) 샘플 PDF 준비 (이 파일은 .gitignore로 제외)
copy "C:\path\to\any.pdf" repro\sample.pdf

:: 2) Chainlit 환경 확인
uv pip show chainlit | findstr Version

:: 3) repro 앱 실행 (기존 앱과 포트 분리, watch 모드 OFF)
uv run chainlit run repro/repro_pdf_side.py --port 8001
```

> ⚠️ `-w` (watch) 옵션은 **빼는 것을 권장**. repro 중 파일 변경 감지로
> 세션이 재시작되면 자동 펼침 타이밍 관찰이 오염됨.

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
| S2 | A1, A3 모두 자동 열림 | **display="side" + 2.11.0 built-in viewer 회귀** → 상류 이슈 보고 + 우리 쪽 우회 필요 (A/B/C안 선택) |
| S3 | B1, B2 (inline) 정상 | inline 전환 고려 (D안 변형) — 단 사이드 패널 UX는 포기 |
| S4 | 모든 케이스 정상 | 자동 열림은 프로덕션 앱의 **다른 코드**와의 상호작용이 원인 → ui/app.py 재현 필요 |

## 5. 메시지 중복 (3번 반복) 증상 별도 확인

이 repro 앱은 자동 펼침만 격리하고, 중복 생성 증상은 다루지 않는다. 중복 증상은
프로덕션 앱(ui/app.py)의 `on_open_pdf` callback에 가드가 없어서 발생했을 가능성이
크다. Chainlit 2.10.1에 <br>`fix(socket): guard on_chat_start against duplicate dispatch on reconnect`<br>
패치가 포함되어 상류에서도 유사 이슈가 알려져 있으니, 로컬 저장소의 `.chainlit/`을
지우고 재현되는지도 확인할 가치가 있다.

## 6. 다음 단계

1. 위 1~3 실행 및 결과 기록표 채우기
2. 시나리오 S1~S4 중 매칭되는 것 확인
3. 매칭 결과를 PR 코멘트 혹은 다음 세션 인풋으로 공유 → 그에 맞는 ui/app.py 수정

> **한 줄 요약**: Chainlit 2.11.0에서 iframe → pdfjs-dist로 PDF 뷰어가 교체된
> 것이 원인 후보 1순위. 8케이스 결과표 채운 뒤 시나리오 S1~S4로 판단.
