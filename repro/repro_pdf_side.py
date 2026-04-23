"""repro/repro_pdf_side.py
cl.Pdf(display="side") 자동 펼침 이슈 (#117) 최소 재현 앱.

왜 이 파일이 필요한가:
---------------------
ui/app.py는 TaskList, 스트리밍, 백그라운드 인덱싱, reranker 워밍업,
Plotly 차트, SourceReference 커스텀 엘리먼트 등 수많은 요소가 뒤섞여
있어 cl.Pdf 단독 동작을 격리 검증하기 어렵다. 이 스크립트는 cl.Pdf만
단독으로 전송하는 가장 얇은 Chainlit 앱으로, 3개 변수를 분리해서
자동 펼침의 진짜 트리거가 무엇인지 찾기 위한 실험 도구다.

테스트 매트릭스 (3변수 × 2값 = 8케이스):
----------------------------------------
- display   : "side" / "inline" / "page"
- page      : None   / 1
- mention   : content에 엘리먼트 name 포함 여부 (True / False)

    케이스  display   page   mention  기대 동작 (공식 문서)
    ------- --------  -----  -------  --------------------
    A1      side      None   True     클릭 시에만 열림
    A2      side      None   False    엘리먼트 미표시
    A3      side      1      True     클릭 시에만 열림
    A4      side      1      False    엘리먼트 미표시
    B1      inline    None   -        인라인 내장 렌더
    B2      inline    1      -        인라인 내장 렌더
    C1      page      None   -        풀페이지 오버레이
    C2      page      1      -        풀페이지 오버레이

각 케이스별로 "클릭 전에 사이드/페이지가 열리는지" 를 눈으로 확인한다.
관찰 결과는 repro/README.md의 결과표에 손으로 기록.

실행:
----
    uv run chainlit run repro/repro_pdf_side.py -w --port 8001

(기존 앱과 포트 충돌 방지 위해 8001 권장)

사전 준비:
---------
repro/sample.pdf 에 아무 PDF 1개 복사. (커밋하지 않음, .gitignore 처리)
"""
from pathlib import Path

import chainlit as cl

SAMPLE_PDF = Path(__file__).parent / "sample.pdf"

# 8케이스 정의 — (case_id, display, page, mention)
CASES: list[tuple[str, str, int | None, bool]] = [
    ("A1", "side",   None, True),
    ("A2", "side",   None, False),
    ("A3", "side",   1,    True),
    ("A4", "side",   1,    False),
    ("B1", "inline", None, True),
    ("B2", "inline", 1,    True),
    ("C1", "page",   None, True),
    ("C2", "page",   1,    True),
]


@cl.on_chat_start
async def on_chat_start() -> None:
    if not SAMPLE_PDF.exists():
        await cl.Message(
            content=(
                f"❌ 샘플 PDF를 찾을 수 없습니다.\n"
                f"`{SAMPLE_PDF}` 경로에 아무 PDF 1개를 두고 새로고침하세요."
            )
        ).send()
        return

    await cl.Message(
        content=(
            "## 🧪 cl.Pdf(display=\"side\") 자동 펼침 재현 실험 (#117)\n\n"
            "**관찰 포인트**: 각 버튼을 눌렀을 때 — \n"
            "1. 메시지가 표시되는 즉시 사이드/페이지 패널이 자동으로 열리는가?\n"
            "2. 아니면 링크를 클릭해야 열리는가?\n\n"
            "각 케이스를 **한 번씩만** 클릭하고, 결과를 repro/README.md에 기록하세요.\n"
            "세션 간 영향을 피하려면 **새 세션** (페이지 새로고침)으로 시작하세요."
        )
    ).send()

    actions = [
        cl.Action(
            name="run_case",
            payload={
                "case":    case_id,
                "display": display,
                "page":    page,
                "mention": mention,
            },
            label=(
                f"Case {case_id}: display={display}, "
                f"page={'1' if page else 'None'}, "
                f"mention={mention}"
            ),
        )
        for case_id, display, page, mention in CASES
    ]
    await cl.Message(content="### 테스트 케이스", actions=actions).send()


@cl.action_callback("run_case")
async def on_run_case(action: cl.Action) -> None:
    """각 케이스별 cl.Pdf 엘리먼트를 실제 전송.

    ui/app.py의 _send_pdf_side_panel과 동일한 방식으로 전송하되
    asyncio.wait_for 타임아웃이나 로깅 등 주변 요소는 전부 제거.
    cl.Pdf 생성자의 순수 동작만 관찰할 수 있게 최소화.
    """
    case_id = action.payload["case"]
    display = action.payload["display"]
    page    = action.payload["page"]
    mention = action.payload["mention"]

    pdf_kwargs: dict = {
        "name":    "sample.pdf",
        "display": display,
        "path":    str(SAMPLE_PDF),
    }
    if page is not None:
        pdf_kwargs["page"] = page

    # content에 엘리먼트 name 포함 여부 — side 케이스에서만 의미가 있지만
    # inline/page 케이스도 일관성 위해 동일 분기 사용.
    if mention:
        content = f"[{case_id}] 📂 원문 보기 — sample.pdf"
    else:
        content = f"[{case_id}] 📂 원문 보기 (name 미포함)"

    await cl.Message(
        content=content,
        elements=[cl.Pdf(**pdf_kwargs)],
    ).send()
