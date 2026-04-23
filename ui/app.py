"""
ui/app.py
Chainlit UI 진입점 — ChatGPT 스타일
파일 업로드 → TaskList 진행 표시 → 전체 요약 → 섹션별 차트(cl.Plotly) → 추천 질문 버튼 + PDF 열기 → Q&A
"""
import asyncio
import logging
import os
import re
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Slider, Switch

from main import run_step1, run_step2, run_step3
from summarizer.llm import SummaryResult, SectionSummary
from summarizer.qa import ask, generate_follow_ups
from summarizer.chart_router import route as chart_route

logger = logging.getLogger(__name__)

MAX_CHARTS_PER_DOC = 5
CHART_SIZE = "small"
CHART_SEND_TIMEOUT = 5
PDF_SEND_TIMEOUT = 5

_APP_USERNAME = os.getenv("APP_USERNAME", "admin")
_APP_PASSWORD = os.getenv("APP_PASSWORD", "1234!")


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if username == _APP_USERNAME and password == _APP_PASSWORD:
        return cl.User(identifier=username, metadata={"role": "admin"})
    return None


def _to_summary_result(result) -> SummaryResult | None:
    if not result:
        return None
    summary_dict = result.get("summary")
    if not summary_dict:
        return None
    sections = [SectionSummary(**s) for s in summary_dict.get("sections", [])]
    return SummaryResult(
        overall=summary_dict.get("overall", ""),
        sections=sections,
        is_image_based=summary_dict.get("is_image_based", False),
    )


def _clean(name: str) -> str:
    name = re.sub(r"^#+\s*", "", name)
    name = re.sub(r"\*+", "", name)
    name = name.strip()
    if not name or name.lower() in ("no section", "(no section)"):
        return "문서 본문"
    return name


def _fix_tilde(text: str) -> str:
    return re.sub(r'(\d+\.?\d*)~+(\d+\.?\d*)', r'\1-\2', text)


def _fmt(value, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value}{suffix}"


async def _stream_by_lines(msg: cl.Message, text: str) -> None:
    """줄 단위로 stream_token 호출.

    SQLAlchemyDataLayer 환경에서 stream_token은 매 호출마다 update_step DB write를
    유발하여 한 글자씩 스트리밍하면 수백~수천회의 DB I/O가 발생, 오히려 병목이 되어
    스트리밍 효과가 사라진다. 줄 단위로 보내면 DB write 수가 수십 회로 줄어들어
    실제 스트리밍 UX가 유지된다.
    """
    for line in text.splitlines(keepends=True):
        if line:
            await msg.stream_token(line)


def _build_follow_ups(summary: SummaryResult | None) -> list[cl.Action]:
    """추천 질문을 cl.Action 버튼 리스트로 구성.

    #70에서 SQLAlchemyDataLayer로 전환되면서 HybridDataLayer 시절의
    action_callback → on_chat_start 재트리거 버그가 재현되지 않는지 재검증 중.
    """
    if not summary:
        defaults = ["재무지표 더 자세히 보여줘", "리스크 요인은 무엇인가요?", "향후 전망은?"]
        return [cl.Action(name="followup", payload={"value": q}, label=q) for q in defaults]
    follow_ups = generate_follow_ups(summary)
    return [
        cl.Action(name="followup", payload={"value": fu.question}, label=fu.question)
        for fu in follow_ups
    ]


async def _send_qa_answer(qa_result) -> None:
    """Q&A 답변 메시지 전송.

    흐름:
      1) 답변 본문 스트리밍
      2) 출처 + 원문 근거 — SourceReference 커스텀 엘리먼트 하나에 통합 (#111)

    원래 설계는 cl.Text(display="page") 풀스크린 오버레이였으나
    chainlit/chainlit#1559, #1827 업스트림 버그로 작동 불가.
    대안으로 마크다운 출처 블록 + 개별 "근거 N" 버튼을 띄우는 방식을 거쳤으나
    (a) 거추장스러운 버튼 블록, (b) msg.elements 순서 미보장 (chainlit#2202)
    문제가 있어 최종적으로:

      - 마크다운 "📌 출처" 블록 제거
      - 개별 "근거 N" 버튼 블록 제거
      - 출처 전체를 단일 CustomElement(SourceReference)에 items 배열로 전달
      - JSX 내부에서 번호 배지 + 섹션명 + snippet을 한 줄 버튼으로 렌더
      - 호버 → HoverCard로 원문 앞부분 미리보기
      - 클릭 → Dialog 모달로 원문 전체 보기

    Perplexity/Granola/Sana 스타일 "claim-to-source" UX에 가까움.
    단일 엘리먼트라 Chainlit 내부 순서 미보장 이슈에도 영향받지 않음 —
    React가 items.map()의 렌더링 순서를 보장.

    props.docId: 세션 중인 원본 문서 파일명 (지금 업로드된 문서).
    모달 헤더에 메타데이터로 표시 — 레퍼런스 신뢰도 향상.
    (thefrontkit/Graphlit 가이드: "expandable source cards"에 title, URL, excerpt 메타데이터)
    """
    msg = cl.Message(content="")
    await msg.send()

    if not qa_result.is_answerable:
        await _stream_by_lines(msg, f"⚠️ {qa_result.answer}")
        await msg.update()
        return

    answer = _fix_tilde(qa_result.answer)
    await _stream_by_lines(msg, answer)

    if qa_result.sources:
        # 출처/근거 items 빌드 — 섹션명, snippet, 원문 청크 전부 JSX에 위임.
        # full_chunk가 비어있어도 JSX가 호버/클릭 없이 정보만 렌더하는
        # fallback을 가지고 있어 여기서 필터링하지 않는다.
        items = []
        for i, s in enumerate(qa_result.sources):
            items.append({
                "index":     i + 1,
                "section":   _clean(s.section),
                "snippet":   _fix_tilde(s.snippet.strip()) if s.snippet else "",
                "fullChunk": s.full_chunk.strip() if s.full_chunk else "",
            })

        if items:
            msg.elements = [
                cl.CustomElement(
                    name="SourceReference",
                    display="inline",
                    props={
                        "items": items,
                        # 현재 세션의 원본 문서 파일명 — 모달 헤더 메타데이터로 표시.
                        # cl.user_session.get("doc_id")는 업로드 시 filename이 들어가지만
                        # Q&A 세션이 리셋된 경우 None이 될 수 있어 빈 문자열로 fallback.
                        "docId": cl.user_session.get("doc_id") or "",
                    },
                )
            ]

    await msg.update()


async def _send_chart(section_label: str, fig) -> None:
    await asyncio.wait_for(
        cl.Message(
            content=f"📊 **{section_label}**",
            elements=[cl.Plotly(name=section_label, figure=fig, display="inline", size=CHART_SIZE)],
        ).send(),
        timeout=CHART_SEND_TIMEOUT,
    )


async def _render_charts(summary: SummaryResult) -> None:
    if not cl.user_session.get("chart_enabled", True):
        return
    max_charts = int(cl.user_session.get("max_charts", MAX_CHARTS_PER_DOC))
    rendered = 0
    for sec in summary.sections:
        if rendered >= max_charts:
            logger.info("차트 상한선 도달 (%d개) — 이후 섹션 차트 생략", max_charts)
            break
        try:
            fig = chart_route(sec.chart_spec)
            if fig is None:
                continue
            await _send_chart(_clean(sec.section), fig)
            rendered += 1
        except asyncio.TimeoutError:
            logger.warning("차트 전송 타임아웃 (section=%r)", sec.section)
        except Exception as e:
            logger.warning("차트 렌더링 실패 (section=%r): %s", sec.section, e)
    if rendered > 0:
        logger.info("차트 렌더링 완료 — %d개 출력 (size=%s)", rendered, CHART_SIZE)


async def _open_pdf_in_sidebar(filename: str, tmp_path: Path) -> None:
    """PDF를 사이드바에 직접 푸시 (#117 C3).

    [배경] C1~C2 조사로 cl.Pdf(display="side")는 엘리먼트 마운트 즉시 사이드를
    자동으로 펼치는 업스트림 버그가 확정됨 (Chainlit 2.11.0). content에서
    파일명을 제거해 링크 렌더를 차단해도 사이드가 여전히 열리므로
    "엘리먼트 존재 자체"가 트리거임이 드러남. display="side" 경로는 포기.

    [대안] 공식 ElementSidebar API는 메시지에 엘리먼트를 붙이지 않고 사이드바에
    직접 set_elements/set_title을 호출하는 저수준 경로로, display="side"의
    자동 링크 치환 로직을 완전히 우회한다. 버그 경로를 아예 안 탄다.

    https://docs.chainlit.io/concepts/element (Element Sidebar 섹션):
        "Setting elements will open the sidebar ...
         Setting the elements to an empty array will close the sidebar"

    [효과]
      - 자동 열림 버그 해소: cl.Pdf에 display="side" 파라미터 자체를 안 씀
      - 링크 메시지 제거: cl.Message(...) 호출이 사라져 "📂 원문 보기 —" 메시지가
        chat 히스토리에 남지 않음 → 이슈에 보고된 "메시지 3번 중복" 현상도 자연 소멸
      - 토글 가능: 같은 버튼으로 열고 닫기 (set_elements([])로 close)

    [제약]
      - ElementSidebar 엘리먼트는 persist되지 않음 (공식 문서 명시).
        세션 resume 시에는 사이드가 비어있는 상태로 시작하며 사용자가 버튼을
        다시 눌러야 한다. 현재 UX로는 수용 가능.
    """
    try:
        await asyncio.wait_for(
            cl.ElementSidebar.set_title(f"📂 {filename}"),
            timeout=PDF_SEND_TIMEOUT,
        )
        await asyncio.wait_for(
            # display 파라미터는 지정하지 않음 — ElementSidebar 컨텍스트에선 불필요.
            # page 파라미터도 제거 (C1 조사에서 원인 아님 확인됨).
            cl.ElementSidebar.set_elements([cl.Pdf(name=filename, path=str(tmp_path))]),
            timeout=PDF_SEND_TIMEOUT,
        )
        logger.info("PDF 사이드바 열기 완료: %s", filename)
    except asyncio.TimeoutError:
        logger.warning("PDF 사이드바 열기 타임아웃 (%ds 초과): %s", PDF_SEND_TIMEOUT, filename)
    except Exception as e:
        logger.warning("PDF 사이드바 열기 실패: %s — %s", filename, e)


async def _close_sidebar() -> None:
    """사이드바 닫기 (#117 C3).

    공식 문서: "Setting the elements to an empty array will close the sidebar."
    """
    try:
        await asyncio.wait_for(
            cl.ElementSidebar.set_elements([]),
            timeout=PDF_SEND_TIMEOUT,
        )
        logger.info("PDF 사이드바 닫기 완료")
    except asyncio.TimeoutError:
        logger.warning("PDF 사이드바 닫기 타임아웃")
    except Exception as e:
        logger.warning("PDF 사이드바 닫기 실패: %s", e)


def _index_in_background(chunks: list[dict], doc_id: str) -> None:
    """별도 스레드에서 인덱싱 실행 (fire-and-forget).

    bge-m3는 CPU 연산으로 GIL을 잡으므로 asyncio.to_thread로 감싸도
    이벤트 루프를 블로킹함. 따라서 threading.Thread로 완전히 분리하여
    이벤트 루프와 독립적으로 실행한다.
    """
    import threading

    def _run():
        try:
            from summarizer.embedder import index_chunks
            index_chunks(chunks, doc_id)
            logger.info("백그라운드 인덱싱 완료: %s", doc_id)
        except Exception as e:
            logger.warning("백그라운드 인덱싱 실패: %s", e)

    threading.Thread(target=_run, daemon=True).start()


def _warmup_reranker_in_background() -> None:
    """별도 스레드에서 reranker 싱글턴을 미리 로드 (fire-and-forget).

    업로드 완료 직후 호출. 사용자가 요약/차트를 읽는 자연 대기 시간(30s~1min)에
    bge-reranker-v2-m3 모델을 미리 싱글턴으로 로드해두어 첫 Q&A의 cold start를
    제거한다. (#109)

    bge-reranker-v2-m3도 bge-m3와 같은 sentence-transformers 계열이라 CPU 연산으로
    GIL을 길게 잡으므로 asyncio.to_thread 금지. threading.Thread로 완전히 분리.
    실패 시 graceful skip — Q&A 시점에 _aget_reranker가 다시 시도하지는 않지만
    (sentinel 처리) reranking 스킵 후 RRF 결과가 그대로 LLM 컨텍스트로 전달되어
    Q&A 자체는 정상 동작한다.
    (선례: _index_in_background)
    """
    import threading

    def _run():
        try:
            from summarizer.qa import _get_reranker
            _get_reranker()
            logger.info("백그라운드 reranker 워밍업 완료")
        except Exception as e:
            logger.warning("백그라운드 reranker 워밍업 실패: %s", e)

    threading.Thread(target=_run, daemon=True).start()


async def _run_qa(question: str) -> None:
    """Q&A 공통 실행 — on_message / on_followup 양쪽에서 사용."""
    result     = cl.user_session.get("result")
    summary    = _to_summary_result(result)
    raw_chunks = cl.user_session.get("raw_chunks") or []
    doc_id     = cl.user_session.get("doc_id")

    if not summary:
        await cl.Message(content="먼저 문서를 업로드해 주세요.").send()
        return

    qa_result = await ask(question, summary, raw_chunks, None, doc_id)
    await _send_qa_answer(qa_result)


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("result", None)
    cl.user_session.set("raw_chunks", [])
    cl.user_session.set("doc_id", None)
    cl.user_session.set("pdf_path", None)
    # 사이드바 열림 상태 — open_pdf 액션이 토글 기준으로 삼는다 (#117 C3)
    cl.user_session.set("pdf_sidebar_open", False)
    settings = await cl.ChatSettings([
        Switch(id="chart_enabled", label="차트 자동 생성",
               description="문서 분석 후 수치 데이터를 Plotly 차트로 자동 렌더링합니다.", initial=True),
        Slider(id="max_charts", label="최대 차트 수",
               description="한 문서당 렌더링할 차트의 최대 개수입니다.",
               initial=MAX_CHARTS_PER_DOC, min=1, max=10, step=1),
    ]).send()
    cl.user_session.set("chart_enabled", settings["chart_enabled"])
    cl.user_session.set("max_charts", int(settings["max_charts"]))
    await cl.Message(content="안녕하세요! 📄 아래 **파일 업로드 버튼**으로 문서를 업로드해 주세요.\n\nPDF · DOCX · HWP · DOC 형식을 지원합니다.").send()


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    cl.user_session.set("chart_enabled", settings["chart_enabled"])
    cl.user_session.set("max_charts", int(settings["max_charts"]))


@cl.on_feedback
async def on_feedback(feedback) -> None:
    emoji = "👍" if getattr(feedback, "value", None) == 1 else "👎"
    comment = f" | 코멘트: {feedback.comment!r}" if getattr(feedback, "comment", None) else ""
    logger.info("[Feedback] %s value=%s doc_id=%s%s",
                emoji, getattr(feedback, "value", "?"),
                cl.user_session.get("doc_id") or "unknown", comment)


@cl.on_message
async def on_message(message: cl.Message):

    if message.elements:
        file     = message.elements[0]
        filename = file.name
        suffix   = Path(filename).suffix
        tmp_path = Path(file.path)

        if tmp_path.suffix.lower() != suffix.lower() and suffix:
            new_path = tmp_path.with_suffix(suffix)
            tmp_path.rename(new_path)
            tmp_path = new_path

        doc_id = filename

        task_list = cl.TaskList()
        task_list.status = "분석 중..."
        task1 = cl.Task(title="Step 1 · 파싱",           status=cl.TaskStatus.RUNNING)
        task2 = cl.Task(title="Step 2 · 청킹",           status=cl.TaskStatus.READY)
        task3 = cl.Task(title="Step 3 · 벡터 인덱스 생성", status=cl.TaskStatus.READY)
        task4 = cl.Task(title="Step 4 · 요약 생성",       status=cl.TaskStatus.READY)
        await task_list.add_task(task1)
        await task_list.add_task(task2)
        await task_list.add_task(task3)
        await task_list.add_task(task4)
        await task_list.send()

        # Step 1
        step1 = await asyncio.to_thread(run_step1, tmp_path)
        if step1.get("status") == "error":
            task1.status = cl.TaskStatus.FAILED
            task1.title  = "Step 1 · 파싱 실패"
            task_list.status = "실패"
            await task_list.send()
            await cl.Message(content=f"❌ 파싱 실패: {step1.get('parse_error', '오류')}").send()
            return

        meta       = step1.get("metadata", {})
        page_count = _fmt(meta.get("page_count"), "페이지")
        language   = _fmt(meta.get("language"))
        clean_len  = f"{step1.get('clean_len', 0):,}자"
        task1.status = cl.TaskStatus.DONE
        task1.title  = f"Step 1 · 파싱 완료 — {page_count} · {clean_len} · {language}"
        task2.status = cl.TaskStatus.RUNNING
        await task_list.send()

        # Step 2
        step2 = await asyncio.to_thread(run_step2, step1)
        if step2.get("status") == "error":
            task2.status = cl.TaskStatus.FAILED
            task2.title  = "Step 2 · 청킹 실패"
            task_list.status = "실패"
            await task_list.send()
            await cl.Message(content=f"❌ 청킹 실패: {step2.get('chunk_error', '오류')}").send()
            return

        chunk_count = step2.get("chunk_count", 0)
        chunks      = step2.get("chunks", [])
        raw_chunks  = [c["text"] for c in chunks if isinstance(c, dict) and c.get("text")]
        task2.status = cl.TaskStatus.DONE
        task2.title  = f"Step 2 · 청킹 완료 — {chunk_count}개 청크"
        task3.status = cl.TaskStatus.RUNNING
        task4.status = cl.TaskStatus.RUNNING
        await task_list.send()

        # 인덱싱 백그라운드 + 요약 (핵심: bge-m3 GIL 블로킹 방지)
        _index_in_background(chunks, filename)
        step3 = await run_step3(step2)

        summary_dict  = step3.get("summary", {})
        section_count = len(summary_dict.get("sections", []))

        task3.status = cl.TaskStatus.DONE
        task3.title  = "Step 3 · 벡터 인덱스 생성 중 (백그라운드)"
        if step3.get("status") == "partial":
            task4.status = cl.TaskStatus.FAILED
            task4.title  = "Step 4 · 요약 실패"
            task_list.status = "부분 완료"
        else:
            task4.status = cl.TaskStatus.DONE
            task4.title  = f"Step 4 · 요약 완료 — {section_count}개 섹션"
            task_list.status = "완료 ✓"
        await task_list.send()

        cl.user_session.set("result", step3)
        cl.user_session.set("raw_chunks", raw_chunks)
        cl.user_session.set("doc_id", doc_id)
        # PDF 경로를 세션에 저장 → 나중에 open_pdf action_callback에서 꺼내 씀.
        # 새 문서가 업로드됐으므로 사이드바 상태 리셋 (이전 PDF 정보 초기화).
        if tmp_path.suffix.lower() == ".pdf":
            cl.user_session.set("pdf_path", str(tmp_path))
        else:
            cl.user_session.set("pdf_path", None)
        cl.user_session.set("pdf_sidebar_open", False)
        summary = _to_summary_result(step3)

        # 전체 요약
        if summary and summary.overall and not summary.overall.startswith("["):
            overall = _fix_tilde(summary.overall)
            msg = cl.Message(content="")
            await msg.send()
            await msg.stream_token("📄 **전체 요약**\n\n")
            await _stream_by_lines(msg, overall)
            await msg.update()
        else:
            await cl.Message(content="⚠️ 전체 요약을 생성하지 못했습니다.").send()

        # 차트
        if summary:
            await _render_charts(summary)

        # 추천 질문 버튼 (cl.Action).
        # PDF인 경우 "📂 원본 PDF 열기/닫기" 토글 액션을 뒤에 추가 (#117 C3).
        # on_open_pdf에서 세션 플래그 pdf_sidebar_open 기준으로 열기/닫기 분기.
        actions = _build_follow_ups(summary)
        if tmp_path.suffix.lower() == ".pdf":
            actions.append(cl.Action(
                name="open_pdf",
                payload={},
                label="📂 원본 PDF 열기/닫기",
            ))
        await cl.Message(content="💬 **이런 것도 물어보세요**", actions=actions).send()

        # reranker 사전 워밍업 (fire-and-forget) — 첫 Q&A cold start 제거 (#109)
        # 사용자가 요약/차트를 읽는 자연 대기 시간에 백그라운드 스레드에서
        # bge-reranker-v2-m3 싱글턴을 미리 로드. 추천 질문 메시지 send 이후에
        # 호출하여 업로드 플로우의 모든 스트리밍이 끝난 뒤 GIL 경합을 최소화.
        _warmup_reranker_in_background()
        return

    # Q&A
    await _run_qa(message.content)


@cl.action_callback("followup")
async def on_followup(action: cl.Action):
    """추천 질문 버튼 클릭 시 호출.

    user 메시지로 질문 텍스트를 띄우고 그대로 _run_qa에 넘김.
    과거 HybridDataLayer 환경에서는 이 콜백이 on_chat_start를 재트리거하여
    세션이 리셋되는 버그가 있었음. SQLAlchemy 환경에서 재현 여부 확인 필요.
    """
    question = action.payload["value"]
    await cl.Message(content=question, author="user").send()
    await _run_qa(question)


@cl.action_callback("open_pdf")
async def on_open_pdf(action: cl.Action):
    """원본 PDF 사이드바 토글 (#117 C3).

    세션 플래그 pdf_sidebar_open 기준으로 열기/닫기 분기:
      - False → _open_pdf_in_sidebar 호출, 플래그 True로
      - True  → _close_sidebar 호출, 플래그 False로

    ElementSidebar API를 사용하므로 메시지가 chat 히스토리에 남지 않고
    사이드바만 조작됨. 이슈에 보고된 "📂 원문 보기 — 파일명.pdf 메시지 3번 중복"
    현상이 구조적으로 발생하지 않는다.

    세션에 pdf_path가 없으면 (비-PDF 문서 or 세션 리셋) 안내 메시지만 출력.
    """
    pdf_path_str = cl.user_session.get("pdf_path")
    doc_id       = cl.user_session.get("doc_id")

    if not pdf_path_str or not doc_id:
        await cl.Message(content="⚠️ 열 수 있는 PDF 파일이 없습니다.").send()
        return

    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        logger.warning("PDF 파일이 서버에서 사라짐: %s", pdf_path_str)
        await cl.Message(content="⚠️ 업로드된 PDF 파일을 찾을 수 없습니다. 파일을 다시 업로드해 주세요.").send()
        return

    is_open = bool(cl.user_session.get("pdf_sidebar_open", False))
    if is_open:
        await _close_sidebar()
        cl.user_session.set("pdf_sidebar_open", False)
    else:
        await _open_pdf_in_sidebar(doc_id, pdf_path)
        cl.user_session.set("pdf_sidebar_open", True)
