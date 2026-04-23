"""
ui/app.py
Chainlit UI 진입점 — ChatGPT 스타일
파일 업로드 → TaskList 진행 표시 → 전체 요약 → 섹션별 차트(cl.Plotly) → 추천 질문 버튼 + PDF 열기 → Q&A
"""
import asyncio
import logging
import os
import re
import time
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Slider, Switch

from main import run_step1, run_step2, run_step3
from summarizer.llm import SummaryResult, SectionSummary
from summarizer.qa import ask, generate_follow_ups, is_reranker_loaded
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

    cl.Pdf(display="side") 자동 열림 업스트림 버그를 우회하기 위해 공식
    ElementSidebar API를 사용. display="side" 경로를 안 타므로 버그 회피 +
    chat 히스토리에 링크 메시지도 남지 않아 중복 메시지 문제도 해소.
    """
    try:
        await asyncio.wait_for(
            cl.ElementSidebar.set_title(f"📂 {filename}"),
            timeout=PDF_SEND_TIMEOUT,
        )
        await asyncio.wait_for(
            cl.ElementSidebar.set_elements([cl.Pdf(name=filename, path=str(tmp_path))]),
            timeout=PDF_SEND_TIMEOUT,
        )
        logger.info("PDF 사이드바 열기 완료: %s", filename)
    except asyncio.TimeoutError:
        logger.warning("PDF 사이드바 열기 타임아웃 (%ds 초과): %s", PDF_SEND_TIMEOUT, filename)
    except Exception as e:
        logger.warning("PDF 사이드바 열기 실패: %s — %s", filename, e)


async def _close_sidebar() -> None:
    """사이드바 닫기 (#117 C3). set_elements([]) = close."""
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
    """별도 스레드에서 인덱싱 실행 (fire-and-forget)."""
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
    """별도 스레드에서 reranker 싱글턴을 미리 로드 (fire-and-forget). (#109)"""
    import threading

    def _run():
        try:
            from summarizer.qa import _get_reranker
            _get_reranker()
            logger.info("백그라운드 reranker 워밍업 완료")
        except Exception as e:
            logger.warning("백그라운드 reranker 워밍업 실패: %s", e)

    threading.Thread(target=_run, daemon=True).start()


class _QaStageTracker:
    """Q&A 진행 Step 핸들러 (#108).

    summarizer.qa.ask()가 발행하는 on_stage 이벤트를 cl.Step 시작/종료로 매핑.
    답변 메시지 안에 접힌 형태로 3개 Step이 차례로 전개된다:

      [답변 메시지]
        ├─ 문서 검색    (BM25 + Dense + RRF)       ~3s
        ├─ 관련성 재정렬 (bge-reranker-v2-m3)      10~60s+
        └─ 답변 생성    (GPT-5.2)                  ~3s

    각 Step은 시작 시점에 input을 "⏳ 진행 중..." 문구로 채우고, 종료 시점에
    output을 "✅ 완료 (N.Ns)"로 채운다. 사용자는 답변 본문에 집중하고 필요 시
    Step을 펼쳐 병목 위치를 확인 (ChatGPT thought 펴보기 UX).

    cold start 감지:
      is_reranker_loaded()가 False면 rerank_start 시점에 "모델 최초 로드 중
      (최대 2분)" 안내로 input 교체. #109 워밍업이 성공했으면 일반 문구.

    예외 처리 제약:
      ask()가 최상위 except로 예외를 삼키므로 Step은 "성공"으로 표시되지만
      답변 본문이 "⚠️ [답변 생성 실패]"로 나타나 실패 신호는 사용자에게 전달됨.
      완전한 Step FAILED 전파는 ask()의 예외 처리 구조 재설계가 필요해 #108
      범위 초과.
    """

    # Step 이벤트 쌍 → (Step 이름, Step type) 매핑.
    _STAGE_CONFIG: dict[str, tuple[str, str]] = {
        "search": ("문서 검색", "retrieval"),
        "rerank": ("관련성 재정렬", "rerank"),
        "answer": ("답변 생성", "llm"),
    }

    def __init__(self) -> None:
        self._steps: dict[str, cl.Step] = {}
        self._t0: dict[str, float] = {}

    async def handle(self, event: str) -> None:
        """ask()가 전달하는 이벤트 이름을 Step 시작/종료로 라우팅."""
        if event.endswith("_start"):
            stage = event[: -len("_start")]
            await self._start(stage)
        elif event.endswith("_done"):
            stage = event[: -len("_done")]
            await self._finish(stage)
        else:
            logger.debug("알 수 없는 stage 이벤트: %s", event)

    async def _start(self, stage: str) -> None:
        name, type_ = self._STAGE_CONFIG.get(stage, (stage, "undefined"))
        step = cl.Step(name=name, type=type_)
        await step.__aenter__()

        step.input = self._initial_input(stage)

        try:
            await step.update()
        except Exception as e:
            # Step.update 실패는 UI 표시 문제일 뿐 Q&A 본체는 계속 진행.
            logger.warning("Step update 실패 (stage=%s): %s", stage, e)

        self._steps[stage] = step
        self._t0[stage] = time.monotonic()

    async def _finish(self, stage: str) -> None:
        step = self._steps.pop(stage, None)
        if step is None:
            # _start 없이 _done이 오는 케이스 방어 (정상 흐름에선 발생 안 함).
            logger.debug("stage %s의 Step 컨텍스트 없음 — skip", stage)
            return

        elapsed = time.monotonic() - self._t0.pop(stage, time.monotonic())
        step.output = f"✅ 완료 ({elapsed:.1f}s)"

        try:
            await step.__aexit__(None, None, None)
        except Exception as e:
            logger.warning("Step __aexit__ 실패 (stage=%s): %s", stage, e)

    def _initial_input(self, stage: str) -> str:
        """Step 시작 시 input 영역에 표시할 문구."""
        if stage == "search":
            return "⏳ BM25 + Dense + RRF 실행 중..."
        if stage == "rerank":
            # cold start 감지: reranker 싱글턴 미로드 시 안내 강화.
            # #109 워밍업이 성공했으면 is_reranker_loaded()가 True라 일반 문구.
            if not is_reranker_loaded():
                return "⏳ bge-reranker-v2-m3 모델 최초 로드 중 (최대 2분 소요)..."
            return "⏳ 재정렬 실행 중..."
        if stage == "answer":
            return "⏳ GPT-5.2 응답 생성 중..."
        return "⏳ 실행 중..."


async def _run_qa(question: str) -> None:
    """Q&A 공통 실행 — on_message / on_followup 양쪽에서 사용.

    #108: ask()에 on_stage 콜백을 전달해 검색/재정렬/답변 3구간을 cl.Step으로
    가시화. reranker cold start(최대 1분+) 구간에서 "응답 없음" 체감 제거.
    Step은 답변 메시지 안에 접힌 형태로 붙고, 펼치면 각 구간 경과시간 확인.
    """
    result     = cl.user_session.get("result")
    summary    = _to_summary_result(result)
    raw_chunks = cl.user_session.get("raw_chunks") or []
    doc_id     = cl.user_session.get("doc_id")

    if not summary:
        await cl.Message(content="먼저 문서를 업로드해 주세요.").send()
        return

    tracker = _QaStageTracker()
    qa_result = await ask(
        question, summary, raw_chunks, None, doc_id,
        on_stage=tracker.handle,
    )
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

        if summary:
            await _render_charts(summary)

        actions = _build_follow_ups(summary)
        if tmp_path.suffix.lower() == ".pdf":
            actions.append(cl.Action(
                name="open_pdf",
                payload={},
                label="📂 원본 PDF 열기/닫기",
            ))
        await cl.Message(content="💬 **이런 것도 물어보세요**", actions=actions).send()

        _warmup_reranker_in_background()
        return

    # Q&A
    await _run_qa(message.content)


@cl.action_callback("followup")
async def on_followup(action: cl.Action):
    """추천 질문 버튼 클릭 시 호출."""
    question = action.payload["value"]
    await cl.Message(content=question, author="user").send()
    await _run_qa(question)


@cl.action_callback("open_pdf")
async def on_open_pdf(action: cl.Action):
    """원본 PDF 사이드바 토글 (#117 C3)."""
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
