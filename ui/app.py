"""
ui/app.py
Chainlit UI 진입점 — ChatGPT 스타일
파일 업로드 → 진행 상태 메시지 → 전체 요약 → 섹션별 차트(cl.Plotly) → PDF 원문 → 추천 질문 → Q&A
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

_APP_USERNAME = os.getenv("APP_USERNAME", "admin")
_APP_PASSWORD = os.getenv("APP_PASSWORD", "1234!")


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if username == _APP_USERNAME and password == _APP_PASSWORD:
        return cl.User(identifier=username, metadata={"role": "admin"})
    return None


def _to_summary_result(result: dict) -> SummaryResult | None:
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


async def _stream(msg: cl.Message, text: str) -> None:
    """줄 단위 스트리밍."""
    for i, line in enumerate(text.split("\n")):
        await msg.stream_token(line)
        if i < len(text.split("\n")) - 1:
            await msg.stream_token("\n")


def _build_follow_ups(summary: SummaryResult | None) -> list[cl.Action]:
    if not summary:
        defaults = ["재무지표 더 자세히 보여줘", "리스크 요인은 무엇인가요?", "향후 전망은?"]
        return [cl.Action(name="followup", payload={"value": q}, label=q) for q in defaults]
    follow_ups = generate_follow_ups(summary)
    return [
        cl.Action(name="followup", payload={"value": fu.question}, label=fu.question)
        for fu in follow_ups
    ]


async def _send_qa_answer(qa_result) -> None:
    msg = cl.Message(content="")
    await msg.send()
    if not qa_result.is_answerable:
        await _stream(msg, f"⚠️ {qa_result.answer}")
        await msg.update()
        return
    await _stream(msg, _fix_tilde(qa_result.answer))
    if qa_result.sources:
        lines = []
        for i, s in enumerate(qa_result.sources):
            section = _clean(s.section)
            snippet = s.snippet.strip() if s.snippet else ""
            lines.append(f"`{i+1}` **{section}**" + (f" — {snippet}" if snippet else ""))
        await msg.stream_token(f"\n\n---\n📌 **출처**\n" + "\n".join(lines))
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
        logger.info("차트 렌더링 완료 — %d개", rendered)


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("result", None)
    cl.user_session.set("raw_chunks", [])
    cl.user_session.set("doc_id", None)
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

        # Step 1
        await cl.Message(content="⏳ **Step 1** · 파싱 중...").send()
        step1 = await asyncio.to_thread(run_step1, tmp_path)
        if step1.get("status") == "error":
            await cl.Message(content=f"❌ 파싱 실패: {step1.get('parse_error', '오류')}").send()
            return

        meta      = step1.get("metadata", {})
        page_count = _fmt(meta.get("page_count"), "페이지")
        clean_len  = f"{step1.get('clean_len', 0):,}자"
        await cl.Message(content=f"⏳ **Step 2** · 청킹 중... ({page_count} · {clean_len})").send()

        # Step 2
        step2 = await asyncio.to_thread(run_step2, step1)
        if step2.get("status") == "error":
            await cl.Message(content=f"❌ 청킹 실패: {step2.get('chunk_error', '오류')}").send()
            return

        chunk_count = step2.get("chunk_count", 0)
        chunks      = step2.get("chunks", [])
        raw_chunks  = [c["text"] for c in chunks if isinstance(c, dict) and c.get("text")]

        await cl.Message(content=f"⏳ **Step 3+4** · 벡터 인덱싱 + 요약 생성 중... ({chunk_count}개 청크)").send()

        # Step 3+4 병렬
        logger.info("asyncio.gather 시작")
        try:
            from summarizer.embedder import index_chunks
            final_doc_id, step3 = await asyncio.gather(
                asyncio.to_thread(index_chunks, chunks, filename),
                run_step3(step2),
            )
        except Exception as e:
            logger.warning("인덱싱 실패 (fallback): %s", e)
            step3 = await run_step3(step2)
            final_doc_id = None

        logger.info("asyncio.gather 완료")

        summary_dict  = step3.get("summary", {})
        section_count = len(summary_dict.get("sections", []))
        await cl.Message(content=f"✅ 분석 완료 — {chunk_count}개 청크 · {section_count}개 섹션").send()

        cl.user_session.set("result", step3)
        cl.user_session.set("raw_chunks", raw_chunks)
        cl.user_session.set("doc_id", final_doc_id or filename)
        summary = _to_summary_result(step3)

        # 전체 요약
        logger.info("전체 요약 스트리밍 시작")
        if summary and summary.overall and not summary.overall.startswith("["):
            msg = cl.Message(content="")
            await msg.send()
            await msg.stream_token("📄 **전체 요약**\n\n")
            await _stream(msg, _fix_tilde(summary.overall))
            await msg.update()
        else:
            await cl.Message(content="⚠️ 전체 요약을 생성하지 못했습니다.").send()

        # 차트
        logger.info("차트 렌더링 시작")
        if summary:
            await _render_charts(summary)

        # PDF 원문
        if tmp_path.suffix.lower() == ".pdf":
            await cl.Message(
                content=f"📂 원문 보기 — {filename}",
                elements=[cl.Pdf(name=filename, display="side", path=str(tmp_path), page=1)],
            ).send()

        # 추천 질문
        actions = _build_follow_ups(summary)
        await cl.Message(content="💬 **이런 것도 물어보세요**", actions=actions).send()
        return

    # Q&A
    result = cl.user_session.get("result")
    if not result:
        await cl.Message(content="먼저 문서를 업로드해 주세요.").send()
        return
    summary = _to_summary_result(result)
    if not summary:
        await cl.Message(content="요약 결과가 없어 Q&A를 실행할 수 없습니다.").send()
        return
    qa_result = await ask(
        message.content,
        summary,
        cl.user_session.get("raw_chunks") or [],
        None,
        cl.user_session.get("doc_id"),
    )
    await _send_qa_answer(qa_result)


@cl.action_callback("followup")
async def on_followup(action: cl.Action):
    result  = cl.user_session.get("result")
    summary = _to_summary_result(result)
    if not summary:
        await cl.Message(content="요약 결과가 없어 Q&A를 실행할 수 없습니다.").send()
        return
    await cl.Message(content=action.payload["value"], author="user").send()
    qa_result = await ask(
        action.payload["value"],
        summary,
        cl.user_session.get("raw_chunks") or [],
        None,
        cl.user_session.get("doc_id"),
    )
    await _send_qa_answer(qa_result)
