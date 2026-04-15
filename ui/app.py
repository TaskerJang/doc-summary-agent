"""
ui/app.py
Chainlit UI 진입점 — ChatGPT 스타일
파일 업로드 → TaskList 진행 표시 → 전체 요약 → 섹션별 차트(cl.Plotly) → PDF 원문 → 추천 질문 → Q&A
"""
import asyncio
import re
from pathlib import Path

import chainlit as cl

from main import run_step1, run_step2, run_step3
from summarizer.llm import SummaryResult, SectionSummary
from summarizer.qa import ask, generate_follow_ups
from summarizer.chart_router import route as chart_route


# ── 헬퍼: result dict → SummaryResult 복원 ────────────────
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


# ── 헬퍼: result dict → raw_chunks 텍스트 리스트 추출 ─────
def _to_raw_chunks(result: dict) -> list[str]:
    chunks = result.get("chunks", [])
    return [c["text"] for c in chunks if isinstance(c, dict) and c.get("text")]


# ── 헬퍼: 섹션명 정제 ──────────────────────────────────────
def _clean(name: str) -> str:
    name = re.sub(r"^#+\s*", "", name)
    name = re.sub(r"\*+", "", name)
    name = name.strip()
    if not name or name.lower() in ("no section", "(no section)"):
        return "문서 본문"
    return name


# ── 헬퍼: ~~ 취소선 방지 ───────────────────────────────────
def _fix_tilde(text: str) -> str:
    return re.sub(r'(\d+\.?\d*)~+(\d+\.?\d*)', r'\1-\2', text)


# ── 헬퍼: None 메타값 표시용 ──────────────────────────────
def _fmt(value, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value}{suffix}"


# ── 추천 질문 생성 ─────────────────────────────────────────
def _build_follow_ups(summary: SummaryResult | None) -> list[cl.Action]:
    if not summary:
        defaults = ["재무지표 더 자세히 보여줘", "리스크 요인은 무엇인가요?", "향후 전망은?"]
        return [
            cl.Action(name="followup", payload={"value": q}, label=q)
            for q in defaults
        ]
    follow_ups = generate_follow_ups(summary)
    return [
        cl.Action(name="followup", payload={"value": fu.question}, label=fu.question)
        for fu in follow_ups
    ]


# ── Q&A 답변 전송 ──────────────────────────────────────────
async def _send_qa_answer(qa_result) -> None:
    msg = cl.Message(content="")
    await msg.send()

    if not qa_result.is_answerable:
        for ch in f"⚠️ {qa_result.answer}":
            await msg.stream_token(ch)
        await msg.update()
        return

    answer = _fix_tilde(qa_result.answer)
    for ch in answer:
        await msg.stream_token(ch)

    if qa_result.sources:
        lines = []
        for i, s in enumerate(qa_result.sources):
            section = _clean(s.section)
            snippet = s.snippet.strip() if s.snippet else ""
            if snippet:
                lines.append(f"`{i+1}` **{section}** — {snippet}")
            else:
                lines.append(f"`{i+1}` **{section}**")
        await msg.stream_token(f"\n\n---\n📌 **출처**\n" + "\n".join(lines))

    await msg.update()


# ── 섹션별 차트 렌더링 ─────────────────────────────────────
async def _render_charts(summary: SummaryResult) -> None:
    """
    SummaryResult의 각 섹션에서 chart_spec을 추출해 cl.Plotly()로 렌더링한다.
    chart_route()가 None을 반환하면 해당 섹션은 조용히 건너뜀 (graceful fallback).
    """
    for sec in summary.sections:
        try:
            fig = chart_route(sec.chart_spec)
            if fig is None:
                continue
            section_label = _clean(sec.section)
            await cl.Message(
                content=f"📊 **{section_label}**",
                elements=[
                    cl.Plotly(name=section_label, figure=fig, display="inline")
                ],
            ).send()
        except Exception as e:
            # 개별 차트 실패는 전체 흐름을 멈추지 않음
            import logging
            logging.getLogger(__name__).warning("차트 렌더링 실패 (section=%r): %s", sec.section, e)


# ── 세션 시작 ──────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("result",     None)
    cl.user_session.set("raw_chunks", [])
    cl.user_session.set("doc_id",     None)
    await cl.Message(
        content="안녕하세요! 📄 아래 **파일 업로드 버튼**으로 문서를 업로드해 주세요.\n\nPDF · DOCX · HWP · DOC 형식을 지원합니다."
    ).send()


# ── 파일 업로드 + Q&A 처리 ─────────────────────────────────
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

        # ── TaskList 초기화 ────────────────────────────────
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

        # ── Step 1: 파싱 ──────────────────────────────────
        step1 = await asyncio.to_thread(run_step1, tmp_path)

        if step1.get("status") == "error":
            task1.status = cl.TaskStatus.FAILED
            task1.title  = "Step 1 · 파싱 실패"
            task_list.status = "실패"
            await task_list.send()
            await cl.Message(content=f"❌ 파싱 실패: {step1.get('parse_error', '알 수 없는 오류')}").send()
            return

        meta       = step1.get("metadata", {})
        page_count = _fmt(meta.get("page_count"), "페이지")
        language   = _fmt(meta.get("language"))
        clean_len  = f"{step1.get('clean_len', 0):,}자"

        task1.status = cl.TaskStatus.DONE
        task1.title  = f"Step 1 · 파싱 완료 — {page_count} · {clean_len} · 언어 {language}"
        task2.status = cl.TaskStatus.RUNNING
        await task_list.send()

        # ── Step 2: 청킹 ──────────────────────────────────
        step2 = await asyncio.to_thread(run_step2, step1)

        if step2.get("status") == "error":
            task2.status = cl.TaskStatus.FAILED
            task2.title  = "Step 2 · 청킹 실패"
            task_list.status = "실패"
            await task_list.send()
            await cl.Message(content=f"❌ 청킹 실패: {step2.get('chunk_error', '알 수 없는 오류')}").send()
            return

        raw_chunks = _to_raw_chunks(step2)

        task2.status = cl.TaskStatus.DONE
        task2.title  = f"Step 2 · 청킹 완료 — {step2.get('chunk_count', 0)}개 청크"
        task3.status = cl.TaskStatus.RUNNING
        await task_list.send()

        # ── Step 3: 벡터 인덱스 생성 (청킹 직후) ──────────
        try:
            from summarizer.embedder import index_chunks
            await asyncio.to_thread(index_chunks, raw_chunks, doc_id)
            task3.status = cl.TaskStatus.DONE
            task3.title  = f"Step 3 · 벡터 인덱스 완료 — {len(raw_chunks)}청크"
        except Exception as e:
            task3.status = cl.TaskStatus.FAILED
            task3.title  = "Step 3 · 벡터 인덱스 실패 (BM25 fallback)"
            doc_id = None  # 인덱싱 실패 시 BM25 단독 사용

        task4.status = cl.TaskStatus.RUNNING
        await task_list.send()

        # ── Step 4: 요약 생성 ──────────────────────────────
        step3 = await asyncio.to_thread(run_step3, step2)

        summary_dict  = step3.get("summary", {})
        section_count = len(summary_dict.get("sections", []))

        if step3.get("status") == "partial":
            task4.status = cl.TaskStatus.FAILED
            task4.title  = "Step 4 · 요약 실패"
            task_list.status = "부분 완료"
        else:
            task4.status = cl.TaskStatus.DONE
            task4.title  = f"Step 4 · 요약 완료 — {section_count}개 섹션"
            task_list.status = "완료 ✓"

        await task_list.send()

        # ── 결과 저장 ──────────────────────────────────────
        cl.user_session.set("result",     step3)
        cl.user_session.set("raw_chunks", raw_chunks)
        cl.user_session.set("doc_id",     doc_id)
        summary = _to_summary_result(step3)

        # ── 전체 요약 스트리밍 ─────────────────────────────
        if summary and summary.overall and not summary.overall.startswith("["):
            overall = _fix_tilde(summary.overall)
            msg     = cl.Message(content="")
            await msg.send()
            await msg.stream_token("📄 **전체 요약**\n\n")
            for ch in overall:
                await msg.stream_token(ch)
            await msg.update()
        else:
            await cl.Message(content="⚠️ 전체 요약을 생성하지 못했습니다.").send()

        # ── 섹션별 차트 렌더링 (이슈 #60) ─────────────────
        if summary:
            await _render_charts(summary)

        # ── PDF 원문 사이드 뷰어 ───────────────────────────
        if tmp_path.suffix.lower() == ".pdf":
            elements = [
                cl.Pdf(name=filename, display="side", path=str(tmp_path), page=1)
            ]
            await cl.Message(content=f"📂 원문 보기 — {filename}", elements=elements).send()

        # ── 추천 질문 ──────────────────────────────────────
        actions = _build_follow_ups(summary)
        await cl.Message(content="💬 **이런 것도 물어보세요**", actions=actions).send()
        return

    # ── 텍스트 질문 처리 (Q&A) ────────────────────────────
    result = cl.user_session.get("result")
    if not result:
        await cl.Message(content="먼저 문서를 업로드해 주세요.").send()
        return

    summary = _to_summary_result(result)
    if not summary:
        await cl.Message(content="요약 결과가 없어 Q&A를 실행할 수 없습니다.").send()
        return

    question   = message.content
    raw_chunks = cl.user_session.get("raw_chunks") or []
    doc_id     = cl.user_session.get("doc_id")
    qa_result  = await asyncio.to_thread(ask, question, summary, raw_chunks, None, doc_id)
    await _send_qa_answer(qa_result)


# ── 추천 질문 버튼 클릭 ────────────────────────────────────
@cl.action_callback("followup")
async def on_followup(action: cl.Action):
    question   = action.payload["value"]
    result     = cl.user_session.get("result")
    raw_chunks = cl.user_session.get("raw_chunks") or []
    doc_id     = cl.user_session.get("doc_id")
    summary    = _to_summary_result(result)

    if not summary:
        await cl.Message(content="요약 결과가 없어 Q&A를 실행할 수 없습니다.").send()
        return

    await cl.Message(content=question, author="user").send()
    qa_result = await asyncio.to_thread(ask, question, summary, raw_chunks, None, doc_id)
    await _send_qa_answer(qa_result)
