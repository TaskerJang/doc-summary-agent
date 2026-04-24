"""
ui/app.py
Chainlit UI 진입점 — ChatGPT 스타일
파일 업로드 → TaskList 진행 표시 → 전체 요약 → 섹션별 차트(cl.Plotly) → 추천 질문 버튼 + PDF 열기 → Q&A
"""
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.input_widget import Slider, Switch
from chainlit.types import ThreadDict

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

# ── #99 Chat Resume — 세션 상태 스냅샷 ────────────────────────────
# raw_chunks는 크기가 커서 step metadata에 넣으면 DB row가 비대해지므로
# doc_id 기반 별도 JSON 파일로 분리 저장. snapshot step은 경량 포인터만 보유.
#
# Step name은 앞에 점(.)을 붙여 Chainlit 2.10.1 기본 UI에서 "시스템 내부 step"
# 으로 인식되도록 유도 (실제 필터링은 프런트 쪽 처리라 100% 숨김은 아니지만
# avatar/라벨 노출을 최소화). type="undefined"는 아바타를 안 붙인다.
_SNAPSHOT_STEP_NAME = ".session_snapshot"
_CHUNKS_STORAGE_DIR = Path(os.getenv("CHUNKS_STORAGE_DIR", "/tmp/doc-summary-agent/chunks"))


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


def _safe_doc_id(doc_id: str) -> str:
    """파일 시스템 안전한 doc_id slug (경로 구분자·유니코드 파일명 대응).

    doc_id는 원본 파일명을 그대로 쓰므로 공백·한글·특수문자가 포함될 수 있다.
    `/tmp/doc-summary-agent/chunks/` 아래 파일로 저장할 때는 안전한 슬러그만 남김.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", doc_id).strip("_") or "doc"


def _chunks_file_path(doc_id: str) -> Path:
    return _CHUNKS_STORAGE_DIR / f"chunks_{_safe_doc_id(doc_id)}.json"


def _save_chunks_to_disk(doc_id: str, raw_chunks: list[str]) -> Path | None:
    """raw_chunks를 /tmp 기반 JSON 파일로 영속화 (#99).

    snapshot step metadata에 chunks 전체를 넣으면 SQLite row 크기가 비대해져
    `list_threads` 쿼리가 느려진다. doc_id 기준 별도 파일에 저장하고 step에는
    경로만 보관.

    반환: 저장 성공 시 Path, 실패 시 None.
    """
    try:
        _CHUNKS_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        path = _chunks_file_path(doc_id)
        path.write_text(json.dumps(raw_chunks, ensure_ascii=False), encoding="utf-8")
        logger.info("raw_chunks 디스크 저장: doc_id=%s chunks=%d path=%s",
                    doc_id, len(raw_chunks), path)
        return path
    except Exception as e:
        logger.warning("raw_chunks 디스크 저장 실패: doc_id=%s err=%s", doc_id, e)
        return None


def _load_chunks_from_disk(path_str: str) -> list[str]:
    """snapshot이 가리키는 chunks 파일을 로드 (#99).

    서버 재시작·/tmp 청소 등으로 파일이 소실됐을 수 있으므로 실패 시 빈
    리스트 반환. 호출 측에서 Q&A 불가 상태로 graceful degrade.
    """
    if not path_str:
        return []
    try:
        path = Path(path_str)
        if not path.exists():
            logger.warning("raw_chunks 파일 소실: %s", path_str)
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x) for x in data]
        logger.warning("raw_chunks 파일 형식 오류 (list 아님): %s", path_str)
        return []
    except Exception as e:
        logger.warning("raw_chunks 파일 로드 실패: path=%s err=%s", path_str, e)
        return []


def _parse_snapshot_meta(raw_meta: Any) -> dict:
    """snapshot step의 metadata를 dict로 보장 (#99 버그 수정).

    SQLAlchemyDataLayer는 step metadata를 DB에 JSON 문자열로 저장하는데,
    `get_thread` 응답에서는 dict로 역직렬화되지 않고 **문자열 그대로** 반환된다
    (Chainlit 2.10.1 확인). 그래서 on_chat_resume에서 `meta.get("doc_id")`를
    호출하면 `AttributeError: 'str' object has no attribute 'get'` 발생.

    이 헬퍼는 세 가지 입력 모두 안전하게 dict로 정규화:
      - dict: 그대로 반환
      - str (JSON): json.loads로 파싱
      - 기타/None/파싱 실패: 빈 dict 반환 + 경고 로그
    """
    if isinstance(raw_meta, dict):
        return raw_meta
    if isinstance(raw_meta, str):
        try:
            parsed = json.loads(raw_meta)
            if isinstance(parsed, dict):
                return parsed
            logger.warning("snapshot metadata JSON이 dict 아님: type=%s", type(parsed).__name__)
            return {}
        except Exception as e:
            logger.warning("snapshot metadata JSON 파싱 실패: err=%s raw=%r",
                           e, (raw_meta[:120] + "...") if len(raw_meta) > 120 else raw_meta)
            return {}
    logger.warning("snapshot metadata 형식 미지원: type=%s", type(raw_meta).__name__)
    return {}


def _wrap_str_chunks_for_index(raw_chunks: list[str]) -> list[dict]:
    """list[str] → list[dict] 래핑 (#99 버그 수정).

    summarizer.embedder.index_chunks는 `chunk["text"]`를 요구해 dict 형식
    필수. on_message 원 경로에서는 chunker의 list[dict]를 그대로 인덱싱에
    넘겼지만, on_chat_resume에서는 raw_chunks(list[str])를 복원하므로
    `c["text"]` 접근에서 `'str' object has no attribute 'get'` 에러 발생.

    resume 경로는 원본 chunker 메타(chunk_index, section)를 잃은 상태라
    최소 필드만 재구성 — 인덱싱 기능엔 영향 없음 (임베딩 대상은 text).
    """
    wrapped = []
    for i, text in enumerate(raw_chunks):
        if not text:
            continue
        wrapped.append({
            "text": text,
            "chunk_index": i,
            "section": "",
        })
    return wrapped


async def _save_session_snapshot(
    doc_id: str,
    summary_dict: dict,
    chunks_path: Path | None,
) -> None:
    """현재 세션 상태를 thread에 숨겨진 Step으로 기록 (#99).

    `on_chat_resume`는 thread["steps"]에서 이 step을 찾아 metadata에서 역복원.
    name은 내부 상수(_SNAPSHOT_STEP_NAME)로 고정해 resume 시 필터링.

    UI 노출 최소화 전략:
      - name 앞에 "." 접두사 — 일부 Chainlit 필터가 시스템 step으로 인식
      - type="undefined" — 아바타 미생성
      - input/output 비움 — content 영역 최소화
      - show_input=False — input 영역 접기
    그럼에도 Chainlit 2.10.1 UI는 이 step을 완전히 숨기지는 않는다(프런트
    필터 범위 밖). 최악의 경우 짧은 구분선 하나 정도로 표시되어 기존
    파란 Alert보다는 훨씬 덜 거슬림.
    """
    try:
        async with cl.Step(
            name=_SNAPSHOT_STEP_NAME,
            type="undefined",
            show_input=False,
        ) as snap:
            snap.input = ""
            snap.output = ""
            snap.metadata = {
                "doc_id": doc_id,
                "summary": summary_dict,
                "chunks_path": str(chunks_path) if chunks_path else "",
            }
        logger.info("세션 스냅샷 저장 완료: doc_id=%s", doc_id)
    except Exception as e:
        # 스냅샷 저장 실패는 resume만 불가능하게 만들 뿐 현재 세션 Q&A에는 영향 없음.
        logger.warning("세션 스냅샷 저장 실패: doc_id=%s err=%s", doc_id, e)


def _find_snapshot(thread: ThreadDict) -> dict | None:
    """thread["steps"]에서 snapshot step을 검색 (#99).

    같은 thread에 여러 문서를 업로드했을 경우 여러 스냅샷이 쌓일 수 있어
    "가장 마지막" 것을 채택 (최신 상태 우선).

    구 이름(`_session_snapshot`)으로 저장된 기존 thread와의 하위 호환을
    위해 두 이름 모두 허용 — 필터링 후 있으면 그대로 사용.
    """
    steps = thread.get("steps") or []
    valid_names = {_SNAPSHOT_STEP_NAME, "_session_snapshot"}
    snapshots = [s for s in steps if s.get("name") in valid_names]
    if not snapshots:
        return None
    return snapshots[-1]


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
    """별도 스레드에서 인덱싱 실행 (fire-and-forget).

    chunks는 **list[dict]**만 받는다 — summarizer.embedder.index_chunks가
    `chunk["text"]`, `chunk.get("section")` 등 dict 키 접근을 전제로 하기 때문.
    on_chat_resume 경로에서는 list[str]이 복원되므로 호출 전에
    `_wrap_str_chunks_for_index`로 래핑 필요.
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


async def _build_chat_settings() -> dict:
    """ChatSettings 위젯 생성 + 세션에 반영 (on_chat_start / on_chat_resume 공용).

    Chainlit 이슈 #1391: resume 후 ChatSettings가 사라지는 버그가 있어
    on_chat_resume에서도 동일 위젯을 다시 send해야 한다. 중복 로직을 막기 위해
    함수로 분리.
    """
    settings = await cl.ChatSettings([
        Switch(id="chart_enabled", label="차트 자동 생성",
               description="문서 분석 후 수치 데이터를 Plotly 차트로 자동 렌더링합니다.", initial=True),
        Slider(id="max_charts", label="최대 차트 수",
               description="한 문서당 렌더링할 차트의 최대 개수입니다.",
               initial=MAX_CHARTS_PER_DOC, min=1, max=10, step=1),
    ]).send()
    cl.user_session.set("chart_enabled", settings["chart_enabled"])
    cl.user_session.set("max_charts", int(settings["max_charts"]))
    return settings


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

    # resume 후 chunks 소실 케이스 — 요약은 보이지만 Q&A는 불가
    if not raw_chunks:
        await cl.Message(
            content=(
                "⚠️ 이 대화의 문서 청크 데이터가 서버에 남아있지 않아 Q&A를 실행할 수 없습니다.\n"
                "새 대화를 시작해 문서를 다시 업로드해 주세요."
            )
        ).send()
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
    await _build_chat_settings()
    await cl.Message(content="안녕하세요! 📄 아래 **파일 업로드 버튼**으로 문서를 업로드해 주세요.\n\nPDF · DOCX · HWP · DOC 형식을 지원합니다.").send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """과거 thread 클릭 시 세션 상태 복원 (#99).

    목적:
      1) 사이드바에서 과거 대화 클릭 → 요약/차트/출처가 다시 보이고 Q&A 재개 가능.
      2) Chainlit 기본 동작은 thread["steps"]로 UI만 재구성 — user_session은 빈 상태라
         Q&A 시 "먼저 문서를 업로드해 주세요" 메시지가 나오는 문제.

    복원 대상:
      - doc_id, raw_chunks, summary(dict)
      - ChatSettings (chart_enabled, max_charts) — 이슈 #1391 워크어라운드
    복원 실패 시 (snapshot 없음 / chunks 파일 소실):
      - user_session은 on_chat_start와 동일하게 초기화
      - Q&A 시도 시 _run_qa가 graceful degrade (안내 메시지)

    재렌더링하지 않는 것:
      - 답변 메시지 본문 · TaskList · 추천 질문 버튼 — Chainlit이 thread 히스토리로 복원
      - 차트(cl.Plotly) — figure 직렬화가 blob_storage에 의존(이슈 #73). 대신
        chart_spec이 summary에 있어 재생성 가능하지만, 메시지 순서가 꼬일 수 있어
        이번 범위에서는 제외. 필요 시 후속 PR에서 on_chat_resume 전용 차트 섹션
        재삽입으로 추가.
      - 원본 PDF 업로드 파일 — 서버 임시 경로라 소실 확정

    버그 수정 이력:
      - step metadata가 str(JSON)로 들어오는 케이스 → `_parse_snapshot_meta`로
        dict 정규화. SQLAlchemyDataLayer 역직렬화 안 되는 이슈 우회.
      - _index_in_background에 list[str] 전달 시 AttributeError → 인덱싱
        직전에 `_wrap_str_chunks_for_index`로 list[dict]로 래핑.
    """
    # user_session 초기화 (on_chat_start와 동일)
    cl.user_session.set("result", None)
    cl.user_session.set("raw_chunks", [])
    cl.user_session.set("doc_id", None)
    cl.user_session.set("pdf_path", None)
    cl.user_session.set("pdf_sidebar_open", False)

    # ChatSettings 재설정 (Chainlit #1391 워크어라운드) — snapshot 유무와 무관하게 필요.
    await _build_chat_settings()

    snapshot = _find_snapshot(thread)
    if not snapshot:
        logger.info("on_chat_resume: snapshot 없음 — thread_id=%s", thread.get("id"))
        return

    # metadata는 SQLAlchemyDataLayer에서 JSON 문자열로 돌아올 수 있어 정규화.
    meta = _parse_snapshot_meta(snapshot.get("metadata"))
    doc_id       = meta.get("doc_id") or ""
    summary_dict = meta.get("summary") or {}
    chunks_path  = meta.get("chunks_path") or ""

    raw_chunks = _load_chunks_from_disk(chunks_path)

    # user_session 역주입
    cl.user_session.set("doc_id", doc_id)
    cl.user_session.set("raw_chunks", raw_chunks)
    cl.user_session.set("result", {"summary": summary_dict})

    logger.info(
        "on_chat_resume: 복원 완료 doc_id=%s chunks=%d summary_sections=%d",
        doc_id, len(raw_chunks), len(summary_dict.get("sections", [])) if isinstance(summary_dict, dict) else 0,
    )

    # 벡터 인덱스 재구성 — 메모리 싱글턴이라 프로세스 재시작 후엔 비어있음.
    # chunks가 있을 때만 의미 있음. list[str] → list[dict] 래핑 필수.
    if raw_chunks:
        _index_in_background(_wrap_str_chunks_for_index(raw_chunks), doc_id)
        _warmup_reranker_in_background()
    else:
        # chunks 파일 소실 — 사용자에게 왜 Q&A가 안 되는지 미리 알림.
        await cl.Message(
            content=(
                "ℹ️ 이전 대화를 불러왔습니다. 요약은 그대로 확인할 수 있지만, "
                "서버에 원본 청크 데이터가 남아있지 않아 **추가 Q&A는 실행할 수 없습니다**.\n"
                "Q&A를 다시 하시려면 새 대화에서 문서를 재업로드해 주세요."
            )
        ).send()


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

        # #99 Chat Resume — 세션 상태를 thread에 스냅샷으로 영속화.
        # raw_chunks는 별도 파일, 나머지 메타는 step metadata에 기록.
        chunks_path = _save_chunks_to_disk(doc_id, raw_chunks)
        await _save_session_snapshot(doc_id, summary_dict, chunks_path)

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
