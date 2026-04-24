"""
ui/app.py
Chainlit UI 진입점 — ChatGPT 스타일
파일 업로드 → TaskList 진행 표시 → 전체 요약 → 섹션별 차트(cl.Plotly) → 추천 질문 버튼 + PDF 열기 → Q&A
"""
import ast
import asyncio
import json
import logging
import os
import re
import tempfile
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
# 저장 전략: Step name은 `_session_snapshot` 고정. metadata + output 두 곳에
# 동시 기록. 파싱 시 JSON → Python repr(ast) → 실패 순으로 다중 시도.
#
# UI 노출: public/stylesheet.css의 `#step-_session_snapshot` 셀렉터로 숨김.
_SNAPSHOT_STEP_NAME = "_session_snapshot"

# 크로스 플랫폼 임시 저장 경로. Windows에서도 tempfile.gettempdir()은
# %TEMP%를 올바르게 반환해 `/tmp` 하드코딩보다 안전.
_CHUNKS_STORAGE_DIR = Path(
    os.getenv("CHUNKS_STORAGE_DIR")
    or (Path(tempfile.gettempdir()) / "doc-summary-agent" / "chunks")
)


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
    """파일 시스템 안전한 doc_id slug.

    이전 버전은 `[^A-Za-z0-9._-]` 외 문자를 모두 `_`로 치환해 한글 파일명이
    전부 `.pdf` 같은 짧은 slug로 축약되어 다른 문서끼리 충돌했다
    (예: "한화_리포트.pdf", "삼성_리포트.pdf" → 둘 다 ".pdf").

    개선:
      - 파일시스템 금지 문자만 제거 (Windows: `<>:"/\\|?*`, 제어문자).
        한글/영숫자는 그대로 보존.
      - doc_id 전체 해시(MD5 8자)를 suffix로 붙여 동일 파일명도 구분.
        → "한화_리포트.pdf_a3f29b12", "삼성_리포트.pdf_c8e17d45"
    """
    import hashlib
    # 파일시스템 금지 문자 + 제어문자만 `_`로 치환.
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", doc_id)
    cleaned = cleaned.strip("_ ").strip() or "doc"
    # 길이 제한 (Windows 파일명 MAX_PATH 260 고려, 여유 있게 120자).
    if len(cleaned) > 120:
        cleaned = cleaned[:120]
    digest = hashlib.md5(doc_id.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned}_{digest}"


def _chunks_file_path(doc_id: str) -> Path:
    return _CHUNKS_STORAGE_DIR / f"chunks_{_safe_doc_id(doc_id)}.json"


def _save_chunks_to_disk(doc_id: str, raw_chunks: list[str]) -> Path | None:
    """raw_chunks를 임시 디렉토리 JSON 파일로 영속화 (#99).

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

    서버 재시작·임시 디렉토리 청소 등으로 파일이 소실됐을 수 있으므로 실패 시
    빈 리스트 반환. 호출 측에서 Q&A 불가 상태로 graceful degrade.
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


def _parse_snapshot_payload(raw: Any, source_hint: str = "") -> dict:
    """snapshot step의 payload를 dict로 정규화 (#99).

    다중 포맷 시도 — Chainlit 버전/DataLayer에 따라 저장 포맷이 달라질 수 있어
    JSON → Python literal → 포기 순으로 시도:

      1. dict 그대로   → 반환
      2. JSON 문자열    → json.loads 성공 시 반환
      3. Python repr    → ast.literal_eval 성공 시 반환
         (예: "{'doc_id': 'x', ...}" single-quote 포맷)
      4. 그 외          → 경고 로그 + 빈 dict

    실패 시 raw의 앞 200자를 로그로 남겨 실제 포맷 파악에 활용.

    source_hint: "metadata" / "output" 등 호출 맥락 표시용.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw:
        return {}

    # 1차: JSON 파싱 (저장 시 json.dumps로 했다면 정상 경로)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception as json_err:
        # 2차: Python literal (repr 포맷, single-quote dict)
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                logger.info("snapshot payload[%s]: ast.literal_eval로 복원 성공", source_hint)
                return parsed
        except Exception as ast_err:
            # 두 파싱 모두 실패 — 원본 샘플 로그로 실제 포맷 파악.
            sample = raw[:200] + "..." if len(raw) > 200 else raw
            logger.warning(
                "snapshot payload[%s] 파싱 실패: json_err=%s ast_err=%s sample=%r",
                source_hint, json_err, ast_err, sample,
            )

    return {}


def _wrap_str_chunks_for_index(raw_chunks: list[str]) -> list[dict]:
    """list[str] → list[dict] 래핑 (#99).

    summarizer.embedder.index_chunks는 `chunk["text"]`를 요구해 dict 형식
    필수. on_chat_resume에서 raw_chunks(list[str])를 복원하므로 인덱싱 직전에
    최소 필드만 재구성.
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
    """세션 상태를 thread에 Step으로 저장 (#99).

    저장 내용을 metadata와 output 두 곳에 동시에 기록:
      - metadata: Chainlit이 자동으로 JSON 직렬화
      - output: 직접 JSON 문자열로 인코딩해 기록 (백업 경로)

    복원 시 어느 쪽이든 살아있으면 OK. 둘 다 파싱 가능하면 metadata 우선.
    """
    try:
        payload = {
            "doc_id": doc_id,
            "summary": summary_dict,
            "chunks_path": str(chunks_path) if chunks_path else "",
        }
        payload_json = json.dumps(payload, ensure_ascii=False)

        async with cl.Step(
            name=_SNAPSHOT_STEP_NAME,
            type="undefined",
            show_input=False,
        ) as snap:
            snap.input = ""
            # output에 JSON 문자열을 직접 기록. CSS로 UI에 안 보이게 숨김.
            snap.output = payload_json
            # metadata에도 dict 형태로 저장 (Chainlit이 dict→JSON으로 직렬화).
            snap.metadata = payload
        logger.info("세션 스냅샷 저장 완료: doc_id=%s", doc_id)
    except Exception as e:
        # 스냅샷 저장 실패는 resume만 불가능하게 만들 뿐 현재 세션 Q&A에는 영향 없음.
        logger.warning("세션 스냅샷 저장 실패: doc_id=%s err=%s", doc_id, e)


def _find_snapshot(thread: ThreadDict) -> dict | None:
    """thread["steps"]에서 snapshot step을 검색 (#99).

    같은 thread에 여러 문서를 업로드했을 경우 여러 스냅샷이 쌓일 수 있어
    "가장 마지막" 것을 채택.

    이전 버전에서 저장된 `.session_snapshot` 이름도 하위 호환 허용.
    """
    steps = thread.get("steps") or []
    valid_names = {_SNAPSHOT_STEP_NAME, ".session_snapshot"}
    snapshots = [s for s in steps if s.get("name") in valid_names]
    if not snapshots:
        return None
    return snapshots[-1]


def _extract_snapshot_payload(snapshot: dict) -> dict:
    """snapshot step에서 payload 추출 — metadata / output 순서로 시도 (#99).

    두 필드 모두 시도하되 `_parse_snapshot_payload`가 JSON, Python literal 등
    다중 포맷을 커버하므로 한 쪽만 살아있어도 복원 성공.
    """
    # 1차: metadata
    payload = _parse_snapshot_payload(snapshot.get("metadata"), source_hint="metadata")
    if payload:
        logger.debug("snapshot payload: metadata 경로 성공")
        return payload

    # 2차: output
    payload = _parse_snapshot_payload(snapshot.get("output"), source_hint="output")
    if payload:
        logger.info("snapshot payload: output fallback 사용")
        return payload

    # 둘 다 실패 — 이미 _parse_snapshot_payload 내부에서 raw 샘플 로깅됨.
    logger.warning(
        "snapshot payload 추출 실패 — keys=%s metadata_type=%s output_type=%s",
        list(snapshot.keys()),
        type(snapshot.get("metadata")).__name__,
        type(snapshot.get("output")).__name__,
    )
    return {}


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

    저장-복원 전략은 `_save_session_snapshot` / `_extract_snapshot_payload`
    docstring 참조. 요약:
      - 저장: Step.metadata + Step.output 동시 기록
      - 복원: metadata → output 순서로 시도, 각 필드는 JSON → ast.literal_eval
        다중 포맷 파싱

    복원 실패 시 (snapshot 없음 / payload 파싱 실패 / chunks 파일 소실):
      - user_session은 on_chat_start와 동일하게 초기화
      - Q&A 시도 시 _run_qa가 graceful degrade (안내 메시지)

    재렌더링 범위 (의도적 구분):
      ✅ 추천 질문 버튼 (`cl.Action`) — 2.10.1이 resume 시 버튼을 살려내지
         못해 재전송 필수. generate_follow_ups는 ~2-3초 LLM 호출이라 부담 경미.
      ❌ 차트(cl.Plotly) — figure 직렬화가 blob_storage(#73) 미설정으로 실패.
         이번 범위에서 제외 (사용자 결정 — 필요도 낮음).
      ❌ 원본 PDF 열기 버튼 — 서버 임시 경로 소실. 버튼 만들면 "PDF 없음"
         에러만 떠 혼란 유발 → 렌더링 안 함.

      - 답변 메시지 본문 · TaskList — Chainlit이 thread 히스토리로 복원
    """
    # user_session 초기화 (on_chat_start와 동일)
    cl.user_session.set("result", None)
    cl.user_session.set("raw_chunks", [])
    cl.user_session.set("doc_id", None)
    cl.user_session.set("pdf_path", None)
    cl.user_session.set("pdf_sidebar_open", False)

    # ChatSettings 재설정 (Chainlit #1391 워크어라운드) — snapshot 유무와 무관하게 필요.
    await _build_chat_settings()

    # 디버그: thread["steps"]에서 snapshot 관련 step 구조 덤프 (문제 진단용).
    steps = thread.get("steps") or []
    snapshot_like = [s for s in steps if s.get("name", "").endswith("session_snapshot")]
    logger.info(
        "on_chat_resume 진입: thread_id=%s total_steps=%d snapshot_like=%d",
        thread.get("id"), len(steps), len(snapshot_like),
    )
    for i, s in enumerate(snapshot_like):
        keys = list(s.keys())
        has_meta = bool(s.get("metadata"))
        has_output = bool(s.get("output"))
        logger.info(
            "  snapshot[%d] name=%r keys=%s has_metadata=%s has_output=%s",
            i, s.get("name"), keys, has_meta, has_output,
        )

    snapshot = _find_snapshot(thread)
    if not snapshot:
        logger.info("on_chat_resume: snapshot step 없음 — thread_id=%s", thread.get("id"))
        return

    payload = _extract_snapshot_payload(snapshot)
    if not payload:
        logger.warning("on_chat_resume: snapshot payload 비어있음 — 복원 중단")
        return

    doc_id       = payload.get("doc_id") or ""
    summary_dict = payload.get("summary") or {}
    chunks_path  = payload.get("chunks_path") or ""

    raw_chunks = _load_chunks_from_disk(chunks_path)

    # user_session 역주입
    cl.user_session.set("doc_id", doc_id)
    cl.user_session.set("raw_chunks", raw_chunks)
    cl.user_session.set("result", {"summary": summary_dict})

    logger.info(
        "on_chat_resume: 복원 완료 doc_id=%s chunks=%d summary_sections=%d",
        doc_id, len(raw_chunks),
        len(summary_dict.get("sections", [])) if isinstance(summary_dict, dict) else 0,
    )

    # 벡터 인덱스 재구성 — 메모리 싱글턴이라 프로세스 재시작 후엔 비어있음.
    # chunks가 있을 때만 의미 있음. list[str] → list[dict] 래핑 필수.
    if raw_chunks:
        _index_in_background(_wrap_str_chunks_for_index(raw_chunks), doc_id)
        _warmup_reranker_in_background()
    elif chunks_path:
        # chunks_path는 있었는데 파일이 없음 = 소실. 사용자에게 명시적 안내.
        await cl.Message(
            content=(
                "ℹ️ 이전 대화를 불러왔습니다. 요약은 그대로 확인할 수 있지만, "
                "서버에 원본 청크 데이터가 남아있지 않아 **추가 Q&A는 실행할 수 없습니다**.\n"
                "Q&A를 다시 하시려면 새 대화에서 문서를 재업로드해 주세요."
            )
        ).send()
        # chunks 소실 상태에선 추천 질문 버튼을 내놓아도 클릭 시 Q&A 불가라
        # 오히려 혼란 → 렌더링 건너뛰고 조기 리턴.
        return

    # 추천 질문 버튼 재렌더링 (#99).
    #
    # 왜 재렌더가 필요한가:
    #   Chainlit 2.10.1은 `cl.Action`을 thread에 persist하지만 resume 시
    #   UI에 "살아있는" 버튼으로 복원하지 않는다. 기존 메시지의 버튼은
    #   클릭해도 아무 일이 일어나지 않는 상태. 따라서 새 메시지로 재전송.
    #
    # UX 관점:
    #   기존(비활성) 버튼 메시지 아래에 새 버튼 메시지가 추가됨 → 약간의
    #   시각적 중복이 있지만 "옛날 건 죽었고 새 건 살아있다"는 기대를
    #   정직하게 반영. 대안(기존 메시지 삭제)은 Chainlit API 미지원.
    summary_obj = _to_summary_result({"summary": summary_dict})
    if summary_obj:
        try:
            actions = _build_follow_ups(summary_obj)
            await cl.Message(content="💬 **이런 것도 물어보세요**", actions=actions).send()
        except Exception as e:
            # 추천 질문 재생성(LLM 호출) 실패는 resume 전체를 깨지 않도록 warning만.
            logger.warning("추천 질문 재렌더링 실패: %s", e)


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
        # raw_chunks는 별도 파일, 나머지 payload는 step metadata+output에 기록.
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
