import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Awaitable, Callable

from openai import OpenAI
from rank_bm25 import BM25Okapi

from summarizer.llm import SummaryResult, _call_api

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL   = "gpt-5.2"

PROMPTS_DIR           = Path(__file__).parent / "prompts"
QA_PROMPT_PATH        = PROMPTS_DIR / "qa_v1.md"
FOLLOW_UP_PROMPT_PATH = PROMPTS_DIR / "follow_up_v1.md"

MAX_SUMMARY_SECTIONS   = 3
MAX_RAW_CHUNKS         = 5    # RRF 퓨전 후보
RERANK_TOP_N           = 3    # reranker 최종 반환 수
CHUNK_BM25_MIN_SCORE   = 0.1
SECTION_BM25_MIN_SCORE = 0.1
RRF_K                  = 60

# ── #108 Q&A 진행 상태 이벤트 ────────────────────────────────────────────
#
# ask()가 외부(UI 레이어)에 단계 진행 신호를 보낼 때 쓰는 이벤트 이름.
# summarizer 레이어에 Chainlit 의존성이 침투하지 않도록 콜백 방식으로 설계.
# ui/app.py의 _run_qa에서 각 이벤트를 cl.Step의 시작/종료로 매핑한다.
#
# 이벤트 쌍:
#   search_start  → search_done  : BM25 + Dense + RRF 실행 구간
#   rerank_start  → rerank_done  : bge-reranker-v2-m3 실행 구간 (병목 구간)
#   answer_start  → answer_done  : GPT-5.2 답변 생성 구간
#
# 각 구간은 예외 발생 가능. UI는 예외 시에도 해당 Step이 FAILED로 표시되도록
# try/except/finally로 Step 컨텍스트를 관리해야 한다.
OnStage = Callable[[str], Awaitable[None]]


_OCR_NOISE_RE = re.compile(
    r"\[OCR [^\]]*\]"
    r"|\[OCR\]"
    r"|※ 이미지 기반 PDF[^\n]*"
    r"|원문:\s*[^\]]*"
)

_UNANSWERABLE_PATTERNS = [
    "문서에서 해당 내용을 찾을 수 없",
    "문서에서 확인할 수 없",
    "제공된 문서에는 해당 내용이 없",
    "해당 정보는 문서에 포함되어 있지 않",
]

_QA_SYSTEM_PROMPT = (
    "당신은 금융 문서 내용을 기반으로 질문에 답변하는 전문 AI입니다. "
    "제공된 문서 내용에만 근거하여 답변하세요. "
    "원문 발췌 > 요약 섹션 > 전체 요약 순으로 우선 참조하세요. "
    "전체 요약에 답이 있으면 반드시 그 내용을 바탕으로 답변하세요. "
    "문서에 없는 내용은 절대 생성하지 마세요."
)

# ── reranker 싱글턴 ─────────────────────────────────────────────────────────
_reranker = None


def _get_reranker():
    """
    bge-reranker-v2-m3 싱글턴 (동기).

    _aget_reranker와 동일한 _reranker 전역 싱글턴을 공유한다. Q&A 경로에서는
    이벤트 루프 블로킹을 피하기 위해 _aget_reranker(asyncio.to_thread 래핑)를
    쓰고, 업로드 직후 백그라운드 워밍업에서는 이 동기 버전을 쓴다.

    bge-reranker-v2-m3는 bge-m3와 동일한 sentence-transformers 계열이라
    CPU 연산으로 GIL을 길게 잡는다. 따라서 이 함수는 반드시 threading.Thread로
    완전히 분리된 스레드에서만 호출되어야 한다. 이벤트 루프가 돌아가는
    async 컨텍스트에서 직접 호출하면 UI 스트리밍이 블로킹된다.
    (선례: summarizer.embedder.index_chunks 동기 함수를 _index_in_background
     threading.Thread에서 호출)

    미설치 또는 로드 실패 시 None 반환, _reranker는 False sentinel로
    세팅되어 재시도를 막는다. 워밍업 실패 시에도 sentinel이 세팅되어
    이후 _aget_reranker 호출 시 재시도하지 않고 reranking을 스킵한다 —
    이 경우 RRF 결과가 그대로 LLM 컨텍스트로 전달된다.
    """
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("bge-reranker-v2-m3 로딩 중...")
            _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
            logger.info("bge-reranker-v2-m3 로딩 완료")
        except Exception as e:
            logger.warning("reranker 로드 실패 — reranking 스킵: %s", e)
            _reranker = False  # 재시도 방지용 sentinel
    return _reranker if _reranker else None


async def _aget_reranker():
    """
    bge-reranker-v2-m3 싱글턴 (async).
    cross-encoder 방식으로 (질문, 청크) 쌍의 관련도를 0~1 스코어로 반환.
    미설치 또는 로드 실패 시 None 반환 → reranking 스킵.

    첫 호출 시 CrossEncoder 초기화가 모델 파일 로드(디스크 I/O + CPU)로
    수 초~수십 초 블로킹되므로 asyncio.to_thread로 감싸 이벤트 루프를
    살려둔다. (#107)

    업로드 직후 _warmup_reranker_in_background에서 동기 _get_reranker로
    이미 싱글턴이 로드되어 있으면 이 함수는 캐시된 _reranker를 즉시 반환.
    (#109)
    """
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("bge-reranker-v2-m3 로딩 중...")
            _reranker = await asyncio.to_thread(
                CrossEncoder, "BAAI/bge-reranker-v2-m3", max_length=512
            )
            logger.info("bge-reranker-v2-m3 로딩 완료")
        except Exception as e:
            logger.warning("reranker 로드 실패 — reranking 스킵: %s", e)
            _reranker = False  # 재시도 방지용 sentinel
    return _reranker if _reranker else None


def is_reranker_loaded() -> bool:
    """UI 레이어가 cold start 여부를 판단할 때 쓰는 조회 함수 (#108).

    _reranker 전역 변수를 직접 import하는 대신 이 함수를 쓰면 레이어
    분리가 깔끔하다. True면 싱글턴 로드 완료, False면 아직 미로드 또는
    로드 실패 sentinel 상태.
    """
    return bool(_reranker)


async def _rerank(question: str, chunks: list[str], top_n: int = RERANK_TOP_N) -> list[str]:
    """
    bge-reranker-v2-m3로 (질문, 청크) 쌍 관련도 스코어링 후 top_n 반환.
    reranker 없거나 청크 수가 top_n 이하면 그대로 반환.

    reranker.predict는 CPU에서 cross-encoder 추론으로 첫 호출 ~1분 걸리므로
    asyncio.to_thread로 분리. 이벤트 루프가 살아있어야 Chainlit websocket
    ping/pong이 유지되어 세션이 끊기지 않는다. (#107)
    """
    if len(chunks) <= top_n:
        return chunks
    reranker = await _aget_reranker()
    if reranker is None:
        return chunks[:top_n]
    try:
        pairs  = [(question, c) for c in chunks]
        scores = await asyncio.to_thread(reranker.predict, pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        result = [c for _, c in ranked[:top_n]]
        logger.info("Reranking 완료 — %d → %d청크", len(chunks), len(result))
        return result
    except Exception as e:
        logger.warning("Reranking 실패 — 원본 순서 유지: %s", e)
        return chunks[:top_n]


def _is_unanswerable(text: str) -> bool:
    return any(p in text for p in _UNANSWERABLE_PATTERNS)


def _strip_ocr_noise(text: str) -> str:
    return _OCR_NOISE_RE.sub("", text).strip()


async def _call_qa(user_prompt: str) -> str:
    """_call_api는 async def이므로 반드시 await 호출."""
    return await _call_api(
        messages=[
            {"role": "system", "content": _QA_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=3000,   # reasoning 모델 대응 — 추론+답변 토큰 동시 수용 (500 → 3000)
    )


class SourceItem:
    def __init__(self, section: str, snippet: str, full_chunk: str = ""):
        self.section    = section
        self.snippet    = snippet
        self.full_chunk = full_chunk


class QAResult:
    def __init__(self, answer: str, sources: list[SourceItem], is_answerable: bool):
        self.answer        = answer
        self.sources       = sources
        self.is_answerable = is_answerable


class FollowUp:
    def __init__(self, question: str, section_indices: list[int]):
        self.question        = question
        self.section_indices = section_indices


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[가-힣a-zA-Z0-9]+", text)


def _fix_tilde(text: str) -> str:
    return re.sub(r'(\d+\.?\d*)~+(\d+\.?\d*)', r'\1-\2', text)


def _best_bullet_by_answer(answer: str, bullets: list[str]) -> str:
    if not bullets:
        return ""
    if len(bullets) == 1:
        return bullets[0]
    answer_tokens = set(_tokenize(answer))
    if not answer_tokens:
        return bullets[0]
    scored = [(len(answer_tokens & set(_tokenize(b))), b) for b in bullets]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored[0][0] > 0 else bullets[0]


def _rrf_fusion(
    bm25_chunks: list[str],
    dense_chunks: list[str],
    k: int = RRF_K,
    top_n: int = MAX_RAW_CHUNKS,
) -> list[str]:
    """
    Reciprocal Rank Fusion:
    score(d) = 1/(k + rank_bm25) + 1/(k + rank_dense)
    """
    all_chunks = list(dict.fromkeys(bm25_chunks + dense_chunks))
    scores: dict[str, float] = {c: 0.0 for c in all_chunks}

    for rank, chunk in enumerate(bm25_chunks, start=1):
        scores[chunk] += 1.0 / (k + rank)
    for rank, chunk in enumerate(dense_chunks, start=1):
        scores[chunk] += 1.0 / (k + rank)

    ranked = sorted(all_chunks, key=lambda c: scores[c], reverse=True)
    return ranked[:top_n]


async def _rrf_search(
    question: str,
    raw_chunks: list[str],
    doc_id: str | None = None,
) -> list[str]:
    """BM25 + Dense(Qdrant) + RRF 퓨전 (#108에서 분리).

    기존 _find_relevant_chunks_hybrid에서 reranker 앞 단계만 떼어낸 함수.
    ask()가 search와 rerank 사이에 Step 경계를 넣을 수 있도록 분리했다.

    doc_id 없거나 Qdrant 미연결 시 BM25 결과만 리턴.
    """
    bm25_chunks = _find_relevant_chunks_bm25(question, raw_chunks)

    dense_chunks: list[str] = []
    if doc_id:
        try:
            from summarizer.embedder import search_chunks
            dense_chunks = search_chunks(question, doc_id, top_k=MAX_RAW_CHUNKS)
        except Exception as e:
            logger.warning("Dense 검색 실패 — BM25 단독 사용: %s", e)

    if dense_chunks:
        fused = _rrf_fusion(bm25_chunks, dense_chunks)
        logger.info("RRF 퓨전 완료 — bm25:%d dense:%d → fused:%d",
                    len(bm25_chunks), len(dense_chunks), len(fused))
        return fused

    return bm25_chunks


def _find_relevant_sections_bm25(question: str, summary: SummaryResult) -> list:
    if not summary.sections:
        return []
    section_texts = [sec.section + " " + " ".join(sec.bullets) for sec in summary.sections]
    tokenized_sections = [_tokenize(t) for t in section_texts]
    tokenized_query    = _tokenize(question)
    valid = [(sec, tok) for sec, tok in zip(summary.sections, tokenized_sections) if tok]
    if not valid:
        return []
    sections_valid, tokenized_valid = zip(*valid)
    bm25   = BM25Okapi(list(tokenized_valid))
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(scores, sections_valid), key=lambda x: x[0], reverse=True)
    return [sec for score, sec in ranked[:MAX_SUMMARY_SECTIONS] if score >= SECTION_BM25_MIN_SCORE]


def _find_relevant_chunks_bm25(question: str, raw_chunks: list[str]) -> list[str]:
    if not raw_chunks:
        return []
    tokenized_chunks = [_tokenize(c) for c in raw_chunks]
    tokenized_query  = _tokenize(question)
    valid = [(chunk, tok) for chunk, tok in zip(raw_chunks, tokenized_chunks) if tok]
    if not valid:
        return []
    chunks_valid, tokenized_valid = zip(*valid)
    bm25   = BM25Okapi(list(tokenized_valid))
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(scores, chunks_valid), key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in ranked[:MAX_RAW_CHUNKS] if score >= CHUNK_BM25_MIN_SCORE]


def _build_context(
    relevant_sections: list,
    relevant_chunks: list[str],
    overall: str = "",
    force_overall: bool = False,
) -> str:
    parts = []
    if not force_overall:
        if relevant_chunks:
            parts.append("## 원문 발췌")
            for i, chunk in enumerate(relevant_chunks, 1):
                parts.append(f"[원문 {i}]\n{chunk}")
        if relevant_sections:
            parts.append("## 요약 섹션")
            for sec in relevant_sections:
                parts.append(f"[{sec.section}]\n" + "\n".join(f"- {b}" for b in sec.bullets))
    if (force_overall or (not relevant_sections and not relevant_chunks)) and overall:
        parts.append("## 전체 요약")
        parts.append(overall[:3000])
    return "\n\n".join(parts)


def _make_sources(
    answer: str,
    relevant_sections: list,
    relevant_chunks: list[str],
) -> list[SourceItem]:
    """
    출처 아이템 생성.

    relevant_sections와 relevant_chunks는 서로 다른 retrieval 파이프라인
    (섹션 BM25 vs RRF+reranker)의 출력이라 순서 대응이 보장되지 않는다.
    지금은 가장 단순한 인덱스 매칭(i번째 섹션 ↔ i번째 청크)으로 시작하고,
    품질 이슈 발생 시 섹션명 기반 매칭·BM25 매칭으로 후속 개선한다. (#111)

    중복 섹션명이 스킵된 경우 해당 인덱스의 full_chunk도 결과에서 빠질 수
    있으나(번호 비연속), 실제 중복 스킵은 드물어 현 단계에선 허용.

    relevant_chunks 길이가 부족하면 full_chunk=""로 세팅 → UI 레이어에서
    해당 링크 생략.
    """
    seen: set[str] = set()
    result = []
    for i, sec in enumerate(relevant_sections):
        key = sec.section.strip()
        if key in seen:
            continue
        seen.add(key)
        snippet    = _fix_tilde(_best_bullet_by_answer(answer, sec.bullets))
        full_chunk = relevant_chunks[i] if i < len(relevant_chunks) else ""
        result.append(SourceItem(
            section=key,
            snippet=snippet,
            full_chunk=full_chunk,
        ))
    return result


async def _emit(on_stage: OnStage | None, event: str) -> None:
    """콜백 안전 호출 (#108).

    on_stage가 None이면 no-op. 콜백 자체가 예외를 던져도 Q&A 본체가
    깨지지 않도록 격리 — UI Step 관리 실패가 Q&A 결과에 영향을 주면 안 된다.
    """
    if on_stage is None:
        return
    try:
        await on_stage(event)
    except Exception as e:
        logger.warning("on_stage 콜백 실패 (event=%s): %s", event, e)


async def ask(
    question: str,
    summary: SummaryResult,
    raw_chunks: list[str] | None = None,
    pinned_section_indices: list[int] | None = None,
    doc_id: str | None = None,
    on_stage: OnStage | None = None,
) -> QAResult:
    """
    Q&A 메인 함수 (async).

    _call_api가 async def이므로 ask()도 async로 전환. (#68 AsyncOpenAI 이관 후속)
    검색 파이프라인: BM25 + Dense(Qdrant) → RRF → bge-reranker-v2-m3 → LLM
    doc_id 없으면 BM25 단독 (하위 호환).

    #108: on_stage 콜백으로 UI 레이어에 단계 진행 신호 발행. 이벤트 쌍은
    (search_start/done), (rerank_start/done), (answer_start/done) 세 구간.
    콜백이 None이면 모든 발행은 no-op이라 기존 호출자는 그대로 동작한다.
    """
    logger.info("Q&A 시작 — 질문: %r (doc_id=%s)", question[:50], doc_id)

    if pinned_section_indices is not None:
        relevant_sections = [
            summary.sections[i]
            for i in pinned_section_indices
            if 0 <= i < len(summary.sections)
        ]
    else:
        relevant_sections = _find_relevant_sections_bm25(question, summary)

    # ── 검색 단계 (BM25 + Dense + RRF) ──────────────────────────────
    await _emit(on_stage, "search_start")
    try:
        rrf_chunks = await _rrf_search(question, raw_chunks or [], doc_id=doc_id)
    finally:
        await _emit(on_stage, "search_done")

    # ── 재정렬 단계 (bge-reranker-v2-m3) ────────────────────────────
    # reranker가 병목 구간 — cold start 시 최대 2분. UI에서 이 구간을
    # 가장 두드러지게 표시해야 "응답 없음" 체감이 사라진다.
    await _emit(on_stage, "rerank_start")
    try:
        relevant_chunks = await _rerank(question, rrf_chunks, top_n=RERANK_TOP_N)
    finally:
        await _emit(on_stage, "rerank_done")

    template = QA_PROMPT_PATH.read_text(encoding="utf-8")

    try:
        if relevant_sections or relevant_chunks:
            context     = _build_context(relevant_sections, relevant_chunks)
            user_prompt = template.format(question=question, context=context)

            # ── 답변 생성 단계 (GPT-5.2) ────────────────────────
            await _emit(on_stage, "answer_start")
            try:
                raw = await _call_qa(user_prompt)
            finally:
                await _emit(on_stage, "answer_done")

            if not _is_unanswerable(raw):
                sources = _make_sources(raw, relevant_sections, relevant_chunks)
                logger.info("Q&A 완료 — 출처 %d개", len(sources))
                return QAResult(answer=raw, sources=sources, is_answerable=True)

            logger.info("검색 결과 답변 불가 — overall fallback 시도")

        if not summary.overall:
            logger.warning("overall 없음 — 답변 불가")
            return QAResult(answer="문서에서 해당 내용을 찾을 수 없습니다.", sources=[], is_answerable=False)

        logger.info("overall fallback 사용")
        context     = _build_context([], [], overall=summary.overall, force_overall=True)
        user_prompt = template.format(question=question, context=context)

        # overall fallback도 LLM 호출이므로 answer Step 대상.
        # 첫 번째 _call_qa가 돌았는데 unanswerable이었던 경우 여기까지 왔으면
        # answer_start/done은 이미 한 번 발행됐을 수 있다. fallback 호출도
        # 동일 이벤트로 감싸서 사용자에게 "답변 재생성 중"임을 표시한다.
        await _emit(on_stage, "answer_start")
        try:
            raw = await _call_qa(user_prompt)
        finally:
            await _emit(on_stage, "answer_done")

        if _is_unanswerable(raw):
            return QAResult(answer=raw, sources=[], is_answerable=False)

        logger.info("Q&A 완료 (overall fallback)")
        return QAResult(answer=raw, sources=[], is_answerable=True)

    except Exception as e:
        logger.error("Q&A 생성 실패: %s", e)
        return QAResult(answer="[답변 생성 실패]", sources=[], is_answerable=False)


def generate_follow_ups(summary: SummaryResult) -> list[FollowUp]:
    n = len(summary.sections)
    all_idx = list(range(n)) if n else [0]
    defaults = [
        FollowUp("재무지표 더 자세히 보여줘", all_idx),
        FollowUp("리스크 요인은 무엇인가요?", all_idx),
        FollowUp("향후 전망은?", all_idx),
    ]
    if not summary.overall:
        return defaults

    overall_context = _strip_ocr_noise(summary.overall)
    if not overall_context.strip():
        return defaults

    try:
        template    = FOLLOW_UP_PROMPT_PATH.read_text(encoding="utf-8")
        user_prompt = template.format(overall=overall_context)

        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": (
                    "당신은 금융 문서 분석 전문가입니다. "
                    "제공된 문서 요약에 근거해 독자가 실제로 물어볼 만한 구체적인 질문을 생성하세요."
                )},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        logger.info("추천 질문 LLM 응답 raw: %r", raw[:300])

        data  = json.loads(raw)
        items = data.get("questions", [])
        logger.info("추천 질문 items type=%s value=%r", type(items).__name__, items)

        result = []
        for item in items[:3]:
            if isinstance(item, dict):
                q   = item.get("q", "")
                idx = item.get("section_index", 0)
            else:
                q   = str(item)
                idx = 0
            if q:
                result.append(FollowUp(question=q, section_indices=[idx]))

        if len(result) >= 3:
            logger.info("추천 질문 생성 완료 — %d개", len(result))
            return result
        else:
            logger.warning("추천 질문 3개 미만 생성 (%d개) — fallback", len(result))

    except Exception as e:
        logger.warning("추천 질문 생성 실패: %s", e)

    return defaults
