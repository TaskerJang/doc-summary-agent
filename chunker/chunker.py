import logging
import re
from collections import Counter
from typing import Literal, TypedDict

from langchain_text_splitters import MarkdownTextSplitter

logger = logging.getLogger(__name__)

# C-01: Compliance/면책 고지 섹션 키워드 (N-08 이관) — 소문자로 정규화
SKIP_SECTION_KEYWORDS: frozenset[str] = frozenset({
    "compliance",
    "면책",
    "투자등급",
    "산업 투자의견",
    "조사분석자료",
    "본 자료는",
    "보도출처",
})

# C-04, C-05: chunk_size=700 / chunk_overlap=200
# 변경 이유: chunk_size=300은 대형 문서(40k자+)에서 청크 80개+ 발생 → LLM 호출 폭발
# 700으로 키우면 동일 문서 기준 청크 수 약 절반 감소 → 처리 속도 개선
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MIN_CHUNK_SIZE = 50

# S-01: SemanticChunker 설정
SEMANTIC_BREAKPOINT_TYPE = "percentile"
SEMANTIC_BREAKPOINT_THRESHOLD: float = 85.0

# S-01: 한국어 금융 문서 sentence 분리 정규식
KOREAN_SENTENCE_SPLIT_REGEX = r"(?<=[.?!。])\s+|\n{1,2}"

# 표 판단 기준 비율
TABLE_LINE_RATIO = 0.5

# pymupdf4llm 출력 기준 최대 ### 까지만 섹션 경계로 인식
_HEADING_PATTERN = re.compile(r"^(#{1,3} .+)$", re.MULTILINE)

# S-02: SemanticChunker 싱글턴
_semantic_splitter = None

# ── #136 청킹 전략 ────────────────────────────────────────────────────────
#
# 전략 사다리 — "구조를 얼마나 아는가" 를 단계적으로 올린다.
#
# | 전략       | 섹션분할 | skip섹션 | 표인지 | 분할기                  |
# |------------|---------|---------|-------|------------------------|
# | fixed      |    ✕    |    ✕    |   ✕   | 문자 슬라이딩            |
# | recursive  |    ○    |    ○    |   ✕   | MarkdownTextSplitter    |
# | semantic   |    ○    |    ○    |   ✕   | SemanticChunker         |
# | structural |    ○    |    ○    |   ○   | Semantic + fallback     |
#
# 설계 근거 1 — skip 섹션(면책·컴플라이언스 제거)은 청킹 전략이 아니라 *전처리*다.
#   전략마다 켜고 끄면 "노이즈 청크 유무" 라는 무관한 변수가 섞여 비교가 오염된다.
#   fixed 만 예외인 이유는, 그것이 "아무 구조도 모르는 기준선" 의 정의이기 때문.
#
# 설계 근거 2 — semantic 과 structural 의 차이를 *표 인지 + 크기 단축* 으로만 좁혔다.
#   그래야 "표를 알아보는 것이 numerical 질문에 얼마나 기여하는가" 가 단독으로 측정된다.
#   부수 효과로 비용도 잡힌다. semantic 을 문서 전체에 걸면 40k자 문서에서
#   bge-m3 CPU 인코딩이 폭발하는데, 섹션 단위로 자르고 들어가면 견딜 만하다.
#
# 설계 근거 3 (#137 안 A) — fixed 도 chunk_overlap 을 동일하게 적용한다.
#   overlap=0 으로 두면 fixed 의 약점이 더 극적으로 드러나지만,
#   "fixed 가 진 이유는 overlap 이 없어서" 라는 반박을 허용하게 된다.
#   파라미터를 통일해야 청킹 방식 자체의 차이로 결론지을 수 있다.
#
ChunkStrategy = Literal["fixed", "recursive", "semantic", "structural"]
DEFAULT_STRATEGY: ChunkStrategy = "structural"

# ── #75 메타데이터 enrichment 상수 ──────────────────────────────────────────

# M-01: 연도 추출 정규식 — 2010~2039 범위 (금융 보고서 실용 범위)
_YEAR_PATTERN = re.compile(r"\b(20[1-3][0-9])년?\b")

# M-02: 섹션 유형 키워드 매핑 (우선순위 순 — 앞쪽일수록 우선)
_SECTION_TYPE_MAP: list[tuple[frozenset[str], str]] = [
    (
        frozenset(["실적", "매출", "영업이익", "순이익", "매출액", "revenue", "earnings", "profit", "손익"]),
        "실적",
    ),
    (
        frozenset(["리스크", "위험", "risk", "충당금", "부실", "손실", "불확실"]),
        "리스크",
    ),
    (
        frozenset(["전망", "outlook", "guidance", "목표", "계획", "전략", "성장", "로드맵", "방향"]),
        "전망",
    ),
]

# M-03: 금융 지표 키워드 집합
_METRIC_KEYWORDS: frozenset[str] = frozenset([
    "영업이익", "순이익", "매출", "매출액", "ROE", "ROA", "EPS", "PER", "PBR",
    "BIS", "부채비율", "자본", "자산", "배당", "배당수익률", "EBITDA",
    "영업현금흐름", "잉여현금흐름", "FCF", "영업이익률", "순이익률",
    "시가총액", "주가", "EV", "ROIC", "레버리지",
])


class Chunk(TypedDict):
    section: str
    page: int | None
    chunk_index: int
    text: str
    chunk_type: Literal["text", "table"]  # #59에서 추가
    # --- #75 추가 필드 ---
    doc_year: str | None       # "2023", "2024" 등 정규식 추출 (최빈 연도)
    section_type: str | None   # "실적" | "리스크" | "전망" | "기타"
    metrics: list[str]         # ["영업이익", "매출", "ROE"] 등


class ChunkStats(TypedDict):
    """#136 실행 통계 — 측정 결과 해석에 필요.

    특히 semantic_bypassed 가 크면 "Semantic 전략을 쟀다" 는 주장 자체가 약해진다.
    이 숫자를 leaderboard 에 함께 실어야 정직한 결과가 된다.
    """
    strategy: str
    strict: bool
    semantic_bypassed: int    # 문장 수 부족으로 SemanticChunker 우회한 횟수
    semantic_fallback: int    # Semantic 실패 → Markdown fallback 횟수 (strict면 항상 0)
    table_chunks: int
    skipped_sections: int
    total_chunks: int
    avg_chunk_len: float


_last_stats: ChunkStats | None = None


def get_last_chunk_stats() -> ChunkStats | None:
    """#136 직전 chunk() 호출의 실행 통계를 반환한다.

    run_eval 이 결과 JSON 에 함께 기록하기 위한 용도.
    """
    return _last_stats


# ── #75 메타데이터 추출 함수 ────────────────────────────────────────────────

def _extract_doc_year(section: str, text: str) -> str | None:
    combined = section + " " + text
    years = _YEAR_PATTERN.findall(combined)
    if not years:
        return None
    counter = Counter(years)
    most_common_count = counter.most_common(1)[0][1]
    candidates = [y for y, c in counter.items() if c == most_common_count]
    return max(candidates)


def _extract_section_type(section: str, text: str) -> str | None:
    target = (section + " " + text[:100]).lower()
    for keywords, label in _SECTION_TYPE_MAP:
        if any(kw.lower() in target for kw in keywords):
            return label
    return None


def _extract_metrics(text: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for kw in _METRIC_KEYWORDS:
        if kw in text and kw not in seen:
            seen.add(kw)
            result.append(kw)
    return result


def _get_semantic_splitter(strict: bool = False):
    """#138: strict=True 면 초기화 실패 시 조용히 넘어가지 않고 예외를 던진다.

    prod 경로에서 fallback 은 옳은 동작이지만, 비교 측정에서는 치명적이다.
    strategy="semantic" 으로 돌린 결과가 실제로는 recursive 결과일 수 있고,
    그 상태로 leaderboard 에 "Semantic" 이라고 기록하면 결론 전체가 틀어진다.
    PR #134 에서 겪은 것과 같은 종류의 오염이다.
    """
    global _semantic_splitter
    if _semantic_splitter is not None:
        return _semantic_splitter

    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings

        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="",
        )
        _semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
            breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_THRESHOLD,
            sentence_split_regex=KOREAN_SENTENCE_SPLIT_REGEX,
        )
        logger.info(
            "SemanticChunker 초기화 완료 (bge-m3, %s=%.1f)",
            SEMANTIC_BREAKPOINT_TYPE,
            SEMANTIC_BREAKPOINT_THRESHOLD,
        )
    except Exception as exc:
        if strict:
            raise RuntimeError(
                f"#138 strict 모드: SemanticChunker 초기화 실패 — 측정 오염 방지를 위해 중단. 원인: {exc}"
            ) from exc
        logger.warning("SemanticChunker 초기화 실패 → MarkdownTextSplitter fallback: %s", exc)
        _semantic_splitter = None

    return _semantic_splitter


def _semantic_split(text: str, strict: bool, stats: dict) -> list[str]:
    """SemanticChunker 분할. 실패 시 빈 리스트 반환 (호출부가 fallback 판단).

    #138: 문장 수 부족 우회는 *예외가 아니라 카운트*다. 정상 동작이지만,
    "semantic 결과의 몇 %가 실제로는 통짜 청크였나" 를 알아야 결과 해석이 된다.
    """
    sentences = [s for s in re.split(KOREAN_SENTENCE_SPLIT_REGEX, text) if s.strip()]
    if len(sentences) < 3:
        logger.debug("문장 수 부족(%d) → SemanticChunker 우회", len(sentences))
        stats["semantic_bypassed"] += 1
        return []

    splitter = _get_semantic_splitter(strict=strict)
    if splitter is None:
        return []

    try:
        return splitter.split_text(text)
    except Exception as exc:
        if strict:
            raise RuntimeError(
                f"#138 strict 모드: SemanticChunker 분할 실패 — 측정 오염 방지를 위해 중단. 원인: {exc}"
            ) from exc
        logger.warning("SemanticChunker 분할 실패 → fallback: %s", exc)
        return []


def _fixed_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """#137 Fixed 전략 — 문서 구조를 전혀 모르는 기준선.

    heading / skip 섹션 / 표 블록 감지를 전부 우회하고 문자 수로만 자른다.
    표가 중간에서 잘려 수치가 헤더와 분리되는 상황을 *의도적으로* 만든다.

    측정 전 가설 (#137):
      - numerical: 가장 불리. 표가 잘려 수치와 헤더가 분리됨
      - negative:  할루시네이션 증가 위험. 잘린 청크가 근거처럼 보임
      - factual:   차이 작을 것. 단문 답은 청크 위치와 무관
      - summary:   불리. 의미 경계 무시
    가설이 틀리는 것도 결과다. #140 측정 후 대조한다.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap({chunk_overlap})은 chunk_size({chunk_size})보다 작아야 한다"
        )

    step = chunk_size - chunk_overlap
    parts: list[str] = []
    for i in range(0, len(text), step):
        piece = text[i: i + chunk_size]
        if piece.strip():
            parts.append(piece)
        if i + chunk_size >= len(text):
            break
    return parts


def chunk(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
    strategy: ChunkStrategy = DEFAULT_STRATEGY,
    strict: bool = False,
) -> list[Chunk]:
    """문서 텍스트를 청크로 분할한다.

    Args:
        strategy: #136 청킹 전략. 기본 "structural" 은 기존 동작과 완전히 동일하다.
                  prod 경로(main.py / ui/app.py)는 인자를 넘기지 않으므로 영향 없음.
        strict:   #138 실험 모드. SemanticChunker 조용한 fallback 을 차단한다.
    """
    global _last_stats

    stats: dict = {
        "strategy": strategy,
        "strict": strict,
        "semantic_bypassed": 0,
        "semantic_fallback": 0,
        "table_chunks": 0,
        "skipped_sections": 0,
    }

    if not text.strip():
        logger.warning("빈 텍스트 입력 — 청크 없음")
        _last_stats = ChunkStats(**stats, total_chunks=0, avg_chunk_len=0.0)
        return []

    logger.info("청킹 시작 — strategy=%s strict=%s size=%d overlap=%d",
                strategy, strict, chunk_size, chunk_overlap)

    # ── fixed: 구조를 전혀 보지 않는다. 섹션·skip·표 전부 우회 ──────────
    if strategy == "fixed":
        chunks: list[Chunk] = []
        for piece in _fixed_split(text, chunk_size, chunk_overlap):
            _append_chunk(chunks, "", piece, min_chunk_size, chunk_type="text")
        _finalize(chunks, stats)
        return chunks

    # ── 이하 recursive / semantic / structural 공통 전처리 ──────────────
    sections = _split_by_heading(text)

    if not sections:
        logger.debug("heading 없음 → \\n\\n 단락 기준 fallback 분리 적용")
        sections = [("", para.strip()) for para in text.split("\n\n") if para.strip()]

    md_splitter: MarkdownTextSplitter | None = None
    chunks: list[Chunk] = []
    seen_table_keys: set[str] = set()

    for section_title, section_body in sections:

        if _is_skip_section(section_title, section_body):
            logger.debug("skip 섹션 건너뜀: %r", section_title[:30])
            stats["skipped_sections"] += 1
            continue

        full_text = f"{section_title}\n\n{section_body}".strip() if section_title else section_body

        # 표 인지와 크기 단축은 structural 만의 특성이다.
        # recursive / semantic 에서 이를 끄는 이유는 설계 근거 2 참조.
        if strategy == "structural":
            if _is_table_block(section_body):
                before = len(chunks)
                _append_table_chunk(chunks, seen_table_keys, section_title, full_text, min_chunk_size)
                stats["table_chunks"] += len(chunks) - before
                continue

            if len(full_text) <= chunk_size:
                _append_chunk(chunks, section_title, full_text, min_chunk_size, chunk_type="text")
                continue

        if strategy == "recursive":
            if md_splitter is None:
                md_splitter = MarkdownTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            for sub in md_splitter.split_text(full_text):
                _append_chunk(chunks, section_title, sub, min_chunk_size, chunk_type="text")
            continue

        # semantic / structural — SemanticChunker 시도
        semantic_subs = _semantic_split(full_text, strict=strict, stats=stats)
        if semantic_subs:
            logger.debug(
                "SemanticChunker 분할 완료 (section=%r, %d → %d청크)",
                section_title[:20], len(full_text), len(semantic_subs),
            )
            for sub in semantic_subs:
                _append_chunk(chunks, section_title, sub, min_chunk_size, chunk_type="text")
        else:
            # 문장 수 부족(bypass) 또는 실패(fallback). strict=True 면 실패는 이미 예외로 끊겼다.
            if stats["semantic_bypassed"] == 0:
                stats["semantic_fallback"] += 1
            logger.debug("MarkdownTextSplitter fallback 적용 (section=%r)", section_title[:20])
            if md_splitter is None:
                md_splitter = MarkdownTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            for sub in md_splitter.split_text(full_text):
                _append_chunk(chunks, section_title, sub, min_chunk_size, chunk_type="text")

    _finalize(chunks, stats)
    return chunks


def _finalize(chunks: list[Chunk], stats: dict) -> None:
    """chunk_index 재부여 + #136 실행 통계 확정."""
    global _last_stats

    for i, c in enumerate(chunks):
        c["chunk_index"] = i

    total = len(chunks)
    avg_len = (sum(len(c["text"]) for c in chunks) / total) if total else 0.0
    _last_stats = ChunkStats(**stats, total_chunks=total, avg_chunk_len=round(avg_len, 1))

    logger.info(
        "청킹 완료 — strategy=%s 총 %d청크 평균 %.1f자 "
        "(표 %d, skip섹션 %d, semantic 우회 %d, fallback %d)",
        stats["strategy"], total, avg_len,
        stats["table_chunks"], stats["skipped_sections"],
        stats["semantic_bypassed"], stats["semantic_fallback"],
    )


def _split_by_heading(text: str) -> list[tuple[str, str]]:
    matches = list(_HEADING_PATTERN.finditer(text))

    if not matches:
        return []

    sections: list[tuple[str, str]] = []

    if matches[0].start() > 0:
        pre = text[:matches[0].start()].strip()
        if pre:
            sections.append(("", pre))

    for i, match in enumerate(matches):
        title = match.group(1).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append((title, body))

    return sections


def _is_skip_section(title: str, body: str) -> bool:
    lines = body.splitlines()
    first_line = lines[0] if lines else ""
    target = (title + " " + first_line).lower()
    return any(kw in target for kw in SKIP_SECTION_KEYWORDS)


def _is_table_block(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    table_lines = [ln for ln in lines if ln.strip().startswith("|") or ln.strip().endswith("|")]
    return len(table_lines) / len(lines) >= TABLE_LINE_RATIO


def _table_key(text: str) -> str:
    lines = text.splitlines()
    first_line = lines[0] if lines else text
    return re.sub(r"[|\s]", "", first_line)[:60]


def _append_table_chunk(
    chunks: list[Chunk],
    seen_table_keys: set[str],
    section: str,
    text: str,
    min_chunk_size: int,
) -> None:
    key = _table_key(text)
    if key in seen_table_keys:
        logger.debug("중복 표 청크 건너뜀 (key=%r)", key)
        return
    seen_table_keys.add(key)
    _append_chunk(chunks, section, text, min_chunk_size, chunk_type="table")


def _append_chunk(
    chunks: list[Chunk],
    section: str,
    text: str,
    min_chunk_size: int,
    chunk_type: Literal["text", "table"] = "text",
) -> None:
    stripped = text.strip()
    if len(stripped) < min_chunk_size:
        logger.debug("최소 크기 미달 청크 건너뜀 (section=%r, len=%d)", section, len(stripped))
        return

    chunks.append(Chunk(
        section=section,
        page=None,
        chunk_index=0,
        text=stripped,
        chunk_type=chunk_type,
        doc_year=_extract_doc_year(section, stripped),
        section_type=_extract_section_type(section, stripped),
        metrics=_extract_metrics(stripped),
    ))
