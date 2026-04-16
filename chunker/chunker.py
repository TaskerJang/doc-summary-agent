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

# C-04, C-05: eval 결과 반영 — chunk_size=300 / chunk_overlap=100 채택
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 100
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

# ── #75 메타데이터 enrichment 상수 ──────────────────────────────────────────

# M-01: 연도 추출 정규식 — 2010~2039 범위 (금융 보고서 실용 범위)
_YEAR_PATTERN = re.compile(r"\b(20[1-3][0-9])년?\b")

# M-02: 섹션 유형 키워드 매핑 (우선순위 순 — 앞쪽일수록 우선)
# 설계 근거: arXiv 2402.05131 (Yepes et al.) — 금융 보고서 섹션 분류 taxonomy 참조
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
# ref: Snowflake Engineering Blog (2025) — 금융 RAG 메타데이터 필드 설계 사례
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


# ── #75 메타데이터 추출 함수 ────────────────────────────────────────────────

def _extract_doc_year(section: str, text: str) -> str | None:
    """M-01: 섹션 제목 + 본문에서 가장 많이 등장하는 연도 반환.

    동율일 경우 가장 최근 연도 우선 (Counter.most_common 정렬 활용).
    연도 없으면 None 반환.

    Args:
        section: 섹션 제목
        text: 청크 본문

    Returns:
        "2023" 형태 문자열 또는 None
    """
    combined = section + " " + text
    years = _YEAR_PATTERN.findall(combined)
    if not years:
        return None
    # 동율 시 최신 연도 우선: Counter는 삽입 순 유지 안 하므로 key로 정렬
    counter = Counter(years)
    most_common_count = counter.most_common(1)[0][1]
    candidates = [y for y, c in counter.items() if c == most_common_count]
    return max(candidates)  # 동율 → 최신 연도


def _extract_section_type(section: str, text: str) -> str | None:
    """M-02: 섹션 제목 + 본문 첫 100자 기준으로 섹션 유형 분류.

    _SECTION_TYPE_MAP 순서대로 키워드 포함 여부 확인 — 첫 매칭 반환.
    매칭 없으면 None 반환 (호출부에서 '기타'로 처리 가능).

    Args:
        section: 섹션 제목
        text: 청크 본문

    Returns:
        "실적" | "리스크" | "전망" | None
    """
    # 섹션 제목 + 본문 앞 100자만 검사 (성능 + 제목 편향 방지)
    target = (section + " " + text[:100]).lower()
    for keywords, label in _SECTION_TYPE_MAP:
        if any(kw.lower() in target for kw in keywords):
            return label
    return None


def _extract_metrics(text: str) -> list[str]:
    """M-03: 본문에서 금융 지표 키워드 스캔 후 등장 순서 유지하며 중복 제거.

    Args:
        text: 청크 본문

    Returns:
        등장한 지표 키워드 리스트 (중복 없음, 등장 순서 유지)
    """
    seen: set[str] = set()
    result: list[str] = []
    for kw in _METRIC_KEYWORDS:
        if kw in text and kw not in seen:
            seen.add(kw)
            result.append(kw)
    return result


def _get_semantic_splitter():
    """S-02: SemanticChunker 싱글턴 반환 (lazy init)."""
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
        logger.warning("SemanticChunker 초기화 실패 → MarkdownTextSplitter fallback: %s", exc)
        _semantic_splitter = None

    return _semantic_splitter


def _semantic_split(text: str) -> list[str]:
    """S-03: SemanticChunker로 텍스트 분할. 빈 리스트 반환 시 호출부에서 fallback."""
    sentences = [s for s in re.split(KOREAN_SENTENCE_SPLIT_REGEX, text) if s.strip()]
    if len(sentences) < 3:
        logger.debug("문장 수 부족(%d) → SemanticChunker 우회", len(sentences))
        return []

    splitter = _get_semantic_splitter()
    if splitter is None:
        return []

    try:
        return splitter.split_text(text)
    except Exception as exc:
        logger.warning("SemanticChunker 분할 실패 → fallback: %s", exc)
        return []


def chunk(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
) -> list[Chunk]:
    """파싱·전처리된 Markdown 텍스트를 청크 배열로 분할하여 반환한다.

    분할 계층:
      1. _split_by_heading() — #/##/### 헤더 기준 섹션 분리
      2. 섹션 내부:
         a. 표 블록 → 독립 청크
         b. chunk_size 이하 → 그대로 1청크
         c. chunk_size 초과 → SemanticChunker(bge-m3) 우선 → MarkdownTextSplitter fallback

    Returns:
        list[Chunk]: section, page, chunk_index, text, chunk_type,
                     doc_year, section_type, metrics 필드를 가진 청크 배열
    """
    if not text.strip():
        logger.warning("빈 텍스트 입력 — 청크 없음")
        return []

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
            continue

        full_text = f"{section_title}\n\n{section_body}".strip() if section_title else section_body

        if _is_table_block(section_body):
            _append_table_chunk(chunks, seen_table_keys, section_title, full_text, min_chunk_size)
            continue

        if len(full_text) <= chunk_size:
            _append_chunk(chunks, section_title, full_text, min_chunk_size, chunk_type="text")
            continue

        semantic_subs = _semantic_split(full_text)
        if semantic_subs:
            logger.debug(
                "SemanticChunker 분할 완료 (section=%r, %d → %d청크)",
                section_title[:20], len(full_text), len(semantic_subs),
            )
            for sub in semantic_subs:
                _append_chunk(chunks, section_title, sub, min_chunk_size, chunk_type="text")
        else:
            logger.debug("MarkdownTextSplitter fallback 적용 (section=%r)", section_title[:20])
            if md_splitter is None:
                md_splitter = MarkdownTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            for sub in md_splitter.split_text(full_text):
                _append_chunk(chunks, section_title, sub, min_chunk_size, chunk_type="text")

    for i, c in enumerate(chunks):
        c["chunk_index"] = i

    return chunks


def _split_by_heading(text: str) -> list[tuple[str, str]]:
    """# ## ### 기준으로 섹션을 분리한다."""
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
    """C-01: 섹션 제목 또는 본문 첫 줄이 SKIP_SECTION_KEYWORDS에 해당하면 True."""
    lines = body.splitlines()
    first_line = lines[0] if lines else ""
    target = (title + " " + first_line).lower()
    return any(kw in target for kw in SKIP_SECTION_KEYWORDS)


def _is_table_block(text: str) -> bool:
    """텍스트의 TABLE_LINE_RATIO 이상이 표(|로 시작하거나 끝나는) 라인이면 표 블록으로 판단."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    table_lines = [ln for ln in lines if ln.strip().startswith("|") or ln.strip().endswith("|")]
    return len(table_lines) / len(lines) >= TABLE_LINE_RATIO


def _table_key(text: str) -> str:
    """C-02: 표 청크 중복 감지용 키 — 첫 줄에서 |·공백 제거 후 60자."""
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
    """C-02: 중복 표 청크 건너뜀 + 최소 크기 검증 후 추가."""
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
    """C-04: 최소 크기 검증 후 청크 추가. #75 메타데이터 자동 enrichment 포함."""
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
