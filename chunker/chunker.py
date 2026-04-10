import logging
import re
from typing import TypedDict

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

# C-04, C-05: eval 결과 반영 — chunk_size=500 / chunk_overlap=100 채택
# (6개 조합 비교 기준 Faithfulness 최고: 30/40, 75.0%)
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MIN_CHUNK_SIZE = 50

# 표 판단 기준 비율 — 실제 문서 테스트 후 조정 예정
TABLE_LINE_RATIO = 0.5

# pymupdf4llm 출력 기준 최대 ### 까지만 섹션 경계로 인식
_HEADING_PATTERN = re.compile(r"^(#{1,3} .+)$", re.MULTILINE)


class Chunk(TypedDict):
    section: str
    page: int | None   # 전처리 단계에서 페이지 번호 제거됨 → 추후 확장
    chunk_index: int
    text: str


def chunk(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
) -> list[Chunk]:
    """
    파싱·전처리된 Markdown 텍스트를 청크 배열로 분할하여 반환한다.

    Args:
        text: preprocessor.clean() 결과 텍스트
        chunk_size: 청크 최대 글자 수
        chunk_overlap: 청크 간 오버랩 글자 수
        min_chunk_size: 이 값 미만인 청크는 건너뜀

    Returns:
        list[Chunk]: section, page, chunk_index, text 필드를 가진 청크 배열
    """
    if not text.strip():
        logger.warning("빈 텍스트 입력 — 청크 없음")
        return []

    sections = _split_by_heading(text)

    # C-03: heading 기반 분리 실패 시 \n\n 단락 기준 fallback
    if not sections:
        logger.debug("heading 없음 → \\n\\n 단락 기준 fallback 분리 적용")
        # 문자열을 단락 단위로 분리하고, 공백을 제거한 뒤, 튜플 리스트로 만드는 한 줄 표현식
        sections = [("", para.strip()) for para in text.split("\n\n") if para.strip()]

    splitter: MarkdownTextSplitter | None = None  # lazy initialization
    chunks: list[Chunk] = []
    seen_table_keys: set[str] = set()  # C-02: 중복 표 청크 추적

    for section_title, section_body in sections:

        # C-01: Compliance/면책 고지 섹션 건너뜀
        if _is_skip_section(section_title, section_body):
            logger.debug("skip 섹션 건너뜀: %r", section_title[:30])
            continue

        # 섹션 제목을 본문 앞에 반복 포함 → 문맥 유지
        full_text = f"{section_title}\n\n{section_body}".strip() if section_title else section_body

        # 표 블록 → 독립 청크 (C-02: 중복 표 제거 포함)
        if _is_table_block(section_body):
            _append_table_chunk(chunks, seen_table_keys, section_title, full_text, min_chunk_size)
            continue

        # chunk_size 이내면 그대로, 초과하면 재분할 (lazy initialization)
        if len(full_text) <= chunk_size:
            # 텍스트가 chunk_size 이하 — 분할 없이 바로 추가
            _append_chunk(chunks, section_title, full_text, min_chunk_size)
        else:
            # 텍스트가 chunk_size 초과 — 분할 필요
            if splitter is None:
                # 아직 생성된 적 없으면 이 시점에 생성 (Lazy Initialization)
                splitter = MarkdownTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            for sub in splitter.split_text(full_text):
                # 분할된 각 조각을 개별 청크로 추가
                _append_chunk(chunks, section_title, sub, min_chunk_size)

    # 리스트를 순회하면서 각 항목에 순번(인덱스)을 부여
    for i, c in enumerate(chunks):
        c["chunk_index"] = i

    return chunks

def _split_by_heading(text: str) -> list[tuple[str, str]]:
    """
    # ## ### 기준으로 섹션을 분리한다.
    heading이 없으면 빈 리스트 반환 → 호출부에서 fallback 처리.

    반환값: (section_title, body) 튜플 배열
    section_title이 빈 문자열이면 heading 이전 텍스트(전문)
    """

    # 1단계 — 헤더 위치 목록 확보
    matches = list(_HEADING_PATTERN.finditer(text))

    # 2단계 — 헤더가 없으면 조기 반환
    if not matches:
        return []

    sections: list[tuple[str, str]] = []

    # 3단계 — 첫 헤더 이전 텍스트 처리
    if matches[0].start() > 0:

        # 첫 헤더 이전까지 슬라이싱
        pre = text[:matches[0].start()].strip()

        # 공백만 있는 경우 제외
        if pre:
            sections.append(("", pre))

    for i, match in enumerate(matches):
        title = match.group(1).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:  # 본문 없는 섹션 건너뜀
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
    _append_chunk(chunks, section, text, min_chunk_size)


def _append_chunk(
    chunks: list[Chunk],
    section: str,
    text: str,
    min_chunk_size: int,
) -> None:
    """C-04: 최소 크기 검증 후 청크 추가. 미달 시 로깅 후 건너뜀."""
    stripped = text.strip()
    if len(stripped) < min_chunk_size:
        logger.debug("최소 크기 미달 청크 건너뜀 (section=%r, len=%d)", section, len(stripped))
        return

    chunks.append(Chunk(
        section=section,
        page=None,
        chunk_index=0,   # 최종 loop에서 덮어씀
        text=stripped,
    ))
