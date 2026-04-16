import logging
import re
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
# (hyperparam_v3, 6개 조합 비교 기준 전 지표 1위: ROUGE-L 0.3151, NumAcc 0.6763, Faithfulness 29/40 72.5%)
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MIN_CHUNK_SIZE = 50

# S-01: SemanticChunker 설정
# breakpoint_threshold_type 선택 근거:
#   - Greg Kamradt (2024) 원저자 및 NirDiamant RAG_Techniques: percentile=90 권장
#   - Superlinked VectorHub 프로덕션 사례: percentile=80 사용
#   - arXiv 2410.13070: 도메인별 최적값 상이 → 금융 문서 특성상 85 시작점 채택 (튜닝 필요)
SEMANTIC_BREAKPOINT_TYPE = "percentile"
SEMANTIC_BREAKPOINT_THRESHOLD: float = 85.0  # 상위 15% 유사도 급락 지점 분할 — 튜닝 대상

# S-01: 한국어 금융 문서 sentence 분리 정규식
# 기본값 r"(?<=[.?!])\s+" 은 영어 기준 — 한국어는 마침표+줄바꿈, 줄바꿈 단독 패턴 추가 필요
# ref: LangChain SemanticChunker API (sentence_split_regex 파라미터)
KOREAN_SENTENCE_SPLIT_REGEX = r"(?<=[.?!。])\s+|\n{1,2}"

# 표 판단 기준 비율 — 실제 문서 테스트 후 조정 예정
TABLE_LINE_RATIO = 0.5

# pymupdf4llm 출력 기준 최대 ### 까지만 섹션 경계로 인식
_HEADING_PATTERN = re.compile(r"^(#{1,3} .+)$", re.MULTILINE)

# S-02: SemanticChunker 싱글턴 — 프로세스당 1회 초기화 (bge-m3 모델 재사용)
_semantic_splitter = None


class Chunk(TypedDict):
    section: str
    page: int | None        # 전처리 단계에서 페이지 번호 제거됨 → 추후 확장
    chunk_index: int
    text: str
    chunk_type: Literal["text", "table"]  # #75 메타데이터 enrichment 선행 확장


def _get_semantic_splitter():
    """S-02: SemanticChunker 싱글턴 반환 (lazy init).

    bge-m3는 벡터 인덱서에서 이미 로드된 모델을 재사용한다.
    langchain-experimental 미설치 또는 모델 로드 실패 시 None 반환
    → 호출부에서 MarkdownTextSplitter fallback 적용.

    ref:
    - LangChain SemanticChunker API:
      https://python.langchain.com/api_reference/experimental/text_splitter/
      langchain_experimental.text_splitter.SemanticChunker.html
    - HuggingFaceBgeEmbeddings 사용법:
      https://huggingface.co/BAAI/bge-small-en-v1.5 (langchain 예시)
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
            # chunking 단계에서는 passage/query prefix 불필요
            # (retrieval 단계의 "passage: " prefix와 구분)
            query_instruction="",
        )
        _semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
            breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_THRESHOLD,
            # S-01: 한국어 문장 분리 정규식 명시
            # 기본값 r"(?<=[.?!])\s+" 은 영어 전용 → 한국어 금융 문서 미적용
            sentence_split_regex=KOREAN_SENTENCE_SPLIT_REGEX,
        )
        logger.info(
            "SemanticChunker 초기화 완료 (bge-m3, %s=%.1f)",
            SEMANTIC_BREAKPOINT_TYPE,
            SEMANTIC_BREAKPOINT_THRESHOLD,
        )
    except Exception as exc:
        # ImportError(langchain-experimental 미설치), OSError(모델 미다운로드) 등
        logger.warning("SemanticChunker 초기화 실패 → MarkdownTextSplitter fallback: %s", exc)
        _semantic_splitter = None

    return _semantic_splitter


def _semantic_split(text: str) -> list[str]:
    """S-03: SemanticChunker로 텍스트 분할. 빈 리스트 반환 시 호출부에서 fallback.

    단문 섹션(3문장 미만) 우회 이유:
      SemanticChunker는 내부에서 np.percentile(distances, threshold) 호출 시
      distances 배열이 비어있으면 IndexError 발생 (알려진 버그).
      ref: GitHub Issue #17106, #23250 (langchain-ai/langchain)
      → 문장 수 < 3이면 SemanticChunker 우회, MarkdownTextSplitter fallback 유도.

    Args:
        text: 분할 대상 텍스트 (chunk_size 초과 섹션 본문)

    Returns:
        분할된 텍스트 리스트. 실패·우회 시 빈 리스트 반환.
    """
    # 문장 수 사전 체크 — IndexError 방어
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
         a. 표 블록 → 독립 청크 (중복 제거 포함)
         b. chunk_size 이하 → 그대로 1청크
         c. chunk_size 초과 → SemanticChunker(bge-m3) 우선
                             → 실패 시 MarkdownTextSplitter(300자) fallback

    설계 근거:
      - 헤더 우선 + semantic 보조: Snowflake(2025) SEC RAG 실험에서
        markdown-aware chunking이 semantic 단독 대비 5~10%p 우위
      - min_chunk_size 하한: FloTorch(2026) — semantic 청킹 파편(평균 43토큰)이
        end-to-end 정확도 54%까지 떨어뜨린 사례 → 50자 하한으로 방어

    Args:
        text: preprocessor.clean() 결과 텍스트
        chunk_size: 청크 최대 글자 수 (ChatSettings #67 연동)
        chunk_overlap: MarkdownTextSplitter fallback용 오버랩 글자 수
        min_chunk_size: 이 값 미만인 청크는 건너뜀

    Returns:
        list[Chunk]: section, page, chunk_index, text, chunk_type 필드를 가진 청크 배열
    """
    if not text.strip():
        logger.warning("빈 텍스트 입력 — 청크 없음")
        return []

    sections = _split_by_heading(text)

    # C-03: heading 기반 분리 실패 시 \n\n 단락 기준 fallback
    if not sections:
        logger.debug("heading 없음 → \\n\\n 단락 기준 fallback 분리 적용")
        sections = [("", para.strip()) for para in text.split("\n\n") if para.strip()]

    md_splitter: MarkdownTextSplitter | None = None  # lazy initialization
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

        # chunk_size 이내 → 분할 없이 바로 추가
        if len(full_text) <= chunk_size:
            _append_chunk(chunks, section_title, full_text, min_chunk_size, chunk_type="text")
            continue

        # chunk_size 초과 → S-03: SemanticChunker 우선 시도
        semantic_subs = _semantic_split(full_text)
        if semantic_subs:
            logger.debug(
                "SemanticChunker 분할 완료 (section=%r, %d → %d청크)",
                section_title[:20], len(full_text), len(semantic_subs),
            )
            for sub in semantic_subs:
                _append_chunk(chunks, section_title, sub, min_chunk_size, chunk_type="text")
        else:
            # fallback: MarkdownTextSplitter 300자 고정 분할
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
    """# ## ### 기준으로 섹션을 분리한다.
    heading이 없으면 빈 리스트 반환 → 호출부에서 fallback 처리.

    반환값: (section_title, body) 튜플 배열
    section_title이 빈 문자열이면 heading 이전 텍스트(전문)
    """
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
        chunk_type=chunk_type,
    ))
