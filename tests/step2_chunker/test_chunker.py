"""
chunker.py 단위 테스트 — 이슈 #59 SemanticChunking + #75 메타데이터 enrichment

실행 방법:
    uv run python tests/step2_chunker/test_chunker.py

테스트 구성:
    - bge-m3 모델 불필요 (SemanticChunker를 Mock으로 우회)
    - 모든 케이스는 fallback(MarkdownTextSplitter) 또는 순수 로직만 검증

검증 항목:
    T-01: 빈 텍스트 입력 → 빈 리스트 반환
    T-02: heading 없는 텍스트 → \\n\\n 단락 기준 fallback 분리
    T-03: 표 블록 → chunk_type="table" 독립 청크
    T-04: 중복 표 블록 → 1개만 반환
    T-05: Compliance/면책 섹션 → skip
    T-06: chunk_size 이하 섹션 → 1청크 그대로
    T-07: chunk_size 초과 → SemanticChunker 실패 시 MarkdownTextSplitter fallback
    T-08: 단문 섹션(2문장) → _semantic_split()이 [] 반환 (IndexError 방어)
    T-09: chunk_index 순번 연속 부여
    T-10: chunk_type 필드 존재 확인 (#75 선행 확장)
    T-11: doc_year — 본문 연도 추출 정확도
    T-12: section_type — 섹션 유형 분류 정확도
    T-13: metrics — 금융 지표 키워드 추출
    T-14: 메타데이터 필드 모두 존재 + 타입 유효성
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chunker.chunker import (
    chunk,
    _semantic_split,
    _extract_doc_year,
    _extract_section_type,
    _extract_metrics,
)

# ── 공통 픽스처 ────────────────────────────────────────────────────────────────

SHORT_SECTION = """## 실적 요약

2023년 영업이익은 1,200억 원으로 전년 대비 15% 증가했습니다.
"""

LONG_SECTION = """## 사업 개요

당사는 글로벌 금융 솔루션을 제공하는 기업으로, 2023년 기준 자산 총계 50조 원을 기록했습니다.
주요 사업 부문은 리테일 뱅킹, 기업 금융, 자산 관리로 구성되어 있으며 각 부문별 수익 기여도는 상이합니다.
리테일 뱅킹 부문은 전체 수익의 42%를 차지하며 안정적인 성장세를 유지하고 있습니다.
기업 금융 부문은 대형 프로젝트 파이낸싱과 채권 발행 주선을 핵심으로 수익을 창출합니다.
자산 관리 부문은 펀드 운용 보수 및 투자 자문 수수료로 수익을 구성하며 고성장 중입니다.
리스크 관리 체계는 Basel III 기준을 준수하며 BIS 비율은 15.2%로 양호한 수준입니다.
향후 전략은 디지털 전환 가속화와 해외 시장 진출 확대를 두 축으로 설정했습니다.
""" * 3

TABLE_SECTION = """## 재무 현황

| 항목 | 2022 | 2023 |
|---|---|---|
| 매출 | 8,200 | 9,400 |
| 영업이익 | 1,040 | 1,200 |
| 순이익 | 780 | 910 |
"""

COMPLIANCE_SECTION = """## compliance 안내

본 자료는 투자 권유를 목적으로 하지 않습니다.
"""

HEADING_DOC = SHORT_SECTION + "\n" + TABLE_SECTION + "\n" + COMPLIANCE_SECTION

NO_HEADING_TEXT = (
    "첫 번째 단락입니다. 2023년 영업이익은 1,200억 원으로 전년 대비 15% 증가했습니다.\n\n"
    "두 번째 단락입니다. 리테일 뱅킹 부문이 전체 수익의 42%를 차지하며 성장 중입니다.\n\n"
    "세 번째 단락입니다. 자산 관리 부문은 펀드 운용 보수로 수익을 구성하며 고성장 중입니다."
)


# ── 기존 테스트 (T-01 ~ T-10) ─────────────────────────────────────────────────

def test_t01_empty_input():
    """T-01: 빈 텍스트 입력 → 빈 리스트 반환"""
    result = chunk("")
    assert result == [], f"빈 리스트 기대, got {result}"
    result2 = chunk("   \n  ")
    assert result2 == [], f"공백만 있는 입력도 빈 리스트 기대"
    print("  ✅ T-01 통과")


def test_t02_no_heading_fallback():
    """T-02: heading 없는 텍스트 → \\n\\n 단락 기준 fallback 분리"""
    result = chunk(NO_HEADING_TEXT, chunk_size=500)
    assert len(result) >= 1, (
        f"최소 1개 청크 기대. "
        f"단락 길이 확인: {[len(p) for p in NO_HEADING_TEXT.split(chr(10)*2)]}"
    )
    assert all("text" in c for c in result), "모든 청크에 text 키 존재"
    print(f"  ✅ T-02 통과 ({len(result)}개 청크)")


def test_t03_table_chunk_type():
    """T-03: 표 블록 → chunk_type='table' 독립 청크"""
    result = chunk(TABLE_SECTION, chunk_size=500)
    table_chunks = [c for c in result if c["chunk_type"] == "table"]
    assert len(table_chunks) >= 1, "표 청크 최소 1개 기대"
    print(f"  ✅ T-03 통과 (표 청크 {len(table_chunks)}개)")


def test_t04_duplicate_table_dedup():
    """T-04: 동일 표 중복 입력 → 1개만 반환"""
    duplicated = TABLE_SECTION + "\n" + TABLE_SECTION
    result = chunk(duplicated, chunk_size=500)
    table_chunks = [c for c in result if c["chunk_type"] == "table"]
    assert len(table_chunks) == 1, f"중복 표 제거 후 1개 기대, got {len(table_chunks)}"
    print("  ✅ T-04 통과")


def test_t05_skip_compliance():
    """T-05: Compliance/면책 섹션 → 청크에 포함되지 않음"""
    result = chunk(HEADING_DOC, chunk_size=500)
    texts = " ".join(c["text"] for c in result)
    assert "본 자료는 투자 권유를 목적으로 하지 않습니다" not in texts, \
        "Compliance 섹션이 청크에 포함됨"
    print("  ✅ T-05 통과")


def test_t06_short_section_single_chunk():
    """T-06: chunk_size 이하 섹션 → 분할 없이 1청크"""
    result = chunk(SHORT_SECTION, chunk_size=500)
    text_chunks = [c for c in result if c["chunk_type"] == "text"]
    assert len(text_chunks) == 1, f"1청크 기대, got {len(text_chunks)}"
    print("  ✅ T-06 통과")


def test_t07_long_section_fallback():
    """T-07: chunk_size 초과 + SemanticChunker 강제 실패 → MarkdownTextSplitter fallback"""
    with patch("chunker.chunker._get_semantic_splitter", return_value=None):
        result = chunk(LONG_SECTION, chunk_size=300, chunk_overlap=50)
    text_chunks = [c for c in result if c["chunk_type"] == "text"]
    assert len(text_chunks) > 1, f"fallback 분할 후 2개 이상 기대, got {len(text_chunks)}"
    for c in text_chunks:
        assert len(c["text"]) >= 50, f"min_chunk_size 미달: {len(c['text'])}자"
    print(f"  ✅ T-07 통과 (fallback {len(text_chunks)}개 청크)")


def test_t08_short_text_semantic_skip():
    """T-08: 단문(2문장) → _semantic_split()이 [] 반환 (IndexError 방어)"""
    short = "영업이익이 증가했습니다. 전년 대비 15% 성장입니다."
    result = _semantic_split(short)
    assert result == [], f"단문은 [] 기대, got {result}"
    print("  ✅ T-08 통과")


def test_t09_chunk_index_sequential():
    """T-09: chunk_index가 0부터 연속 부여"""
    result = chunk(HEADING_DOC, chunk_size=300)
    indices = [c["chunk_index"] for c in result]
    assert indices == list(range(len(result))), f"순번 불일치: {indices}"
    print(f"  ✅ T-09 통과 (총 {len(result)}개, index 0~{len(result)-1})")


def test_t10_chunk_type_field_exists():
    """T-10: chunk_type 필드가 모든 청크에 존재하고 유효한 값"""
    result = chunk(HEADING_DOC, chunk_size=300)
    for c in result:
        assert "chunk_type" in c, f"chunk_type 필드 없음: {c}"
        assert c["chunk_type"] in ("text", "table"), \
            f"유효하지 않은 chunk_type: {c['chunk_type']}"
    print(f"  ✅ T-10 통과 (전체 {len(result)}개 청크 chunk_type 검증)")


# ── #75 신규 테스트 (T-11 ~ T-14) ─────────────────────────────────────────────

def test_t11_doc_year_extraction():
    """T-11: doc_year — 본문 내 연도 추출 정확도"""
    # 단일 연도
    assert _extract_doc_year("", "2023년 영업이익은 1,200억 원입니다.") == "2023"
    # 최빈 연도 (2023이 2번, 2024가 1번)
    assert _extract_doc_year("", "2023년 실적과 2023년 비교 및 2024년 전망") == "2023"
    # 동율 시 최신 연도
    assert _extract_doc_year("", "2022년과 2023년 비교") == "2023"
    # 섹션 제목에 연도 있는 경우
    assert _extract_doc_year("## 2024년 사업계획", "전략 방향을 설명합니다.") == "2024"
    # 연도 없음
    assert _extract_doc_year("", "영업이익이 증가했습니다.") is None
    print("  ✅ T-11 통과")


def test_t12_section_type_classification():
    """T-12: section_type — 섹션 유형 분류 정확도"""
    assert _extract_section_type("## 실적 요약", "") == "실적"
    assert _extract_section_type("", "매출액 9,400억 원 기록") == "실적"
    assert _extract_section_type("## 리스크 요인", "") == "리스크"
    assert _extract_section_type("", "불확실한 시장 환경으로 인한 위험 요소") == "리스크"
    assert _extract_section_type("## 향후 전망", "") == "전망"
    assert _extract_section_type("", "성장 전략 및 로드맵 방향") == "전망"
    # 매칭 없음 → None
    assert _extract_section_type("## 회사 개요", "설립 연혁 및 주요 연혁") is None
    print("  ✅ T-12 통과")


def test_t13_metrics_extraction():
    """T-13: metrics — 금융 지표 키워드 추출 + 중복 없음"""
    text = "영업이익 1,200억, 매출 9,400억, ROE 12.3%, 영업이익 증가세 지속"
    result = _extract_metrics(text)
    assert "영업이익" in result
    assert "매출" in result
    assert "ROE" in result
    # 중복 없음 확인
    assert len(result) == len(set(result)), f"중복 지표 존재: {result}"
    # 없는 지표는 포함 안 됨
    assert "EPS" not in result
    print(f"  ✅ T-13 통과 (추출 지표: {result})")


def test_t14_metadata_fields_all_present():
    """T-14: 모든 청크에 #75 메타데이터 필드 존재 + 타입 유효성"""
    result = chunk(HEADING_DOC, chunk_size=300)
    assert len(result) > 0, "청크가 없어 검증 불가"
    for c in result:
        # 필드 존재 확인
        assert "doc_year" in c, f"doc_year 필드 없음: {c['section']}"
        assert "section_type" in c, f"section_type 필드 없음: {c['section']}"
        assert "metrics" in c, f"metrics 필드 없음: {c['section']}"
        # 타입 유효성
        assert c["doc_year"] is None or isinstance(c["doc_year"], str), \
            f"doc_year 타입 오류: {type(c['doc_year'])}"
        assert c["section_type"] is None or c["section_type"] in ("실적", "리스크", "전망"), \
            f"section_type 유효하지 않은 값: {c['section_type']}"
        assert isinstance(c["metrics"], list), \
            f"metrics 타입 오류: {type(c['metrics'])}"
    print(f"  ✅ T-14 통과 (전체 {len(result)}개 청크 메타데이터 필드 검증)")


# ── 실행 ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_t01_empty_input,
        test_t02_no_heading_fallback,
        test_t03_table_chunk_type,
        test_t04_duplicate_table_dedup,
        test_t05_skip_compliance,
        test_t06_short_section_single_chunk,
        test_t07_long_section_fallback,
        test_t08_short_text_semantic_skip,
        test_t09_chunk_index_sequential,
        test_t10_chunk_type_field_exists,
        test_t11_doc_year_extraction,
        test_t12_section_type_classification,
        test_t13_metrics_extraction,
        test_t14_metadata_fields_all_present,
    ]

    print("\n" + "=" * 60)
    print("🧩 chunker.py 단위 테스트 — #59 SemanticChunking + #75 메타데이터 enrichment")
    print("=" * 60)

    passed = 0
    for fn in tests:
        print(f"\n▶ {fn.__name__}")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ 실패: {e}")
        except Exception as e:
            print(f"  ❌ 에러: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print(f"결과: {passed}/{len(tests)} 통과")
    print("=" * 60)
    if passed == len(tests):
        print("\n✅ 전체 통과 — PR 머지 가능")
    else:
        print("\n❌ 실패 항목 확인 후 머지하세요")
