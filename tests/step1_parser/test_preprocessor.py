"""
preprocessor.py 노이즈 제거 테스트 — REQ-05

실행 방법:
    uv run python tests/step1_parser/test_preprocessor.py

검증 방식:
    파서 output 파일(output/*.md)을 읽어 clean() 적용 후
    노이즈가 제거됐는지 assert로 확인하고
    before/after 결과를 output/*__clean.md 로 저장

⚠️  output/ 폴더는 .gitignore 처리되어 있음. 로컬에서만 실행 가능.
"""
import re
import sys
from pathlib import Path

# doc_parser/ 패키지 import를 위해 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from doc_parser.preprocessor import clean

OUTPUT_DIR = Path(__file__).parent / "output"


def _read(filename: str) -> str | None:
    path = OUTPUT_DIR / filename
    if not path.exists():
        print(f"  [SKIP] 파일 없음: {filename}")
        return None
    return path.read_text(encoding="utf-8")


def _save(filename: str, text: str) -> None:
    path = OUTPUT_DIR / filename
    path.write_text(text, encoding="utf-8")


def _report(label: str, raw: str, cleaned: str) -> None:
    before = len(raw)
    after = len(cleaned)
    ratio = (1 - after / before) * 100 if before else 0
    print(f"  {label}: {before:,}자 → {after:,}자 ({ratio:.1f}% 감소)")


def test_pdf_ds():
    raw = _read("pdf_ds__pymupdf4llm.md")
    if raw is None:
        return
    cleaned = clean(raw, source="pdf")
    _save("pdf_ds__pymupdf4llm__clean.md", cleaned)
    _report("pdf_ds", raw, cleaned)
    assert "==> picture" not in cleaned, "N-01: picture omit 태그 잔존"
    assert "Start of picture text" not in cleaned, "N-02: picture text 블록 잔존"
    empty_bullets = re.findall(r"(?m)^[\s▪•\-–—]+\s*$", cleaned)
    assert not empty_bullets, f"N-11: 빈 bullet {len(empty_bullets)}개 잔존"
    assert "\n\n\n" not in cleaned, "N-12: 과잉 공백 잔존"
    print("  ✅ pdf_ds 통과")


def test_pdf_hanwha():
    raw = _read("pdf_hanwha__pymupdf4llm.md")
    if raw is None:
        return
    cleaned = clean(raw, source="pdf")
    _save("pdf_hanwha__pymupdf4llm__clean.md", cleaned)
    _report("pdf_hanwha", raw, cleaned)
    assert "==> picture" not in cleaned, "N-01: picture omit 태그 잔존"
    page_numbers = re.findall(r"(?m)^\s*\d{1,3}\s*$", cleaned)
    assert not page_numbers, f"N-06: 페이지 번호 라인 {len(page_numbers)}개 잔존"
    assert cleaned.count("두산밥캣 (241560)") <= 1, "N-07: 반복 헤더 '두산밥캣 (241560)' 잔존"
    assert cleaned.count("[한화리서치]") <= 1, "N-07: 반복 헤더 '[한화리서치]' 잔존"
    assert "\n\n\n" not in cleaned, "N-12: 과잉 공백 잔존"
    print("  ✅ pdf_hanwha 통과")


def test_pdf_miraeasset_4q():
    raw = _read("pdf_miraeasset_4q__pymupdf4llm.md")
    if raw is None:
        return
    cleaned = clean(raw, source="ir_report")
    _save("pdf_miraeasset_4q__pymupdf4llm__clean.md", cleaned)
    _report("pdf_miraeasset_4q", raw, cleaned)
    assert "==> picture" not in cleaned, "N-01: picture omit 태그 잔존"
    assert "Start of picture text" not in cleaned, "N-02: picture text 블록 잔존"
    assert "<br>" not in cleaned, "N-03: <br> 태그 잔존"
    assert "<br/>" not in cleaned, "N-03: <br/> 태그 잔존"
    assert cleaned.count("요약손익계산서") < 3, f"N-03: 요약손익계산서 {cleaned.count('요약손익계산서')}회 중복 잔존"
    assert cleaned.count("2025년실적보고서") < 3, f"N-04: 페이지 타이틀 {cleaned.count('2025년실적보고서')}회 반복 잔존"
    assert "\n\n\n" not in cleaned, "N-12: 과잉 공백 잔존"
    print("  ✅ pdf_miraeasset_4q 통과")


def test_hwp_nonghyup():
    raw = _read("hwp_nonghyup__pyhwp.md")
    if raw is None:
        return
    cleaned = clean(raw, source="hwp")
    _save("hwp_nonghyup__pyhwp__clean.md", cleaned)
    _report("hwp_nonghyup", raw, cleaned)
    assert ".Section-0" not in cleaned, "N-05: CSS 블록 '.Section-0' 잔존"
    assert ".HeaderPageFooter" not in cleaned, "N-05: CSS 블록 '.HeaderPageFooter' 잔존"
    assert "<style" not in cleaned, "N-05: <style> 태그 잔존"
    assert "\n\n\n" not in cleaned, "N-12: 과잉 공백 잔존"
    assert "사업보고서" in cleaned, "본문 손실: '사업보고서' 텍스트 없음"
    assert "재무상태표" in cleaned, "본문 손실: '재무상태표' 텍스트 없음"
    print("  ✅ hwp_nonghyup 통과")


def test_doc_fss():
    raw = _read("doc_fss__python-docx.md")
    if raw is None:
        return
    cleaned = clean(raw, source="docx")
    _save("doc_fss__python-docx__clean.md", cleaned)
    _report("doc_fss", raw, cleaned)
    assert "본 자료를 인용하여 보도할 경우" not in cleaned, "N-08: 보도출처 안내문 잔존"
    assert "\n\n\n" not in cleaned, "N-12: 과잉 공백 잔존"
    assert "회사채" in cleaned, "본문 손실: '회사채' 텍스트 없음"
    print("  ✅ doc_fss 통과")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("📋 preprocessor.py 노이즈 제거 테스트")
    print("=" * 60)

    tests = [
        test_pdf_ds,
        test_pdf_hanwha,
        test_pdf_miraeasset_4q,
        test_hwp_nonghyup,
        test_doc_fss,
    ]

    passed = 0
    for test_fn in tests:
        print(f"\n▶ {test_fn.__name__}")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ 실패: {e}")
        except Exception as e:
            print(f"  ❌ 에러: {e}")

    print("\n" + "=" * 60)
    print(f"결과: {passed}/{len(tests)} 통과")
    print("=" * 60)
    print("\n💡 clean 결과는 output/*__clean.md 에서 확인하세요.")
