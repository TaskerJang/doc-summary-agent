"""
HWP 파서 비교 테스트
대상 라이브러리: libhwp / python-hwpx

.hwp 와 .hwpx 는 포맷 구조가 달라 별도 처리 (REQ-03)
⚠️  LLM·VLM 사용 금지
"""
import time
import traceback
from pathlib import Path
from config import DOCS, output_path


# ── 1. libhwp (.hwp) ─────────────────────────────────────────
def parse_libhwp(hwp_path: Path) -> str:
    """
    libhwp: .hwp 전용 (텍스트·표 추출)
    ⚠️  라이브러리 실제 동작 검증 필요 (기능정의서 REQ-03 비고)
    ⚠️  pyo3 Rust 패닉은 BaseException으로만 잡힘 — Exception으로는 잡히지 않음
    """
    try:
        from libhwp import HWPReader
        reader = HWPReader(str(hwp_path))
        lines = []
        for paragraph in reader.get_paragraphs():
            text = str(paragraph).strip()
            if text:
                lines.append(text)
        return "\n".join(lines)
    except ImportError:
        return "[libhwp] 라이브러리 미설치 — uv add libhwp 실행 필요"
    except BaseException as e:
        # Rust(pyo3) 레벨 패닉은 PanicException으로 발생 → BaseException으로만 포착 가능
        return f"[libhwp] 파싱 실패 (Rust 패닉): {e}"


# ── 2. python-hwpx (.hwpx, ZIP+XML 기반) ─────────────────────
def parse_python_hwpx(hwp_path: Path) -> str:
    """
    python-hwpx: .hwpx 전용 (순수 Python, ZIP+XML)
    .hwp 입력 시 변환 불가 → 결과에 명시
    """
    if hwp_path.suffix.lower() == ".hwp":
        return "[python-hwpx] .hwp 미지원 — .hwpx 파일 필요"
    try:
        import hwpx
        doc = hwpx.load(str(hwp_path))
        return doc.get_text()
    except ImportError:
        return "[python-hwpx] 라이브러리 미설치 — uv add python-hwpx 실행 필요"


# ── 실행 ─────────────────────────────────────────────────────
def run():
    parsers = {
        "libhwp":       parse_libhwp,
        "python-hwpx":  parse_python_hwpx,
    }

    hwp_targets = {
        "hwp_nonghyup": DOCS["hwp_nonghyup"],
    }

    for doc_key, doc_path in hwp_targets.items():
        if not doc_path.exists():
            print(f"[SKIP] 문서 없음: {doc_path.name}")
            continue

        print(f"\n{'='*60}")
        print(f"📄 {doc_path.name}")
        print(f"{'='*60}")

        for lib_name, parse_fn in parsers.items():
            print(f"  ▶ {lib_name} ... ", end="", flush=True)
            try:
                t0 = time.perf_counter()
                result = parse_fn(doc_path)
                elapsed = time.perf_counter() - t0

                out = output_path(lib_name, doc_key)
                out.write_text(result, encoding="utf-8")
                print(f"✅ {elapsed:.2f}s | {len(result):,}자 → {out.name}")

            except BaseException as e:
                print(f"❌ 실패: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    run()
