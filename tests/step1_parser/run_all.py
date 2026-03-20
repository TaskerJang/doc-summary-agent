"""
전체 파서 비교 테스트 실행
python tests/step1_parser/run_all.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import test_pdf_parsers
import test_docx_parsers
import test_hwp_parsers

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# PDF 파서 비교")
    print("#" * 60)
    test_pdf_parsers.run()

    print("\n" + "#" * 60)
    print("# DOCX 파서 비교")
    print("#" * 60)
    test_docx_parsers.run()

    print("\n" + "#" * 60)
    print("# HWP 파서 비교")
    print("#" * 60)
    test_hwp_parsers.run()

    print("\n" + "=" * 60)
    print("✅ 전체 완료 — output/ 폴더에서 결과 육안 검토 후 평가표에 기입")
    print("=" * 60)
