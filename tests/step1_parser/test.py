# tests/step1_parser/test.py
from pathlib import Path
import sys
import importlib

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# 캐시 무시하고 강제 리로드
import parser
import parser.pdf
importlib.reload(parser.pdf)
importlib.reload(parser)

from parser.preprocessor import clean
from parser.metadata import extract
from chunker.chunker import chunk

PDF_PATH = ROOT / "tests/step1_parser/sample_docs/미래에셋증권 1분기 실적보고서.pdf"

print("=" * 50)
print("[ 이미지 기반 판단 ]")
from parser.pdf import _is_image_based_pdf
print("이미지 기반?", _is_image_based_pdf(PDF_PATH))

print("\n" + "=" * 50)
print("[ 1. 메타데이터 ]")
meta = extract(PDF_PATH)
print(meta.model_dump())

print("\n" + "=" * 50)
print("[ 2. 파싱 ]")
text = parser.parse(PDF_PATH)
print(f"추출 텍스트: {len(text)}자")
print(text[:300])

print("\n" + "=" * 50)
print("[ 3. 전처리 ]")
clean_text = clean(text, source="ir_report")
print(f"전처리 후: {len(clean_text)}자 (감소: {len(text) - len(clean_text)}자)")
print(clean_text[:300])

print("\n" + "=" * 50)
print("[ 4. 청킹 ]")
chunks = chunk(clean_text)
print(f"총 청크 수: {len(chunks)}")
for c in chunks[:5]:
    print(f"  [{c['chunk_index']}] {repr(c['section'][:20])} | {len(c['text'])}자")
    print(f"       {c['text'][:80]}")