# test_doctr.py
import time
import fitz

PDF_PATH = r"sample_docs\미래에셋증권 1분기 실적보고서.pdf"
PAGE_IDX = 4  # p5 재무실적 요약

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

start = time.time()
model = ocr_predictor(
    det_arch='db_mobilenet_v3_large',
    reco_arch='crnn_mobilenet_v3_small',
    pretrained=True
)
doc = DocumentFile.from_pdf(PDF_PATH)
result = model(doc)
elapsed = time.time() - start

page_result = result.pages[PAGE_IDX]
lines = [
    " ".join(w.value for w in line.words)
    for block in page_result.blocks
    for line in block.lines
]

print(f"소요시간: {elapsed:.1f}초")
print(f"추출 라인 수: {len(lines)}")
print("\n".join(lines[:20]))