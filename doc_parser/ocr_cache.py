import hashlib
import json
from pathlib import Path

import easyocr
import fitz

CACHE_DIR = Path(".ocr_cache")
CACHE_DIR.mkdir(exist_ok=True)

_reader = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['ko', 'en'], gpu=False)
    return _reader


def _pdf_hash(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]


def _cache_path(pdf_hash: str) -> Path:
    return CACHE_DIR / f"{pdf_hash}.json"


def extract_text_with_cache(pdf_path: Path) -> list[str]:
    """
    PDF 전체 페이지 OCR 결과를 캐시에서 반환.
    최초 실행 시에만 EasyOCR 수행 후 저장.
    반환값: 페이지별 텍스트 리스트 (index = page_index)

    변경 이력:
    - EasyOCR dpi=150 → dpi=300 상향 (정확도 개선)
    - PaddleOCR v3 시도했으나 Windows CPU 환경 미지원으로 EasyOCR 유지
    """
    pdf_path = Path(pdf_path)
    h = _pdf_hash(pdf_path)
    cache = _cache_path(h)

    if cache.exists():
        print(f"[OCR] 캐시 히트 ({h})")
        return json.loads(cache.read_text(encoding="utf-8"))

    print(f"[OCR] 캐시 없음 — EasyOCR 시작 ({pdf_path.name})")
    reader = _get_reader()
    doc = fitz.open(str(pdf_path))
    pages_text = []

    for i, page in enumerate(doc):
        print(f"  p{i + 1}/{len(doc)} 처리 중...", end="\r")

        # dpi=300 고해상도 렌더링 (기존 150 → 300)
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        lines = reader.readtext(img_bytes, detail=0)
        pages_text.append("\n".join(lines))

    print(f"\n[OCR] 완료 — 캐시 저장: {cache}")
    cache.write_text(json.dumps(pages_text, ensure_ascii=False), encoding="utf-8")
    return pages_text
