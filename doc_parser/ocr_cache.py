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


def _page_cache_path(pdf_hash: str, page_index: int) -> Path:
    """페이지 단위 캐시 경로 — {hash}_p{index}.json"""
    return CACHE_DIR / f"{pdf_hash}_p{page_index}.json"


def extract_text_with_cache(pdf_path: Path) -> list[str]:
    """
    PDF 전체 페이지 OCR 결과를 캐시에서 반환.
    이미지 기반 PDF(전체 스캔본) 전용.
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

        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        lines = reader.readtext(img_bytes, detail=0)
        pages_text.append("\n".join(lines))

    print(f"\n[OCR] 완료 — 캐시 저장: {cache}")
    cache.write_text(json.dumps(pages_text, ensure_ascii=False), encoding="utf-8")
    return pages_text


def extract_page_with_cache(pdf_path: Path, page_index: int) -> str:
    """
    특정 페이지만 EasyOCR로 추출. 페이지 단위 캐시 사용.
    텍스트 기반 PDF에서 이미지 임베딩 페이지만 선택적으로 OCR할 때 사용.

    Args:
        pdf_path:   PDF 파일 경로
        page_index: 0-based 페이지 인덱스

    Returns:
        해당 페이지의 OCR 텍스트 (줄바꿈 구분)
    """
    pdf_path = Path(pdf_path)
    h = _pdf_hash(pdf_path)
    cache = _page_cache_path(h, page_index)

    if cache.exists():
        print(f"[OCR] 페이지 캐시 히트 ({h}_p{page_index})")
        return json.loads(cache.read_text(encoding="utf-8"))

    print(f"[OCR] p{page_index + 1} OCR 시작...")
    reader = _get_reader()
    doc = fitz.open(str(pdf_path))

    if page_index >= len(doc):
        return ""

    pix = doc[page_index].get_pixmap(dpi=300)
    img_bytes = pix.tobytes("png")
    lines = reader.readtext(img_bytes, detail=0)
    text = "\n".join(lines)

    cache.write_text(json.dumps(text, ensure_ascii=False), encoding="utf-8")
    print(f"[OCR] p{page_index + 1} 완료 — {len(text)}자")
    return text
