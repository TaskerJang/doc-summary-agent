import hashlib
import json
from pathlib import Path

import fitz

CACHE_DIR = Path(".ocr_cache")
CACHE_DIR.mkdir(exist_ok=True)

# PaddleOCR 인스턴스 — 최초 1회만 초기화
_ocr = None


def _get_ocr():
    """
    PaddleOCR v3 (PP-OCRv5) 인스턴스 반환
    - lang='korean': 한국어 + 영어 혼용 문서 지원
    - use_angle_cls=True: 회전된 텍스트 보정
    최초 호출 시 모델 다운로드 후 캐시
    """
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(
            lang='korean',
            use_angle_cls=True,
        )
    return _ocr


def _pdf_hash(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]


def _cache_path(pdf_hash: str) -> Path:
    return CACHE_DIR / f"{pdf_hash}.json"


def extract_text_with_cache(pdf_path: Path) -> list[str]:
    """
    PDF 전체 페이지 OCR 결과를 캐시에서 반환.
    최초 실행 시에만 PaddleOCR 수행 후 저장.
    반환값: 페이지별 텍스트 리스트 (index = page_index)

    변경 이력:
    - EasyOCR (dpi=150) → doctr (dpi=300) → PaddleOCR v3 PP-OCRv5 (dpi=300)
    - 정확도 최우선: 한국어 금융 문서 수치 오인식 최소화
    """
    pdf_path = Path(pdf_path)
    h = _pdf_hash(pdf_path)
    cache = _cache_path(h)

    if cache.exists():
        print(f"[OCR] 캐시 히트 ({h})")
        return json.loads(cache.read_text(encoding="utf-8"))

    print(f"[OCR] 캐시 없음 — PaddleOCR v3 시작 ({pdf_path.name})")
    ocr = _get_ocr()
    doc = fitz.open(str(pdf_path))
    pages_text = []

    for i, page in enumerate(doc):
        print(f"  p{i + 1}/{len(doc)} 처리 중...", end="\r")

        # dpi=300 고해상도 렌더링
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")

        import numpy as np
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(img)

        # PaddleOCR 실행 — result: [[[box, (text, confidence)], ...], ...]
        result = ocr.ocr(img_array, cls=True)

        # 텍스트 추출 (confidence 0.5 이상만)
        lines = []
        if result and result[0]:
            for line in result[0]:
                text, confidence = line[1]
                if confidence >= 0.5:
                    lines.append(text)

        pages_text.append("\n".join(lines))

    print(f"\n[OCR] 완료 — 캐시 저장: {cache}")
    cache.write_text(json.dumps(pages_text, ensure_ascii=False), encoding="utf-8")
    return pages_text
