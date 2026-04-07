import hashlib
import json
from pathlib import Path

import fitz

CACHE_DIR = Path(".ocr_cache")
CACHE_DIR.mkdir(exist_ok=True)

# doctr 모델은 최초 1회만 로드
_det_model = None
_reco_model = None


def _get_predictor():
    """
    doctr OCR predictor — 한국어 포함 다국어 지원
    최초 호출 시 모델 다운로드 후 캐시
    """
    global _det_model, _reco_model
    from doctr.models import ocr_predictor
    if _det_model is None:
        _det_model = ocr_predictor(
            det_arch='db_resnet50',
            reco_arch='crnn_vgg16_bn',
            pretrained=True,
        )
    return _det_model


def _pdf_hash(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]


def _cache_path(pdf_hash: str) -> Path:
    return CACHE_DIR / f"{pdf_hash}.json"


def extract_text_with_cache(pdf_path: Path) -> list[str]:
    """
    PDF 전체 페이지 OCR 결과를 캐시에서 반환.
    최초 실행 시에만 doctr OCR 수행 후 저장.
    반환값: 페이지별 텍스트 리스트 (index = page_index)

    변경 이력:
    - EasyOCR (dpi=150) → doctr db_resnet50 + crnn_vgg16_bn (dpi=300)
    - 정확도 개선 목적 (한국어 금융 수치 오인식 감소)
    """
    pdf_path = Path(pdf_path)
    h = _pdf_hash(pdf_path)
    cache = _cache_path(h)

    if cache.exists():
        print(f"[OCR] 캐시 히트 ({h})")
        return json.loads(cache.read_text(encoding="utf-8"))

    print(f"[OCR] 캐시 없음 — doctr OCR 시작 ({pdf_path.name})")
    predictor = _get_predictor()
    doc = fitz.open(str(pdf_path))
    pages_text = []

    for i, page in enumerate(doc):
        print(f"  p{i + 1}/{len(doc)} 처리 중...", end="\r")

        # dpi=300으로 고해상도 렌더링 (기존 150 → 300)
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")

        # doctr는 numpy array 또는 파일 경로 입력
        import numpy as np
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(img)

        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor

        # 페이지 단위 OCR
        doc_input = DocumentFile.from_images([img_array])
        result = predictor(doc_input)

        # 결과 텍스트 추출 (블록 → 라인 → 단어 순서로 flatten)
        page_lines = []
        for block in result.pages[0].blocks:
            for line in block.lines:
                words = [word.value for word in line.words]
                page_lines.append(" ".join(words))

        pages_text.append("\n".join(page_lines))

    print(f"\n[OCR] 완료 — 캐시 저장: {cache}")
    cache.write_text(json.dumps(pages_text, ensure_ascii=False), encoding="utf-8")
    return pages_text
