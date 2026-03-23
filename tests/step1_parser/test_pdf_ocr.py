"""
이미지 기반 PDF OCR 테스트
전략: pymupdf로 페이지를 PNG 렌더링 → pytesseract(한국어) OCR

⚠️  LLM·VLM 사용 금지 — 순수 OCR만 사용 (REQ-01)

사전 설치 필요:
  uv add pytesseract pymupdf Pillow
  # tesseract 바이너리 + 한국어 언어팩
  # Windows: https://github.com/UB-Mannheim/tesseract/wiki
  #   → 설치 시 'Korean' 체크박스 선택
  # macOS:   brew install tesseract tesseract-lang
  # Ubuntu:  apt install tesseract-ocr tesseract-ocr-kor
"""
import time
import traceback
from pathlib import Path
from config import DOCS, output_path

# 이미지 기반 PDF 키 목록 (미래에셋 1~3Q 가 이미지 기반으로 확인됨)
IMAGE_PDF_KEYS = ["pdf_miraeasset_1q", "pdf_miraeasset_2q", "pdf_miraeasset_3q"]
IMAGE_PDF_DOCS = {k: v for k, v in DOCS.items() if k in IMAGE_PDF_KEYS}

# OCR 렌더링 DPI (높을수록 정확하나 느림. 200~300 권장)
DPI = 200


def is_image_based(pdf_path: Path, sample_pages: int = 3) -> bool:
    """앞 페이지 몇 장을 보고 이미지 기반 PDF 여부를 판별한다."""
    import fitz  # pymupdf
    doc = fitz.open(str(pdf_path))
    for i in range(min(sample_pages, len(doc))):
        if doc[i].get_text().strip():
            return False
    return True


def ocr_page(page, dpi: int = DPI) -> str:
    """pymupdf 페이지를 PNG로 렌더링한 뒤 tesseract로 OCR한다."""
    import pytesseract
    from PIL import Image
    import io

    mat = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(mat.tobytes("png")))
    return pytesseract.image_to_string(img, lang="kor+eng")


def parse_ocr(pdf_path: Path) -> str:
    """PDF 전체 페이지를 OCR하여 텍스트를 반환한다."""
    import fitz

    doc = fitz.open(str(pdf_path))
    pages_text = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            # 텍스트 레이어가 있는 페이지는 그대로 사용 (혼합 PDF 대응)
            pages_text.append(f"<!-- page {i+1}: text layer -->\n{text}")
        else:
            # 이미지 페이지 → OCR
            ocr_text = ocr_page(page)
            pages_text.append(f"<!-- page {i+1}: OCR -->\n{ocr_text}")
    return "\n\n".join(pages_text)


def check_dependencies() -> bool:
    """필수 라이브러리 및 tesseract 바이너리 존재 여부를 확인한다."""
    try:
        import fitz
        import pytesseract
        from PIL import Image
        pytesseract.get_tesseract_version()
        return True
    except Exception as e:
        print(f"[의존성 오류] {e}")
        print("  → tesseract 설치 및 PATH 등록 여부를 확인하세요.")
        return False


def run():
    print("=" * 60)
    print("이미지 기반 PDF OCR 테스트 (tesseract kor+eng)")
    print(f"렌더링 DPI: {DPI}")
    print("=" * 60)

    if not check_dependencies():
        return

    for doc_key, doc_path in IMAGE_PDF_DOCS.items():
        if not doc_path.exists():
            print(f"\n[SKIP] 문서 없음: {doc_path.name}")
            continue

        print(f"\n📄 {doc_path.name}")

        # 이미지 기반 여부 사전 판별
        detected = is_image_based(doc_path)
        print(f"  이미지 기반 감지: {'✅ YES' if detected else '❌ NO (텍스트 레이어 존재)'}")

        print(f"  ▶ OCR 진행 중 ... ", end="", flush=True)
        try:
            t0 = time.perf_counter()
            result = parse_ocr(doc_path)
            elapsed = time.perf_counter() - t0

            out = output_path("ocr_tesseract", doc_key, ext="txt")
            out.write_text(result, encoding="utf-8")

            # 품질 간이 지표: 추출된 한글 문자 수
            kor_chars = sum(1 for c in result if '가' <= c <= '힣')
            print(f"✅ {elapsed:.1f}s | 전체 {len(result):,}자 | 한글 {kor_chars:,}자 → {out.name}")

            # 앞 500자 미리보기
            preview = result[:500].replace("\n", " ")
            print(f"  미리보기: {preview}...")

        except Exception as e:
            print(f"❌ 실패: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    run()
