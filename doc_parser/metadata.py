import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import fitz
from docx import Document
from langdetect import detect, DetectorFactory
from pydantic import BaseModel

from doc_parser.exceptions import FileNotSupportedError

logger = logging.getLogger(__name__)
DetectorFactory.seed = 0  # langdetect 결과 일관성 보장


class DocumentMetadata(BaseModel):
    filename: str
    file_format: str                       # "pdf" | "docx" | "doc" | "hwp" | "hwpx"
    file_size_bytes: int
    page_count: Optional[int] = None       # HWP·DOC는 추출 불가 → None
    language: Optional[str] = None         # langdetect 기반, 실패 시 None
    created_at: Optional[datetime] = None  # 추출 불가 시 None


def extract(file_path: Path) -> DocumentMetadata:
    """문서에서 메타데이터를 추출하여 반환한다."""
    ext = file_path.suffix.lower().lstrip(".")

    base = dict(
        filename=file_path.name,
        file_format=ext,
        file_size_bytes=file_path.stat().st_size,
    )

    if ext == "pdf":
        try:
            return DocumentMetadata(**base, **_extract_pdf(file_path))
        except Exception as e:
            logger.warning("PDF 메타데이터 추출 실패: %s", e)
            return DocumentMetadata(**base)

    if ext in ("docx", "doc"):
        try:
            return DocumentMetadata(**base, **_extract_docx(file_path))
        except Exception as e:
            logger.warning("DOCX 메타데이터 추출 실패: %s", e)
            return DocumentMetadata(**base)

    if ext in ("hwp", "hwpx"):
        return DocumentMetadata(**base)  # 페이지 수·날짜 추출 불가

    raise FileNotSupportedError(f"지원하지 않는 포맷: '.{ext}'")


def _extract_pdf(file_path: Path) -> dict:
    with fitz.open(file_path) as doc:
        raw_meta = doc.metadata or {}

        created_at = None
        raw_date = raw_meta.get("creationDate") or ""
        if raw_date.startswith("D:"):
            try:
                created_at = datetime.strptime(raw_date[2:16], "%Y%m%d%H%M%S")
            except ValueError:
                pass

        sample = " ".join(
            doc.load_page(i).get_text()
            for i in range(min(3, doc.page_count))
        )

        return dict(
            page_count=doc.page_count,
            language=_detect_language(sample),
            created_at=created_at,
        )


def _extract_docx(file_path: Path) -> dict:
    doc = Document(file_path)
    props = doc.core_properties

    # core_properties.language가 비어있으면 본문 텍스트로 langdetect fallback
    language = None
    if props.language:
        language = props.language.split("-")[0]
    else:
        sample = " ".join(p.text for p in doc.paragraphs[:20] if p.text.strip())
        language = _detect_language(sample)

    return dict(
        language=language,
        created_at=props.created,
    )


def _detect_language(text: str) -> Optional[str]:
    """langdetect로 언어 감지. 실패하면 None 반환."""
    if not text.strip():
        return None
    try:
        return detect(text)
    except Exception:
        return None
