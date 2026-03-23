"""
메타데이터 추출 — REQ-04
pydantic BaseModel 기반 스키마 정의 및 타입 검증
"""
from pathlib import Path
from pydantic import BaseModel
from typing import Optional


class DocumentMetadata(BaseModel):
    filename: str
    page_count: Optional[int] = None
    language: Optional[str] = None
    created_at: Optional[str] = None


def extract(file_path: Path) -> DocumentMetadata:
    """문서에서 메타데이터를 추출하여 반환한다."""
    raise NotImplementedError
