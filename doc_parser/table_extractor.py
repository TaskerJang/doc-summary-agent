"""
gmft 기반 구조화 표 추출기

이미지 임베딩 PDF 페이지에서 표 구조를 감지하고 Markdown으로 변환.
pymupdf4llm이 'intentionally omitted'으로 생략한 이미지가 표인 경우
EasyOCR보다 구조화된 결과를 제공.

gmft 미설치 시 ImportError를 발생시켜 호출부에서 EasyOCR fallback 처리.

설치:
    uv add gmft
"""
from pathlib import Path


def _table_to_markdown(table) -> str:
    """
    gmft CroppedTable 또는 FormattedTable 객체를 Markdown 표로 변환.
    header 행은 굵게, 이후 행은 일반 텍스트.
    """
    try:
        import gmft
        from gmft.pdf_bindings import PyPDFium2Document
        from gmft.auto import AutoTableFormatter

        formatter = AutoTableFormatter()
        ft = formatter.extract(table)
        df = ft.df()

        if df is None or df.empty:
            return ""

        # DataFrame → Markdown
        lines = []
        headers = [str(c) for c in df.columns]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "---|" * len(headers))
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(v) if v is not None else "" for v in row) + " |")
        return "\n".join(lines)
    except Exception:
        return ""


def extract_tables_from_page(pdf_path: Path, page_index: int) -> str:
    """
    gmft로 특정 페이지(0-based)의 모든 표를 추출해 Markdown으로 반환.

    Args:
        pdf_path:   PDF 파일 경로
        page_index: 0-based 페이지 인덱스

    Returns:
        추출된 표들의 Markdown 문자열 (표 여러 개면 빈 줄로 구분)
        표가 없거나 실패 시 빈 문자열

    Raises:
        ImportError: gmft가 설치되지 않은 경우
    """
    import gmft
    from gmft.pdf_bindings import PyPDFium2Document
    from gmft.auto import AutoTableDetector

    detector = AutoTableDetector()

    doc = PyPDFium2Document(str(pdf_path))
    try:
        page = doc[page_index]
        tables = detector.extract(page)
    finally:
        doc.close()

    if not tables:
        return ""

    results = []
    for table in tables:
        md = _table_to_markdown(table)
        if md.strip():
            results.append(md)

    return "\n\n".join(results)


def extract_tables_from_page_safe(pdf_path: Path, page_index: int) -> str:
    """
    extract_tables_from_page의 safe wrapper.
    gmft 미설치 또는 모든 예외 시 빈 문자열 반환 (호출부 fallback 처리용).
    """
    try:
        return extract_tables_from_page(pdf_path, page_index)
    except ImportError:
        raise  # 호출부에서 ImportError 감지 → EasyOCR fallback 트리거
    except Exception:
        return ""
