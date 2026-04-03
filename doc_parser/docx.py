import subprocess
import sys
import tempfile
from pathlib import Path

from doc_parser.exceptions import EmptyDocumentError, FileCorruptedError, FileNotSupportedError

if sys.platform == "win32":
    LIBREOFFICE_BIN = r"C:\Program Files\LibreOffice\program\soffice.exe"
else:
    LIBREOFFICE_BIN = "libreoffice"

_HEADING_PREFIX = {
    "Heading 1": "# ",
    "Heading 2": "## ",
    "Heading 3": "### ",
}

SUPPORTED_EXTENSIONS = {".docx", ".doc"}


def parse(path: Path) -> str:
    """
    진입점 — 외부에서 유일하게 호출하는 함수
    .doc 입력 시 LibreOffice로 .docx 변환 후 처리
    python-docx 시도 → 결과 없으면 docx2python fallback
    """
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise FileNotSupportedError(
            f"DOCX 파서는 {SUPPORTED_EXTENSIONS} 포맷만 지원합니다: '{ext}'"
        )

    if path.stat().st_size == 0:
        raise FileCorruptedError(f"빈 파일입니다: {path.name}")

    if ext == ".doc":
        path = _convert_doc_to_docx(path)

    text = _parse_python_docx(path)
    if not text.strip():
        text = _parse_docx2python(path)

    if not text.strip():
        raise EmptyDocumentError(f"텍스트를 추출할 수 없습니다: {path.name}")

    return text


def parse_bytes(data: bytes, filename: str = "upload.docx") -> str:
    """parse()의 바이트 버전 — FastAPI UploadFile 대응"""
    if not data:
        raise FileCorruptedError("빈 바이트 데이터입니다.")

    suffix = Path(filename).suffix or ".docx"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        return parse(Path(tmp.name))


def _convert_doc_to_docx(doc_path: Path) -> Path:
    """LibreOffice CLI로 .doc → .docx 변환"""
    out_dir = doc_path.parent
    try:
        result = subprocess.run(
            [LIBREOFFICE_BIN, "--headless", "--convert-to", "docx",
             "--outdir", str(out_dir), str(doc_path)],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        raise FileCorruptedError(
            "LibreOffice가 설치되어 있지 않습니다. .doc 변환 불가."
        )

    if result.returncode != 0:
        raise FileCorruptedError(f"LibreOffice 변환 실패: {result.stderr}")

    converted = doc_path.with_suffix(".docx")
    if not converted.exists():
        raise FileCorruptedError(f"변환 결과 파일 없음: {converted}")
    return converted


def _parse_python_docx(docx_path: Path) -> str:
    """python-docx로 단락·헤딩·표 추출"""
    try:
        from docx import Document
        doc = Document(docx_path)
    except Exception as e:
        raise FileCorruptedError(f"python-docx 파일 열기 실패: {e}") from e

    lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        prefix = _HEADING_PREFIX.get(para.style.name, "")
        lines.append(f"{prefix}{text}")

    for table in doc.tables:
        for row in table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))

    return "\n".join(lines)


def _parse_docx2python(docx_path: Path) -> str:
    """docx2python fallback — 중첩 리스트 평탄화"""
    try:
        from docx2python import docx2python
        result = docx2python(docx_path)
    except Exception as e:
        raise FileCorruptedError(f"docx2python 파일 열기 실패: {e}") from e

    def _flatten(obj) -> str:
        if isinstance(obj, str):
            return obj
        if isinstance(obj, list):
            return "\n".join(_flatten(item) for item in obj if item)
        return str(obj)

    lines = []
    for section in result.body:
        text = _flatten(section).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)
