import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from doc_parser.exceptions import EmptyDocumentError, FileCorruptedError, FileNotSupportedError

SUPPORTED_EXTENSIONS = {".hwp", ".hwpx"}


def parse(hwp_path: Path) -> str:
    """
    진입점 — 외부에서 유일하게 호출하는 함수
    .hwp  → pyhwp (hwp5html) 처리
    .hwpx → NotImplementedError (추후 구현)
    """
    ext = hwp_path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise FileNotSupportedError(
            f"HWP 파서는 {SUPPORTED_EXTENSIONS} 포맷만 지원합니다: '{ext}'"
        )

    if hwp_path.stat().st_size == 0:
        raise FileCorruptedError(f"빈 파일입니다: {hwp_path.name}")

    if ext == ".hwpx":
        raise NotImplementedError("hwpx 파서 미구현 — python-hwpx 연결 예정")

    return _parse_pyhwp(hwp_path)


def parse_bytes(data: bytes, filename: str = "upload.hwp") -> str:
    """parse()의 바이트 버전 — FastAPI UploadFile 대응"""
    if not data:
        raise FileCorruptedError("빈 바이트 데이터입니다.")

    suffix = Path(filename).suffix or ".hwp"
    tmp_path = Path(tempfile.mktemp(suffix=suffix))
    try:
        tmp_path.write_bytes(data)
        return parse(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _parse_pyhwp(hwp_path: Path) -> str:
    """hwp5html로 HTML 추출 후 표 포함 텍스트 파싱"""
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        try:
            result = subprocess.run(
                ["hwp5html", "--output", str(tmp_dir), str(hwp_path)],
                capture_output=True,
            )
        except FileNotFoundError:
            raise FileCorruptedError(
                "pyhwp가 설치되어 있지 않습니다. uv add pyhwp six 실행 필요."
            )

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise FileCorruptedError(f"hwp5html 실패: {stderr}")

        html_files = sorted(tmp_dir.rglob("*.xhtml")) or sorted(tmp_dir.rglob("*.html"))
        if not html_files:
            raise FileCorruptedError(f"hwp5html 출력 파일 없음: {hwp_path.name}")

        sections = []
        for html_file in html_files:
            raw = html_file.read_text(encoding="utf-8", errors="replace")
            text = re.sub(r"<[^>]+>", " ", raw)
            text = html.unescape(text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r" {2,}", " ", text)
            sections.append(text.strip())

        text = "\n\n".join(sections)

        if not text.strip():
            raise EmptyDocumentError(f"텍스트를 추출할 수 없습니다: {hwp_path.name}")

        return text

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
