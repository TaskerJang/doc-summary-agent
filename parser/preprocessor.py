import re
from collections import Counter


def clean(text: str, *, source: str = "generic", preserve_picture_text: bool = False) -> str:
    """노이즈를 제거한 텍스트를 반환한다.

    Args:
        text: 파서가 출력한 원본 텍스트
        source: 문서 포맷
        preserve_picture_text: True 시 picture text 블록 보존
    """
    # CSS 블록 최우선 제거 => 불필요한 추가 연산 방지를 위해
    if source == "hwp":
        text = _remove_css_blocks(text)

    # <br> 표 최우선 제거 => 불필요한 추가 연산 방지를 위해
    if source == "ir_report":
        text = _convert_br_and_dedup_tables(text)

    # picture omit 태그, picture_text 블록 제거 -> 차트 이미지 없이 수치는 그냥 제거하는게 맞다.
    if source in ("pdf", "ir_report"):
        text = _remove_picture_omit_tags(text)
        text = _remove_picture_text_blocks(text)

    # PDF의 경우, 페이지 타이틀이 반복되면 제거
    if source in ("pdf", "ir_report"):
        text = _remove_repeated_titles(text, threshold=3)

    # PDF의 경우, 페이지 번호가 존재하므로 제거
    if source in ("pdf", "ir_report"):
        text = _remove_page_numbers(text)

    # PDF의 경우, 헤더가 반복되므로 제거
    if source in ("pdf", "ir_report"):
        text = _remove_repeated_lines(text, threshold=2)

    # 표의 헤더가 중복되는 것을 방지
    if source == "docx":
        text = _dedup_consecutive_lines(text)

    # 반복되는 주석 제거
    if source == "docx":
        text = _dedup_footnotes(text)
        text = re.sub(r"(?m)^☞.*출처.*바랍니다.*$", "", text)

    # 막대 그래프 이미지라 bullet 기호만 남는 경우
    text = _remove_empty_bullets(text)

    # 공백 제거: 노이즈 제거 후 공백이 생기므로 마지막 작업
    text = _normalize_whitespace(text)

    return text

def _remove_css_blocks(text: str) -> str:
    # <style> 태그가 있는 경우
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    # 태그 없이 CSS 코드만 남은 경우 (.클래스명 { ... } 패턴)
    text = re.sub(r"\.[A-Za-z][\w\-]*[^{]*\{[^}]*\}", "", text, flags=re.DOTALL)
    return text

def _convert_br_and_dedup_tables(text: str) -> str:
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")

    parts = re.split(r"(?=\n\|)", text)
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        lines = part.strip().splitlines()
        # | 포함 라인들에서 텍스트만 추출해서 키 생성
        table_lines = [ln for ln in lines if "|" in ln]
        if table_lines:
            # 표 첫 번째 줄에서 | 와 * 와 공백 제거 후 60자를 키로
            key = re.sub(r"[|\*\s]", "", table_lines[0])[:60]
        else:
            key = part.strip()[:120]
        if key not in seen:
            seen.add(key)
            result.append(part)
    return "".join(result)

def _remove_picture_omit_tags(text: str) -> str:
    """
    pymupdf4llm이 이미지를 읽지 못할 때 삽입하는 문구를 제거
    ex) **==> picture [564 x 89] intentionally omitted <==**
    """
    return re.sub(
        r"\*\*==>?\s*picture\s*\[\d+\s*x\s*\d+\]\s*intentionally omitted\s*<==\*\*\n?",
        "",
        text,
    )

def _remove_picture_text_blocks(text: str) -> str:
    """
    차트 이미지 위 텍스트 레이어를 긁어온 블록을 제거

    ----- Start of picture text ----- 부터
    ----- End of picture text ----- 까지 통째로 삭제.
    """
    return re.sub(
        r"\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*\n?",
        "",
        text,
        flags=re.DOTALL,
    )

def _remove_repeated_titles(text: str, threshold: int = 3) -> str:
    """
    threshold 값 이상 등장하는 타이틀의 첫 1회를 유지하고 이후를 제거.
    페이지 타이틀('2025년실적보고서')은 문서 제목이므로 첫 번째 등장은 남긴다.
    """
    lines = text.splitlines()
    counts = Counter(ln.strip() for ln in lines if ln.strip())
    noise = {ln for ln, cnt in counts.items() if cnt >= threshold}
    seen: set[str] = set()
    result: list[str] = []
    for ln in lines:
        key = ln.strip()
        if key in noise:
            if key not in seen:
                seen.add(key)
                result.append(ln)
        else:
            result.append(ln)
    return "\n".join(result)

def _remove_page_numbers(text: str) -> str:
    """
    페이지 번호 제거
    {1,3}으로 1~3자리 숫자만 잡아 제거. 연도는 살린다.
    """
    return re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)

def _remove_repeated_lines(text: str, threshold: int = 2) -> str:
    lines = text.splitlines()

    def normalize(ln: str) -> str:
        return re.sub(r"^#+\s*", "", ln.strip())

    counts = Counter(normalize(ln) for ln in lines if ln.strip())
    noise = {ln for ln, cnt in counts.items() if cnt >= threshold}
    return "\n".join(
        ln for ln in lines
        if normalize(ln) not in noise
    )

def _dedup_consecutive_lines(text: str) -> str:
    """
    바로 이전 행과 완전히 동일한 행이 연속으로 나오면 건너뛴다.

    가로 병합·셀 내 줄바꿈·세로 병합 3중 오류의 근본 해결은
    docx.py 파서 레벨에서 표 구조를 재구성하는 것이 맞다.
    preprocessor에서는 연속 동일 행 제거로 보조 처리한다.
    """
    lines = text.splitlines()
    result: list[str] = []
    prev: str | None = None
    for ln in lines:
        stripped = ln.strip()
        if stripped and stripped == prev:
            continue
        result.append(ln)
        prev = stripped if stripped else prev
    return "\n".join(result)

def _dedup_footnotes(text: str) -> str:
    """
    주1), 주2), ☞, * 로 시작하는 각주 라인의 중복을 제거.
    주1)과 주 1) 처럼 공백 차이가 있는 경우도 동일 주석으로 처리하기 위해
    공백을 모두 제거한 뒤 비교한다.
    """
    lines = text.splitlines()
    seen: set[str] = set()
    result: list[str] = []
    for ln in lines:
        stripped = ln.strip()
        is_footnote = bool(re.match(r"^(주\s*\d+\)|[*☞])", stripped))
        if is_footnote:
            key = re.sub(r"\s+", "", stripped)
            if key in seen:
                continue
            seen.add(key)
        result.append(ln)
    return "\n".join(result)

def _remove_empty_bullets(text: str) -> str:
    """
    ▪, •, -, –, — 기호만 있고 내용이 없는 단독 라인을 삭제.
    """
    return re.sub(r"(?m)^[\s▪•\-–—]+\s*$", "", text)

def _normalize_whitespace(text: str) -> str:
    """3개 이상 연속 빈 줄 → 2개, 연속 공백 → 1개, 앞뒤 정리.
    lstrip() 대신 strip()으로 앞뒤를 모두 정리한다.
    """
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
