import re
from collections import Counter


def clean(text: str, *, source: str = "generic", preserve_picture_text: bool = False) -> str:
    """노이즈를 제거한 텍스트를 반환한다.

    Args:
        text: 파서가 출력한 원본 텍스트
        source: 문서 포맷
        preserve_picture_text: True 시 picture text 블록 보존
    """
    if source == "hwp":
        text = _remove_css_blocks(text)

    if source == "ir_report":
        text = _convert_br_and_dedup_tables(text)

    if source in ("pdf", "ir_report"):
        text = _remove_picture_omit_tags(text)
        text = _remove_picture_text_blocks(text)

    if source in ("pdf", "ir_report"):
        text = _remove_repeated_titles(text, threshold=3)

    if source in ("pdf", "ir_report"):
        text = _remove_page_numbers(text)

    if source in ("pdf", "ir_report"):
        text = _remove_repeated_lines(text, threshold=2)

    if source == "docx":
        text = _dedup_consecutive_lines(text)

    if source == "docx":
        text = _dedup_footnotes(text)
        text = re.sub(r"(?m)^☞.*출처.*바랍니다.*$", "", text)

    text = _remove_empty_bullets(text)
    text = _normalize_whitespace(text)

    return text


def _remove_css_blocks(text: str) -> str:
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"\.[A-Za-z][\w\-]*[^{]*\{[^}]*\}", "", text, flags=re.DOTALL)
    return text


def _convert_br_and_dedup_tables(text: str) -> str:
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")

    parts = re.split(r"(?=\n\|)", text)
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        lines = part.strip().splitlines()
        table_lines = [ln for ln in lines if "|" in ln]
        if table_lines:
            key = re.sub(r"[|\*\s]", "", table_lines[0])[:60]
        else:
            key = part.strip()[:120]
        if key not in seen:
            seen.add(key)
            result.append(part)
    return "".join(result)


def _remove_picture_omit_tags(text: str) -> str:
    return re.sub(
        r"\*\*==>?\s*picture\s*\[\d+\s*x\s*\d+\]\s*intentionally omitted\s*<==\*\*\n?",
        "",
        text,
    )


def _remove_picture_text_blocks(text: str) -> str:
    return re.sub(
        r"\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*\n?",
        "",
        text,
        flags=re.DOTALL,
    )


def _remove_repeated_titles(text: str, threshold: int = 3) -> str:
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
    return re.sub(r"(?m)^[\s▪•\-–—]+\s*$", "", text)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
