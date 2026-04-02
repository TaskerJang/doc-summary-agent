import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import parser as doc_parser
from parser.preprocessor import clean
from parser.metadata import extract
from parser.pdf import _is_image_based_pdf
from chunker.chunker import chunk
from summarizer.llm import summarize

# ── 로깅 설정 ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pipeline")

SEP = "=" * 60


def _source_from_path(path: Path) -> str:
    """파일 경로에서 preprocessor source 파라미터를 추론."""
    ext  = path.suffix.lower()
    stem = path.stem.lower()
    if ext == ".pdf":
        return "ir_report" if any(k in stem for k in ["실적", "분기", "연간", "ir"]) else "pdf"
    if ext in (".docx", ".doc"):
        return "docx"
    if ext == ".hwp":
        return "hwp"
    return "generic"


def run(path: Path, no_summary: bool = False) -> dict:
    """
    단일 문서에 대해 전체 파이프라인을 실행한다.

    Returns:
        result dict — status / metadata / chunks / summary / *_error 필드 포함
    """
    started_at = datetime.now()
    result: dict = {"file": path.name, "status": "ok"}
    logger.info("파이프라인 시작: %s", path.name)

    # ── Step 1-1: 메타데이터 ──────────────────────────────
    print(f"\n{SEP}")
    print(f"[Step 1] 파싱 시작: {path.name}")
    print(SEP)

    try:
        meta = extract(path)
        result["metadata"] = meta.model_dump()
        logger.info("메타데이터 추출 완료 — lang=%s pages=%s", meta.language, meta.page_count)
        print(f"  메타데이터  lang={meta.language}  pages={meta.page_count}")
    except Exception as e:
        logger.warning("메타데이터 추출 실패: %s", e)
        result["metadata_error"] = str(e)
        print(f"  메타데이터  ⚠️  {e}")

    # ── Step 1-2: 파싱 ────────────────────────────────────
    is_image_based = False
    try:
        if path.suffix.lower() == ".pdf":
            is_image_based = _is_image_based_pdf(path)
        text = doc_parser.parse(path)
        result["parse_len"] = len(text)
        result["is_image_based"] = is_image_based
        logger.info("파싱 완료 — %d자  image_based=%s", len(text), is_image_based)
        print(f"  파싱 완료   {len(text):,}자"
              + ("  [이미지 기반 PDF → OCR]" if is_image_based else ""))
    except Exception as e:
        logger.error("파싱 실패: %s", e, exc_info=True)
        result.update({"status": "error", "parse_error": str(e)})
        print(f"  파싱 실패   ❌ {e}")
        return result

    # ── Step 1-3: 전처리 ──────────────────────────────────
    source = _source_from_path(path)
    try:
        clean_text = clean(text, source=source)
        pct = (len(text) - len(clean_text)) / len(text) * 100 if text else 0
        result["clean_len"] = len(clean_text)
        logger.info("전처리 완료 — %d자 (%.1f%%↓)", len(clean_text), pct)
        print(f"  전처리 완료  {len(clean_text):,}자  ({pct:.1f}%↓)")
    except Exception as e:
        logger.error("전처리 실패: %s", e, exc_info=True)
        result.update({"status": "error", "clean_error": str(e)})
        print(f"  전처리 실패  ❌ {e}")
        return result

    # ── Step 2: 청킹 ──────────────────────────────────────
    print(f"\n{SEP}")
    print("[Step 2] 청킹")
    print(SEP)

    try:
        chunks = chunk(clean_text)
        result["chunks"]      = chunks
        result["chunk_count"] = len(chunks)
        logger.info("청킹 완료 — 총 %d개", len(chunks))
        print(f"  청킹 완료   총 {len(chunks)}개")
        for c in chunks[:3]:
            section = (c["section"] or "(no section)")[:30]
            print(f"    [{c['chunk_index']:02d}] {section}  {len(c['text'])}자")
        if len(chunks) > 3:
            print(f"    ... 외 {len(chunks) - 3}개")
    except Exception as e:
        logger.error("청킹 실패: %s", e, exc_info=True)
        result.update({"status": "error", "chunk_error": str(e)})
        print(f"  청킹 실패   ❌ {e}")
        return result

    if no_summary:
        elapsed = (datetime.now() - started_at).total_seconds()
        logger.info("Step1~2 완료 (--no-summary) — %.1fs", elapsed)
        print(f"\n✅ Step1~2 완료  ({elapsed:.1f}s)")
        return result

    # ── Step 3: 요약 ──────────────────────────────────────
    print(f"\n{SEP}")
    print("[Step 3] 요약 생성")
    print(SEP)

    try:
        summary = summarize(chunks, is_image_based=is_image_based)
        result["summary"] = summary.model_dump()
        logger.info("요약 완료 — 전체요약 %d자  섹션 %d개",
                    len(summary.overall), len(summary.sections))
        print(f"  전체 요약   {len(summary.overall)}자")
        print(f"  섹션 요약   {len(summary.sections)}개")
        print()
        print("── 전체 요약 " + "─" * 46)
        print(summary.overall)
        print()
        print("── 섹션별 요약 " + "─" * 44)
        for sec in summary.sections:
            print(f"\n### {sec.section}  ({sec.source})")
            for bullet in sec.bullets:
                print(f"  · {bullet}")
    except Exception as e:
        logger.error("요약 실패: %s", e, exc_info=True)
        result.update({"status": "error", "summary_error": str(e)})
        print(f"  요약 실패   ❌ {e}")

    elapsed = (datetime.now() - started_at).total_seconds()
    logger.info("파이프라인 완료 — %.1fs", elapsed)
    print(f"\n✅ 완료  ({elapsed:.1f}s)")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="doc-summary-agent 파이프라인")
    ap.add_argument("path",         help="문서 경로 (PDF / DOCX / HWP / DOC)")
    ap.add_argument("--no-summary", action="store_true", help="Step1~2만 실행")
    ap.add_argument("--json",       action="store_true", help="결과를 JSON으로 출력")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        logger.error("파일 없음: %s", path)
        print(f"[ERROR] 파일 없음: {path}")
        sys.exit(1)

    result = run(path, no_summary=args.no_summary)

    if args.json:
        # chunks는 용량 커서 JSON 출력 시 제외
        output = {k: v for k, v in result.items() if k != "chunks"}
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()