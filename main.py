import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import doc_parser
from doc_parser.preprocessor import clean
from doc_parser.metadata import extract
from doc_parser.pdf import is_image_based_pdf
from chunker.chunker import chunk
from summarizer.llm import summarize, SummaryResult

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
    ext  = path.suffix.lower()
    stem = path.stem.lower()
    if ext == ".pdf":
        return "ir_report" if any(k in stem for k in ["실적", "분기", "연간", "ir"]) else "pdf"
    if ext in (".docx", ".doc"):
        return "docx"
    if ext == ".hwp":
        return "hwp"
    return "generic"


# ── Step별 분리 함수 (UI 비저블 리즈닝용) ─────────────────

def run_step1(path: Path) -> dict:
    """
    Step 1: 메타데이터 추출 + 파싱 + 전처리
    Returns: {
        metadata, metadata_error,
        text, parse_len, is_image_based, parse_error,
        clean_text, clean_len, clean_error,
        source, file, status
    }
    """
    result: dict = {"file": path.name, "status": "ok"}
    logger.info("Step 1 시작: %s", path.name)

    # 메타데이터
    try:
        meta = extract(path)
        result["metadata"] = meta.model_dump()
        logger.info("메타데이터 추출 완료 — lang=%s pages=%s", meta.language, meta.page_count)
    except Exception as e:
        logger.warning("메타데이터 추출 실패: %s", e)
        result["metadata_error"] = str(e)

    # 파싱
    is_image_based = False
    try:
        if path.suffix.lower() == ".pdf":
            is_image_based = is_image_based_pdf(path)
        text = doc_parser.parse(path)
        result["text"] = text
        result["parse_len"] = len(text)
        result["is_image_based"] = is_image_based
        logger.info("파싱 완료 — %d자  image_based=%s", len(text), is_image_based)
    except Exception as e:
        logger.error("파싱 실패: %s", e, exc_info=True)
        result.update({"status": "error", "parse_error": str(e)})
        return result

    # 전처리
    source = _source_from_path(path)
    result["source"] = source
    try:
        clean_text = clean(text, source=source)
        result["clean_text"] = clean_text
        result["clean_len"] = len(clean_text)
        logger.info("전처리 완료 — %d자", len(clean_text))
    except Exception as e:
        logger.error("전처리 실패: %s", e, exc_info=True)
        result.update({"status": "error", "clean_error": str(e)})

    return result


def run_step2(step1_result: dict) -> dict:
    """
    Step 2: 청킹
    Returns: step1_result + { chunks, chunk_count, chunk_error }
    """
    result = dict(step1_result)
    clean_text = result.get("clean_text", "")
    logger.info("Step 2 시작 — 청킹")

    try:
        chunks = chunk(clean_text)
        result["chunks"] = chunks
        result["chunk_count"] = len(chunks)
        logger.info("청킹 완료 — 총 %d개", len(chunks))
    except Exception as e:
        logger.error("청킹 실패: %s", e, exc_info=True)
        result.update({"status": "error", "chunk_error": str(e)})

    return result


def run_step3(step2_result: dict) -> dict:
    """
    Step 3: LLM 요약 생성
    Returns: step2_result + { summary, summary_error }
    """
    result = dict(step2_result)
    chunks = result.get("chunks", [])
    is_image_based = result.get("is_image_based", False)
    logger.info("Step 3 시작 — 요약 생성")

    try:
        summary: SummaryResult = summarize(chunks, is_image_based=is_image_based)
        result["summary"] = summary.model_dump()
        logger.info("요약 완료 — 섹션 %d개", len(summary.sections))
    except Exception as e:
        logger.error("요약 실패: %s", e, exc_info=True)
        result.update({
            "status": "partial",
            "summary_error": str(e),
            "summary": {
                "overall": "[요약 생성 실패]",
                "sections": [],
                "is_image_based": is_image_based,
            }
        })

    return result


# ── 기존 run() — CLI 및 하위 호환 유지 ────────────────────

def run(path: Path, no_summary: bool = False) -> dict:
    started_at = datetime.now()
    logger.info("파이프라인 시작: %s", path.name)

    print(f"\n{SEP}\n[Step 1] 파싱 시작: {path.name}\n{SEP}")
    result = run_step1(path)
    if result.get("status") == "error":
        return result

    meta = result.get("metadata", {})
    print(f"  메타데이터  lang={meta.get('language')}  pages={meta.get('page_count')}")
    print(f"  파싱 완료   {result.get('parse_len', 0):,}자")
    print(f"  전처리 완료  {result.get('clean_len', 0):,}자")

    print(f"\n{SEP}\n[Step 2] 청킹\n{SEP}")
    result = run_step2(result)
    if result.get("status") == "error":
        return result

    chunks = result.get("chunks", [])
    print(f"  청킹 완료   총 {len(chunks)}개")
    for c in chunks[:3]:
        section = (c["section"] or "(no section)")[:30]
        print(f"    [{c['chunk_index']:02d}] {section}  {len(c['text'])}자")
    if len(chunks) > 3:
        print(f"    ... 외 {len(chunks) - 3}개")

    if no_summary:
        elapsed = (datetime.now() - started_at).total_seconds()
        logger.info("Step1~2 완료 (--no-summary) — %.1fs", elapsed)
        return result

    print(f"\n{SEP}\n[Step 3] 요약 생성\n{SEP}")
    result = run_step3(result)

    summary = result.get("summary", {})
    print(f"  전체 요약   {len(summary.get('overall', ''))}자")
    print(f"  섹션 요약   {len(summary.get('sections', []))}개")

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
        sys.exit(1)

    result = run(path, no_summary=args.no_summary)

    if args.json:
        output = {k: v for k, v in result.items() if k not in ("chunks", "text", "clean_text")}
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()