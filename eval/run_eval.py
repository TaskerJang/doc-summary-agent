"""
eval/run_eval.py
평가 파이프라인 실행 진입점 (#127 + multi-doc QA 지원)

## 메트릭 (Tier 1 + Tier 2)

### Tier 1 — 전통 + FineSurE
- ROUGE-1/2/L
- 수치 정확도
- Faithfulness / Completeness / Conciseness (FineSurE ACL 2024)
- Numerical Faithfulness

### Tier 2 — RAGAS + GraphRAG-Bench (#127)
- Answer Correctness (RAGAS + GraphRAG-Bench Accuracy) — 의미적 일치 1-5점
- Semantic Similarity (RAGAS) — bge-m3 cosine
- Entity Coverage (RAGAS Context Entities Recall 변형)

## Multi-doc QA 지원

평가셋의 doc 필드가 콤마로 여러 파일명을 포함하면 multi-doc QA로 인식하여
각 문서를 독립 파이프라인으로 처리한 뒤 chunks / clean_text / summary 를
병합하여 ask() 1회 호출. RAG 시스템이 multi-doc 영역을 처리할 때의 실제 동작과
한계를 측정하기 위함.

예시 doc 필드:
    "미래에셋증권_1분기_실적보고서.pdf,미래에셋증권_2분기_실적보고서.pdf"

## 실행

    uv run python eval/run_eval.py
    uv run python eval/run_eval.py --chunk-size 500 700 1000
    uv run python eval/run_eval.py --no-dense --tag bm25_only
    uv run python eval/run_eval.py --no-semantic --tag fast_dryrun  # bge-m3 끄기

#127 LLM 토글 — OpenRouter 경유 4 모델 측정 + Claude Haiku 4.5 judge:

기본 패턴 (측정 + judge 모두 OpenRouter):
    OPENROUTER_API_KEY 만 있으면 측정 + judge 둘 다 OpenRouter 경유로 가능.

    # Kimi K2.5 측정 + Claude Haiku 4.5 judge
    uv run python eval/run_eval.py \\
      --llm-model "moonshotai/kimi-k2.5" \\
      --llm-base-url "https://openrouter.ai/api/v1" \\
      --llm-api-key-env "OPENROUTER_API_KEY" \\
      --judge-model "anthropic/claude-haiku-4.5" \\
      --judge-base-url "https://openrouter.ai/api/v1" \\
      --judge-api-key-env "OPENROUTER_API_KEY" \\
      --no-dense --tag "kimi_multi_doc_80qa"

    # DeepSeek V3.2 (BM25 단독)
    uv run python eval/run_eval.py \\
      --llm-model "deepseek/deepseek-v3.2" \\
      --llm-base-url "https://openrouter.ai/api/v1" \\
      --llm-api-key-env "OPENROUTER_API_KEY" \\
      --judge-model "anthropic/claude-haiku-4.5" \\
      --judge-base-url "https://openrouter.ai/api/v1" \\
      --judge-api-key-env "OPENROUTER_API_KEY" \\
      --no-dense --tag "deepseek_bm25"
"""
import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import run_step1, run_step2, run_step3
from summarizer.qa import ask
from summarizer.llm import SummaryResult, SectionSummary, configure_llm

from eval.metrics.rouge_score import compute_rouge
from eval.metrics.numerical_accuracy import compute_numerical_accuracy
from eval.metrics.faithfulness_judge import (
    judge_faithfulness,
    judge_numerical_faithfulness,
    configure_judge_llm,
)
# #127 Tier 2 메트릭
from eval.metrics.answer_correctness import judge_answer_correctness
from eval.metrics.entity_coverage import compute_entity_coverage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR   = Path(__file__).parent
DOCS_DIR   = Path(__file__).parent.parent / "tests" / "step1_parser" / "sample_docs"
QA_PATH    = EVAL_DIR / "dataset" / "qa_pairs.json"
RESULT_DIR = EVAL_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)


def _to_summary_result(result: dict) -> SummaryResult | None:
    summary_dict = result.get("summary")
    if not summary_dict:
        return None
    sections = [SectionSummary(**s) for s in summary_dict.get("sections", [])]
    return SummaryResult(
        overall=summary_dict.get("overall", ""),
        sections=sections,
        is_image_based=summary_dict.get("is_image_based", False),
    )


def _extract_raw_chunks(step3: dict) -> list[str]:
    """step3 결과에서 원문 청크 텍스트 목록 추출."""
    chunks = step3.get("chunks", [])
    return [c["text"] for c in chunks if c.get("text", "").strip()]


async def run_pipeline(
    doc_path: Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    use_dense: bool = True,
) -> dict:
    """단일 문서에 대해 파이프라인 실행 후 결과 반환.

    run_step3는 async (#68 AsyncOpenAI 이관 후속) 이므로 run_pipeline도
    async로 선언하고 await로 호출해야 한다.

    Args:
        use_dense: True면 step3 후 Qdrant에 청크 인덱싱하고 doc_id를
                   step3에 박아 evaluate_qa가 ask()에 전달하도록 한다.
                   False면 인덱싱 스킵 → ask()는 BM25 단독으로 동작.
    """
    logger.info("파이프라인 실행: %s (chunk_size=%s, chunk_overlap=%s, dense=%s)",
                doc_path.name, chunk_size, chunk_overlap, use_dense)

    step1 = run_step1(doc_path)
    if step1.get("status") == "error":
        return step1

    step2_kwargs = {}
    if chunk_size is not None:
        step2_kwargs["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        step2_kwargs["chunk_overlap"] = chunk_overlap

    step2 = run_step2(step1, **step2_kwargs)
    if step2.get("status") == "error":
        return step2

    step3 = await run_step3(step2)

    if use_dense and step3.get("status") != "error":
        try:
            from summarizer.embedder import index_chunks
            chunks_for_index = step3.get("chunks", [])
            if chunks_for_index:
                doc_id = doc_path.stem
                index_chunks(chunks_for_index, doc_id)
                step3["doc_id"] = doc_id
                logger.info("Qdrant 인덱싱 완료: doc_id=%s (%d청크)",
                            doc_id, len(chunks_for_index))
        except Exception as e:
            logger.warning("Qdrant 인덱싱 실패 — BM25 단독 fallback: %s", e)

    return step3


async def _run_pipeline_multi(
    doc_names: list[str],
    docs_dir: Path,
    chunk_size: int | None,
    chunk_overlap: int | None,
    use_dense: bool,
) -> dict | None:
    """multi-doc QA 용 파이프라인 — 여러 문서를 독립 실행 후 결과 병합.

    각 문서를 run_pipeline() 으로 처리한 뒤:
    - chunks: 모든 문서의 chunk 를 합침 (각 chunk 에 source_doc 메타 추가)
    - clean_text: 모든 문서의 본문을 === <doc> === 구분자로 연결
    - summary: 각 문서 summary 의 sections 를 모두 합쳐 단일 SummaryResult 구성
    - doc_id: Qdrant 인덱싱은 use_dense=True 일 때만, 첫 문서 doc_id 사용
      (multi-doc Qdrant 검색은 별도 인프라 필요 — 본 작업 범위 밖)

    Returns:
        병합된 step3 dict. 실패 시 None.
    """
    merged_chunks: list[dict] = []
    merged_clean_text_parts: list[str] = []
    merged_sections: list[dict] = []
    merged_overall_parts: list[str] = []
    is_image_based = False
    first_doc_id = None

    for doc_name in doc_names:
        doc_path = docs_dir / doc_name
        step3 = await run_pipeline(doc_path, chunk_size, chunk_overlap, use_dense=use_dense)
        if step3.get("status") == "error":
            logger.error("multi-doc 내 문서 실패: %s", doc_name)
            return None

        # chunks 병합 — 출처 추적용 source_doc 메타 부착
        for c in step3.get("chunks", []):
            chunk_copy = dict(c)
            chunk_copy["source_doc"] = doc_name
            merged_chunks.append(chunk_copy)

        # clean_text 병합 (judge 가 source_text 로 사용)
        clean = step3.get("clean_text", "")
        if clean:
            merged_clean_text_parts.append(f"=== {doc_name} ===\n{clean}")

        # summary 병합
        summary_dict = step3.get("summary") or {}
        sections = summary_dict.get("sections", [])
        # 각 section 의 heading 에 출처 doc 표시
        for s in sections:
            s_copy = dict(s)
            if "heading" in s_copy:
                s_copy["heading"] = f"[{doc_name}] {s_copy['heading']}"
            merged_sections.append(s_copy)

        overall = summary_dict.get("overall", "")
        if overall:
            merged_overall_parts.append(f"[{doc_name}] {overall}")

        if summary_dict.get("is_image_based"):
            is_image_based = True

        if first_doc_id is None and step3.get("doc_id"):
            first_doc_id = step3.get("doc_id")

    merged_step3 = {
        "chunks": merged_chunks,
        "clean_text": "\n\n".join(merged_clean_text_parts),
        "summary": {
            "overall": "\n\n".join(merged_overall_parts),
            "sections": merged_sections,
            "is_image_based": is_image_based,
        },
        "is_image_based": is_image_based,
        "status": "ok",
    }
    if first_doc_id:
        merged_step3["doc_id"] = first_doc_id

    logger.info("multi-doc 병합 완료 — %d docs, %d chunks, clean_text=%d chars",
                len(doc_names), len(merged_chunks),
                len(merged_step3["clean_text"]))

    return merged_step3


async def evaluate_qa(
    qa: dict,
    summary: SummaryResult,
    step3: dict,
    use_bm25: bool = True,
    use_semantic: bool = True,
) -> dict:
    """단일 QA 쌍에 대해 모든 지표 (Tier 1 + Tier 2) 를 계산한다.

    Tier 1: ROUGE / 수치 정확도 / Faithfulness / Numerical Faithfulness
    Tier 2: Answer Correctness / Semantic Similarity / Entity Coverage

    Args:
        use_bm25: False면 raw_chunks=[] 전달 + pinned_section_indices=[] 로
                  BM25 섹션/청크 검색을 모두 비활성화.
        use_semantic: False면 Semantic Similarity 메트릭 스킵 (CPU 부담 회피).
    """
    question  = qa["question"]
    reference = qa["answer"]
    qa_type   = qa.get("type") or qa.get("pattern", "unknown")
    # #127: qa_set / key_entities — VectorRAG / GraphRAG 분계 필드
    qa_set       = qa.get("qa_set", "vectorrag")
    key_entities = qa.get("key_entities", [])

    if use_bm25:
        raw_chunks          = _extract_raw_chunks(step3)
        pinned_indices      = None
    else:
        raw_chunks          = []
        pinned_indices      = []

    doc_id = step3.get("doc_id")

    qa_result  = await ask(
        question, summary,
        raw_chunks=raw_chunks,
        pinned_section_indices=pinned_indices,
        doc_id=doc_id,
    )
    prediction = qa_result.answer if qa_result.is_answerable else "[답변 불가]"

    # Tier 1
    rouge   = compute_rouge(prediction, reference)
    num_acc = compute_numerical_accuracy(prediction, reference)

    result = {
        "id":             qa["id"],
        "qa_set":         qa_set,
        "doc":            qa["doc"],
        "type":           qa_type,
        "pattern":        qa.get("pattern", ""),
        "question":       question,
        "reference":      reference,
        "prediction":     prediction,
        "is_image_based": step3.get("is_image_based", False),
        "rouge1":         rouge["rouge1"],
        "rouge2":         rouge["rouge2"],
        "rougeL":         rouge["rougeL"],
        "num_accuracy":   num_acc["accuracy"],
        "num_matched":    num_acc["matched"],
        "num_missed":     num_acc["missed"],
        "bm25_enabled":   use_bm25,
        "dense_enabled":  doc_id is not None,
    }

    source_text = step3.get("clean_text", "")
    judge = judge_faithfulness(source_text, prediction)
    result.update({
        "faithfulness":          judge.get("faithfulness", "Error"),
        "faithfulness_reason":   judge.get("faithfulness_reason", ""),
        "completeness":          judge.get("completeness", None),
        "conciseness":           judge.get("conciseness", None),
    })

    num_judge = judge_numerical_faithfulness(source_text, prediction)
    result["numerical_faithfulness"]        = num_judge.get("numerical_faithfulness", "Error")
    result["numerical_faithfulness_reason"] = num_judge.get("reason", "")

    # ── Tier 2 메트릭 ──────────────────────────────────────

    # Answer Correctness (RAGAS + GraphRAG-Bench)
    ac = judge_answer_correctness(question, reference, prediction)
    result["answer_correctness"]        = ac.get("answer_correctness", None)
    result["answer_correctness_reason"] = ac.get("reason", "")

    # Semantic Similarity (RAGAS) — optional
    if use_semantic:
        try:
            from eval.metrics.semantic_similarity import compute_semantic_similarity
            sim = compute_semantic_similarity(prediction, reference)
            result["semantic_similarity"] = sim.get("semantic_similarity", None)
        except Exception as e:
            logger.warning("semantic_similarity 스킵: %s", e)
            result["semantic_similarity"] = None
    else:
        result["semantic_similarity"] = None

    # Entity Coverage (RAGAS Context Entities Recall 변형)
    ec = compute_entity_coverage(prediction, reference, key_entities=key_entities)
    result["entity_coverage"] = ec["entity_coverage"]
    result["entity_matched"]  = ec["matched"]
    result["entity_missed"]   = ec["missed"]
    result["entity_total"]    = ec["total"]

    return result


async def run_eval(
    qa_pairs: list[dict],
    docs_dir: Path,
    chunk_sizes: list[int] | None = None,
    chunk_overlaps: list[int] | None = None,
    filter_doc: str | None = None,
    use_bm25: bool = True,
    use_dense: bool = True,
    use_semantic: bool = True,
) -> list[dict]:
    """전체 평가 파이프라인 실행.

    #multi-doc multi-doc QA 지원:
    - qa["doc"] 가 콤마로 여러 파일명을 포함하면 multi-doc QA 로 처리.
    - 각 문서별로 독립적으로 step1~step3 실행 후, chunk 풀과 summary 를
      병합하여 ask() 1회 호출.
    """
    if chunk_sizes is None:
        chunk_sizes = [None]
    if chunk_overlaps is None:
        chunk_overlaps = [None]

    all_results = []

    # QA 를 single-doc / multi-doc 로 분리
    single_doc_qas: list[dict] = []
    multi_doc_qas: list[dict] = []
    for qa in qa_pairs:
        doc = qa["doc"]
        if filter_doc and filter_doc not in doc:
            continue
        if "," in doc:
            multi_doc_qas.append(qa)
        else:
            single_doc_qas.append(qa)

    logger.info("QA 분포 — single-doc: %d, multi-doc: %d",
                len(single_doc_qas), len(multi_doc_qas))

    # ── single-doc QA 처리 ─────────────────────────────────
    single_docs: dict[str, list[dict]] = {}
    for qa in single_doc_qas:
        single_docs.setdefault(qa["doc"], []).append(qa)

    for doc_name, qas in single_docs.items():
        doc_path = docs_dir / doc_name
        if not doc_path.exists():
            logger.warning("문서 없음 (single-doc): %s", doc_path)
            continue

        for chunk_size in chunk_sizes:
            for chunk_overlap in chunk_overlaps:
                logger.info("=== single-doc | %s | cs=%s ov=%s bm25=%s dense=%s ===",
                            doc_name, chunk_size, chunk_overlap, use_bm25, use_dense)
                step3 = await run_pipeline(
                    doc_path, chunk_size, chunk_overlap, use_dense=use_dense,
                )
                if step3.get("status") == "error":
                    logger.error("파이프라인 실패: %s", step3)
                    continue

                summary = _to_summary_result(step3)
                if not summary:
                    logger.error("요약 결과 없음: %s", doc_name)
                    continue

                for qa in qas:
                    result = await evaluate_qa(
                        qa, summary, step3,
                        use_bm25=use_bm25,
                        use_semantic=use_semantic,
                    )
                    result["chunk_size"]    = chunk_size
                    result["chunk_overlap"] = chunk_overlap
                    result["doc_mode"]      = "single"
                    all_results.append(result)
                    logger.info(
                        "QA [%s] ROUGE-L=%.3f Faithfulness=%s Correctness=%s",
                        qa["id"], result["rougeL"], result["faithfulness"],
                        result.get("answer_correctness"),
                    )

    # ── multi-doc QA 처리 ──────────────────────────────────
    # 각 multi-doc QA 마다 콤마 split → 각 문서 step1~step3 → 결과 병합 → evaluate_qa
    # 같은 doc 조합은 pipeline_cache 로 중복 파싱 회피.
    pipeline_cache: dict[tuple, dict] = {}

    for qa in multi_doc_qas:
        doc_names = [d.strip() for d in qa["doc"].split(",") if d.strip()]

        # 모든 문서 존재 확인
        missing = [d for d in doc_names if not (docs_dir / d).exists()]
        if missing:
            logger.warning("multi-doc QA [%s] 문서 누락: %s — skip", qa["id"], missing)
            continue

        for chunk_size in chunk_sizes:
            for chunk_overlap in chunk_overlaps:
                cache_key = (tuple(sorted(doc_names)), chunk_size, chunk_overlap)

                if cache_key in pipeline_cache:
                    merged_step3 = pipeline_cache[cache_key]
                    logger.info("multi-doc 캐시 hit — %d docs", len(doc_names))
                else:
                    logger.info("=== multi-doc | %d docs | cs=%s ov=%s ===",
                                len(doc_names), chunk_size, chunk_overlap)
                    merged_step3 = await _run_pipeline_multi(
                        doc_names, docs_dir, chunk_size, chunk_overlap, use_dense,
                    )
                    if merged_step3 is None:
                        logger.error("multi-doc 파이프라인 실패 — qa=%s", qa["id"])
                        continue
                    pipeline_cache[cache_key] = merged_step3

                summary = _to_summary_result(merged_step3)
                if not summary:
                    logger.error("multi-doc 요약 결과 없음 — qa=%s", qa["id"])
                    continue

                result = await evaluate_qa(
                    qa, summary, merged_step3,
                    use_bm25=use_bm25,
                    use_semantic=use_semantic,
                )
                result["chunk_size"]    = chunk_size
                result["chunk_overlap"] = chunk_overlap
                result["doc_mode"]      = "multi"
                result["doc_count"]     = len(doc_names)
                all_results.append(result)
                logger.info(
                    "QA [%s] (multi-doc, %d docs) ROUGE-L=%.3f Faithfulness=%s Correctness=%s",
                    qa["id"], len(doc_names), result["rougeL"], result["faithfulness"],
                    result.get("answer_correctness"),
                )

    return all_results


def save_results(results: list[dict], tag: str = "") -> Path:
    """결과를 JSON 파일로 저장."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"eval_results_{tag}_{ts}.json" if tag else f"eval_results_{ts}.json"
    path = RESULT_DIR / name
    try:
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("결과 저장: %s", path)
    except Exception as e:
        logger.error("결과 저장 실패: %s", e)
    return path


def print_summary(results: list[dict]) -> None:
    """평가 결과 요약 출력 — Tier 1 + Tier 2 + qa_set 분계 + doc_mode 분계."""
    if not results:
        print("결과 없음")
        return

    def _avg(rs: list[dict], key: str) -> float:
        vals = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    def _faithful_rate(rs: list[dict]) -> tuple[int, int]:
        f = sum(1 for r in rs if r.get("faithfulness") == "Faithful")
        return f, len(rs)

    bm25_flag  = results[0].get("bm25_enabled", True)
    dense_flag = results[0].get("dense_enabled", False)
    mode_label = []
    if bm25_flag:
        mode_label.append("BM25")
    if dense_flag:
        mode_label.append("Dense+Rerank")
    mode_str = "+".join(mode_label) if mode_label else "Overall-only"

    print("\n" + "=" * 78)
    print(f"📊 평가 결과 요약  (총 {len(results)}개 QA | 검색 모드: {mode_str})")
    print("=" * 78)

    # qa_set 별 분계 (vectorrag / graphrag)
    qa_sets = sorted(set(r.get("qa_set", "vectorrag") for r in results))
    for qa_set in qa_sets:
        subset = [r for r in results if r.get("qa_set", "vectorrag") == qa_set]
        if not subset:
            continue
        f, t = _faithful_rate(subset)
        print()
        print(f"## {qa_set.upper()} QA  (n={len(subset)})")
        print(f"  ── Tier 1 (전통 + FineSurE) ──")
        print(f"  ROUGE-L:               {_avg(subset, 'rougeL'):.4f}")
        print(f"  수치 정확도:             {_avg(subset, 'num_accuracy'):.4f}")
        print(f"  Faithfulness:          {f}/{t} ({(f / t * 100) if t else 0:.1f}%)")
        print(f"  Completeness:          {_avg(subset, 'completeness'):.2f} / 5")
        print(f"  Conciseness:           {_avg(subset, 'conciseness'):.2f} / 5")
        print(f"  ── Tier 2 (RAGAS + GraphRAG-Bench) ──")
        print(f"  Answer Correctness:    {_avg(subset, 'answer_correctness'):.2f} / 5")
        print(f"  Semantic Similarity:   {_avg(subset, 'semantic_similarity'):.4f}")
        print(f"  Entity Coverage:       {_avg(subset, 'entity_coverage'):.4f}")

    print("=" * 78)

    # 질문 유형별 ROUGE-L + Correctness
    types = set(r["type"] for r in results)
    print("\n📋 질문 유형별 ROUGE-L / Answer Correctness")
    for t in sorted(types):
        subset = [r for r in results if r["type"] == t]
        avg_r = sum(r["rougeL"] for r in subset) / len(subset) if subset else 0.0
        ac_vals = [r["answer_correctness"] for r in subset if isinstance(r.get("answer_correctness"), int)]
        avg_a = sum(ac_vals) / len(ac_vals) if ac_vals else 0.0
        print(f"  {t:12s}: ROUGE-L={avg_r:.4f}  Correctness={avg_a:.2f}/5  (n={len(subset)})")

    # ── #multi-doc single-doc / multi-doc 분리 결과 ──────
    single_results = [r for r in results if r.get("doc_mode") == "single"]
    multi_results  = [r for r in results if r.get("doc_mode") == "multi"]

    if single_results and multi_results:
        print("\n" + "=" * 78)
        print(f"📁 문서 모드별 결과")
        print("=" * 78)

        for label, subset in [("single-doc", single_results), ("multi-doc", multi_results)]:
            if not subset:
                continue
            n = len(subset)
            f = sum(1 for r in subset if r.get("faithfulness") == "Faithful")
            print(f"\n  [{label}] n={n}")
            print(f"    ROUGE-L:               {_avg(subset, 'rougeL'):.4f}")
            print(f"    수치 정확도:             {_avg(subset, 'num_accuracy'):.4f}")
            print(f"    Faithfulness:          {f}/{n} ({(f / n * 100) if n else 0:.1f}%)")
            print(f"    Answer Correctness:    {_avg(subset, 'answer_correctness'):.2f} / 5")
            print(f"    Semantic Similarity:   {_avg(subset, 'semantic_similarity'):.4f}")
            print(f"    Entity Coverage:       {_avg(subset, 'entity_coverage'):.4f}")


def main():
    parser = argparse.ArgumentParser(description="평가 파이프라인 실행")
    parser.add_argument("--chunk-size",    nargs="+", type=int,
                        help="청크 크기 목록 (예: 500 700 1000)")
    parser.add_argument("--chunk-overlap", nargs="+", type=int,
                        help="청크 오버랩 목록 (예: 50 100 200)")
    parser.add_argument("--doc",  type=str, default="",
                        help="특정 문서만 평가 (파일명)")
    parser.add_argument("--tag",  type=str, default="",
                        help="결과 파일 태그")
    parser.add_argument("--no-bm25", action="store_true",
                        help="BM25 검색 비활성화 — overall fallback만 사용")
    parser.add_argument("--no-dense", action="store_true",
                        help="Dense(Qdrant) 검색 비활성화 — BM25 단독 측정용")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Semantic Similarity 메트릭 끄기 (bge-m3 CPU 부담 회피)")
    # #127 측정 LLM 토글
    parser.add_argument("--llm-model", type=str, default=None,
                        help="측정 대상 LLM 모델 ID (예: deepseek/deepseek-v3.2)")
    parser.add_argument("--llm-base-url", type=str, default=None,
                        help="측정 대상 LLM OpenAI-호환 endpoint base URL")
    parser.add_argument("--llm-api-key-env", type=str, default="OPENAI_API_KEY",
                        help="측정 대상 LLM API 키 환경변수 이름")
    # #127 Judge LLM 토글
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Judge LLM 모델 ID (예: anthropic/claude-haiku-4.5)")
    parser.add_argument("--judge-base-url", type=str, default=None,
                        help="Judge LLM OpenAI-호환 endpoint base URL")
    parser.add_argument("--judge-api-key-env", type=str, default="OPENAI_API_KEY",
                        help="Judge LLM API 키 환경변수 이름")
    args = parser.parse_args()

    # #127 측정 LLM 재설정
    if args.llm_model or args.llm_base_url or args.llm_api_key_env != "OPENAI_API_KEY":
        configure_llm(
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key_env=args.llm_api_key_env,
        )

    # #127 Judge LLM 재설정
    if args.judge_model or args.judge_base_url or args.judge_api_key_env != "OPENAI_API_KEY":
        configure_judge_llm(
            model=args.judge_model,
            base_url=args.judge_base_url,
            api_key_env=args.judge_api_key_env,
        )

    qa_pairs = json.loads(QA_PATH.read_text(encoding="utf-8"))
    # qa_set 필드 없으면 vectorrag 로 자동 박힘 (기존 호환)
    for qa in qa_pairs:
        qa.setdefault("qa_set", "vectorrag")
    logger.info("QA 쌍 로드: %d개", len(qa_pairs))

    results = asyncio.run(run_eval(
        qa_pairs=qa_pairs,
        docs_dir=DOCS_DIR,
        chunk_sizes=args.chunk_size,
        chunk_overlaps=args.chunk_overlap,
        filter_doc=args.doc or None,
        use_bm25=not args.no_bm25,
        use_dense=not args.no_dense,
        use_semantic=not args.no_semantic,
    ))

    save_results(results, tag=args.tag)
    print_summary(results)


if __name__ == "__main__":
    main()
