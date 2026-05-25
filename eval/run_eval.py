"""
eval/run_eval.py
평가 파이프라인 실행 진입점 (#127 — Tier 2 메트릭 4개 추가)

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

## 실행

    uv run python eval/run_eval.py
    uv run python eval/run_eval.py --chunk-size 500 700 1000
    uv run python eval/run_eval.py --no-dense --tag bm25_only
    uv run python eval/run_eval.py --no-semantic --tag fast_dryrun  # bge-m3 끄기

#127 LLM 토글 — OpenRouter 경유 4 모델 측정 + Claude Haiku 4.5 judge:

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
    chunks = step3.get("chunks", [])
    return [c["text"] for c in chunks if c.get("text", "").strip()]


async def run_pipeline(
    doc_path: Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    use_dense: bool = True,
) -> dict:
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


async def evaluate_qa(
    qa: dict,
    summary: SummaryResult,
    step3: dict,
    use_bm25: bool = True,
    use_semantic: bool = True,
) -> dict:
    question  = qa["question"]
    reference = qa["answer"]
    qa_type   = qa["type"]
    # #127: qa_set / key_entities — VectorRAG / GraphRAG 분계 필드
    qa_set       = qa.get("qa_set", "vectorrag")  # 기본 vectorrag (기존 qa_pairs.json 호환)
    key_entities = qa.get("key_entities", [])

    if use_bm25:
        raw_chunks     = _extract_raw_chunks(step3)
        pinned_indices = None
    else:
        raw_chunks     = []
        pinned_indices = []

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
        "faithfulness":        judge.get("faithfulness", "Error"),
        "faithfulness_reason": judge.get("faithfulness_reason", ""),
        "completeness":        judge.get("completeness", None),
        "conciseness":         judge.get("conciseness", None),
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
    if chunk_sizes is None:
        chunk_sizes = [None]
    if chunk_overlaps is None:
        chunk_overlaps = [None]

    all_results = []

    docs = {}
    for qa in qa_pairs:
        doc = qa["doc"]
        if filter_doc and doc != filter_doc:
            continue
        docs.setdefault(doc, []).append(qa)

    for doc_name, qas in docs.items():
        doc_path = docs_dir / doc_name
        if not doc_path.exists():
            logger.warning("문서 없음: %s", doc_path)
            continue

        for chunk_size in chunk_sizes:
            for chunk_overlap in chunk_overlaps:
                logger.info("=== 문서: %s | chunk_size: %s | chunk_overlap: %s | bm25: %s | dense: %s ===",
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
                    all_results.append(result)
                    logger.info(
                        "QA [%s] ROUGE-L=%.3f Faithfulness=%s Correctness=%s",
                        qa["id"], result["rougeL"], result["faithfulness"],
                        result.get("answer_correctness"),
                    )

    return all_results


def save_results(results: list[dict], tag: str = "") -> Path:
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

    # qa_set 별 분계 (없으면 전체)
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

    types = set(r["type"] for r in results)
    print("\n📋 질문 유형별 ROUGE-L / Answer Correctness")
    for t in sorted(types):
        subset = [r for r in results if r["type"] == t]
        avg_r = sum(r["rougeL"] for r in subset) / len(subset) if subset else 0.0
        ac_vals = [r["answer_correctness"] for r in subset if isinstance(r.get("answer_correctness"), int)]
        avg_a = sum(ac_vals) / len(ac_vals) if ac_vals else 0.0
        print(f"  {t:12s}: ROUGE-L={avg_r:.4f}  Correctness={avg_a:.2f}/5  (n={len(subset)})")


def main():
    parser = argparse.ArgumentParser(description="평가 파이프라인 실행")
    parser.add_argument("--chunk-size",    nargs="+", type=int)
    parser.add_argument("--chunk-overlap", nargs="+", type=int)
    parser.add_argument("--doc",  type=str, default="")
    parser.add_argument("--tag",  type=str, default="")
    parser.add_argument("--no-bm25", action="store_true")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Semantic Similarity 메트릭 끄기 (bge-m3 CPU 부담)")
    # #127 측정 LLM
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--llm-base-url", type=str, default=None)
    parser.add_argument("--llm-api-key-env", type=str, default="OPENAI_API_KEY")
    # #127 Judge LLM
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--judge-base-url", type=str, default=None)
    parser.add_argument("--judge-api-key-env", type=str, default="OPENAI_API_KEY")
    args = parser.parse_args()

    if args.llm_model or args.llm_base_url or args.llm_api_key_env != "OPENAI_API_KEY":
        configure_llm(
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key_env=args.llm_api_key_env,
        )

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
