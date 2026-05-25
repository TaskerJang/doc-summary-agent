"""
eval/run_eval.py
평가 파이프라인 실행 진입점

실행:
    uv run python eval/run_eval.py
    uv run python eval/run_eval.py --chunk-size 500 700 1000
    uv run python eval/run_eval.py --chunk-overlap 50 100 200
    uv run python eval/run_eval.py --chunk-size 500 --chunk-overlap 100 --tag baseline_v3
    uv run python eval/run_eval.py --chunk-size 500 --chunk-overlap 100 --no-bm25 --tag no_bm25_v3
    uv run python eval/run_eval.py --doc 한화투자증권_두산밥캣_기업분석_리포트.pdf
    uv run python eval/run_eval.py --no-dense --tag bm25_only       # Qdrant 미사용 (BM25 단독)

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
      --tag "kimi_k2_5"

    # DeepSeek V3.2
    uv run python eval/run_eval.py \\
      --llm-model "deepseek/deepseek-v3.2" \\
      --llm-base-url "https://openrouter.ai/api/v1" \\
      --llm-api-key-env "OPENROUTER_API_KEY" \\
      --judge-model "anthropic/claude-haiku-4.5" \\
      --judge-base-url "https://openrouter.ai/api/v1" \\
      --judge-api-key-env "OPENROUTER_API_KEY" \\
      --tag "deepseek_v32"

    # GPT-5 Mini
    uv run python eval/run_eval.py \\
      --llm-model "openai/gpt-5-mini" \\
      --llm-base-url "https://openrouter.ai/api/v1" \\
      --llm-api-key-env "OPENROUTER_API_KEY" \\
      --judge-model "anthropic/claude-haiku-4.5" \\
      --judge-base-url "https://openrouter.ai/api/v1" \\
      --judge-api-key-env "OPENROUTER_API_KEY" \\
      --tag "gpt5_mini"

    # Grok 4.20
    uv run python eval/run_eval.py \\
      --llm-model "x-ai/grok-4.20" \\
      --llm-base-url "https://openrouter.ai/api/v1" \\
      --llm-api-key-env "OPENROUTER_API_KEY" \\
      --judge-model "anthropic/claude-haiku-4.5" \\
      --judge-base-url "https://openrouter.ai/api/v1" \\
      --judge-api-key-env "OPENROUTER_API_KEY" \\
      --tag "grok_4_20"

    # 인자 미지정 시 기본 GPT-5.2 OpenAI 직결 (기존 동작, OPENAI_API_KEY 필요)
    uv run python eval/run_eval.py --tag baseline_gpt52
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
    async로 선언하고 await로 호출해야 한다. sync로 두면 coroutine이
    그대로 반환되어 step3.get(...) 호출 시 AttributeError 발생.
    (evaluate_qa는 이미 await로 갱신됐는데 run_pipeline은 누락된
     stale dependency 였음 — 4월 중순부터 평가 경로가 동작 불능.)

    Args:
        use_dense: True면 step3 후 Qdrant에 청크 인덱싱하고 doc_id를
                   step3에 박아 evaluate_qa가 ask()에 전달하도록 한다.
                   False면 인덱싱 스킵 → ask()는 BM25 단독으로 동작
                   (기존 README 평가 시점과 동일 조건).
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

    # Qdrant 인덱싱 — 평가 경로에 dense 검색 활성화
    # 기존엔 doc_id 미전달로 ask()가 BM25 단독으로 떨어졌다.
    # use_dense=True 일 때만 인덱싱하고 doc_id를 step3에 박는다.
    # 인덱싱 실패 시 doc_id 미설정 → ask() 자동 BM25 fallback.
    if use_dense and step3.get("status") != "error":
        try:
            from summarizer.embedder import index_chunks
            chunks_for_index = step3.get("chunks", [])
            if chunks_for_index:
                doc_id = doc_path.stem  # 파일명(확장자 제외)을 doc_id로 사용
                index_chunks(chunks_for_index, doc_id)
                step3["doc_id"] = doc_id
                logger.info("Qdrant 인덱싱 완료: doc_id=%s (%d청크)",
                            doc_id, len(chunks_for_index))
        except Exception as e:
            logger.warning("Qdrant 인덱싱 실패 — BM25 단독 fallback: %s", e)

    return step3


async def evaluate_qa(qa: dict, summary: SummaryResult, step3: dict, use_bm25: bool = True) -> dict:
    """단일 QA 쌍에 대해 모든 지표를 계산한다.

    ask()는 async def이므로 await 필요. (#68 AsyncOpenAI 이관 이후
    run_eval.py 쪽이 갱신되지 않아 coroutine을 그대로 반환받던 버그를
    #107 작업과 함께 수정.)

    step3에 doc_id가 박혀있으면 ask()에 전달 → BM25+Dense+RRF+Reranker
    풀 하이브리드 경로. 없으면 BM25 단독으로 동작 (자동 fallback).

    Args:
        use_bm25: False면 raw_chunks=[] 전달 + pinned_section_indices=[] 로
                  BM25 섹션/청크 검색을 모두 비활성화 (BM25 도입 전 기준값 측정용)
    """
    question  = qa["question"]
    reference = qa["answer"]
    qa_type   = qa["type"]

    if use_bm25:
        # BM25 활성화: 원문 청크 전달 → _find_relevant_chunks_bm25 + _find_relevant_sections_bm25 동작
        raw_chunks          = _extract_raw_chunks(step3)
        pinned_indices      = None
    else:
        # BM25 비활성화: 청크·섹션 BM25 검색 모두 스킵 → overall fallback만 사용
        raw_chunks          = []
        pinned_indices      = []   # 빈 리스트 → relevant_sections = []

    doc_id = step3.get("doc_id")  # run_pipeline에서 인덱싱 성공 시 박힌 값

    qa_result  = await ask(
        question, summary,
        raw_chunks=raw_chunks,
        pinned_section_indices=pinned_indices,
        doc_id=doc_id,
    )
    prediction = qa_result.answer if qa_result.is_answerable else "[답변 불가]"

    rouge   = compute_rouge(prediction, reference)
    num_acc = compute_numerical_accuracy(prediction, reference)

    result = {
        "id":             qa["id"],
        "doc":            qa["doc"],
        "type":           qa_type,
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

    return result


async def run_eval(
    qa_pairs: list[dict],
    docs_dir: Path,
    chunk_sizes: list[int] | None = None,
    chunk_overlaps: list[int] | None = None,
    filter_doc: str | None = None,
    use_bm25: bool = True,
    use_dense: bool = True,
) -> list[dict]:
    """전체 평가 파이프라인 실행."""
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
                    result = await evaluate_qa(qa, summary, step3, use_bm25=use_bm25)
                    result["chunk_size"]    = chunk_size
                    result["chunk_overlap"] = chunk_overlap
                    all_results.append(result)
                    logger.info("QA [%s] ROUGE-L=%.3f Faithfulness=%s",
                                qa["id"], result["rougeL"], result["faithfulness"])

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
    """평가 결과 요약 출력."""
    if not results:
        print("결과 없음")
        return

    total       = len(results)
    avg_rouge1  = sum(r["rouge1"]      for r in results) / total
    avg_rouge2  = sum(r["rouge2"]      for r in results) / total
    avg_rougeL  = sum(r["rougeL"]      for r in results) / total
    avg_num_acc = sum(r["num_accuracy"] for r in results) / total
    faithful    = sum(1 for r in results if r["faithfulness"] == "Faithful")

    comp_vals = [r["completeness"] for r in results if r["completeness"] is not None]
    conc_vals = [r["conciseness"]  for r in results if r["conciseness"]  is not None]
    avg_comp  = sum(comp_vals) / len(comp_vals) if comp_vals else 0.0
    avg_conc  = sum(conc_vals) / len(conc_vals) if conc_vals else 0.0

    bm25_flag  = results[0].get("bm25_enabled", True)
    dense_flag = results[0].get("dense_enabled", False)
    mode_label = []
    if bm25_flag:
        mode_label.append("BM25")
    if dense_flag:
        mode_label.append("Dense+Rerank")
    mode_str = "+".join(mode_label) if mode_label else "Overall-only"

    print("\n" + "=" * 60)
    print(f"📊 평가 결과 요약  (총 {total}개 QA | 검색 모드: {mode_str})")
    print("=" * 60)
    print(f"  ROUGE-1:           {avg_rouge1:.4f}")
    print(f"  ROUGE-2:           {avg_rouge2:.4f}")
    print(f"  ROUGE-L:           {avg_rougeL:.4f}")
    print(f"  수치 정확도:       {avg_num_acc:.4f}")
    print(f"  Faithfulness:      {faithful}/{total} ({faithful/total*100:.1f}%)")
    print(f"  Completeness 평균: {avg_comp:.2f} / 5")
    print(f"  Conciseness 평균:  {avg_conc:.2f} / 5")
    print("=" * 60)

    types = set(r["type"] for r in results)
    print("\n📋 질문 유형별 ROUGE-L")
    for t in sorted(types):
        subset = [r for r in results if r["type"] == t]
        avg    = sum(r["rougeL"] for r in subset) / len(subset)
        print(f"  {t:12s}: {avg:.4f}  (n={len(subset)})")


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
                        help="BM25 검색 비활성화 (섹션·청크 BM25 모두 OFF, overall fallback만 사용) — BM25 도입 전 기준값 측정용")
    parser.add_argument("--no-dense", action="store_true",
                        help="Dense(Qdrant) 검색 비활성화 — BM25 단독 측정용. README 시점과 동일 조건 재현 가능.")
    # #127 측정 LLM 토글
    parser.add_argument("--llm-model", type=str, default=None,
                        help="측정 대상 LLM 모델 ID (예: openai/gpt-5-mini, moonshotai/kimi-k2.5, deepseek/deepseek-v3.2, x-ai/grok-4.20). 미지정 시 기본 GPT-5.2.")
    parser.add_argument("--llm-base-url", type=str, default=None,
                        help="측정 대상 LLM OpenAI-호환 endpoint base URL (예: https://openrouter.ai/api/v1). 미지정 시 OpenAI 기본.")
    parser.add_argument("--llm-api-key-env", type=str, default="OPENAI_API_KEY",
                        help="측정 대상 LLM API 키 환경변수 이름. 기본 OPENAI_API_KEY, OpenRouter 경유 시 OPENROUTER_API_KEY.")
    # #127 Judge LLM 토글 — 측정 대상과 독립 설정 (bias 방지)
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Judge LLM 모델 ID (예: anthropic/claude-haiku-4.5). 미지정 시 기본 GPT-5.2.")
    parser.add_argument("--judge-base-url", type=str, default=None,
                        help="Judge LLM OpenAI-호환 endpoint base URL (예: https://openrouter.ai/api/v1).")
    parser.add_argument("--judge-api-key-env", type=str, default="OPENAI_API_KEY",
                        help="Judge LLM API 키 환경변수 이름. 기본 OPENAI_API_KEY, OpenRouter 경유 시 OPENROUTER_API_KEY.")
    args = parser.parse_args()

    # #127 측정 LLM 재설정 — 인자 명시 시에만 호출. 미지정 시 기존 GPT-5.2 + OpenAI 직결.
    if args.llm_model or args.llm_base_url or args.llm_api_key_env != "OPENAI_API_KEY":
        configure_llm(
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key_env=args.llm_api_key_env,
        )

    # #127 Judge LLM 재설정 — 측정 대상과 독립. 인자 명시 시에만 호출.
    if args.judge_model or args.judge_base_url or args.judge_api_key_env != "OPENAI_API_KEY":
        configure_judge_llm(
            model=args.judge_model,
            base_url=args.judge_base_url,
            api_key_env=args.judge_api_key_env,
        )

    qa_pairs = json.loads(QA_PATH.read_text(encoding="utf-8"))
    logger.info("QA 쌍 로드: %d개", len(qa_pairs))

    results = asyncio.run(run_eval(
        qa_pairs=qa_pairs,
        docs_dir=DOCS_DIR,
        chunk_sizes=args.chunk_size,
        chunk_overlaps=args.chunk_overlap,
        filter_doc=args.doc or None,
        use_bm25=not args.no_bm25,
        use_dense=not args.no_dense,
    ))

    save_results(results, tag=args.tag)
    print_summary(results)


if __name__ == "__main__":
    main()
