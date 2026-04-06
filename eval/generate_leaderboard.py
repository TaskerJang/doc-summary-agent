"""
eval/generate_leaderboard.py
평가 결과 JSON → leaderboard.md 자동 생성

실행:
    uv run python eval/generate_leaderboard.py
    uv run python eval/generate_leaderboard.py --result eval/results/eval_results_xxx.json
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

EVAL_DIR   = Path(__file__).parent
RESULT_DIR = EVAL_DIR / "results"
LEADER_PATH = EVAL_DIR / "leaderboard.md"


def load_latest_result() -> list[dict]:
    jsons = sorted(RESULT_DIR.glob("eval_results_*.json"), reverse=True)
    if not jsons:
        raise FileNotFoundError("평가 결과 파일 없음. 먼저 run_eval.py를 실행하세요.")
    return json.loads(jsons[0].read_text(encoding="utf-8")), jsons[0].name


def avg(values: list) -> str:
    if not values:
        return "-"
    return f"{sum(values)/len(values):.4f}"


def faithful_rate(results: list[dict]) -> str:
    if not results:
        return "-"
    n = sum(1 for r in results if r["faithfulness"] == "Faithful")
    return f"{n}/{len(results)} ({n/len(results)*100:.1f}%)"


def generate_leaderboard(results: list[dict], source_file: str) -> str:
    total = len(results)
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# 📊 평가 리더보드",
        f"",
        f"> 자동 생성: {now}  |  소스: `{source_file}`  |  총 QA: {total}개",
        "",
        "---",
        "",
        "## 1. 전체 요약",
        "",
        "| 지표 | 값 |",
        "|---|---|",
        f"| 총 QA 수 | {total} |",
        f"| ROUGE-1 | {avg([r['rouge1'] for r in results])} |",
        f"| ROUGE-2 | {avg([r['rouge2'] for r in results])} |",
        f"| ROUGE-L | {avg([r['rougeL'] for r in results])} |",
        f"| 수치 정확도 | {avg([r['num_accuracy'] for r in results])} |",
        f"| Faithfulness | {faithful_rate(results)} |",
        f"| Completeness (avg) | {avg([r['completeness'] for r in results if r['completeness']])} / 5 |",
        f"| Conciseness (avg) | {avg([r['conciseness'] for r in results if r['conciseness']])} / 5 |",
        "",
        "---",
        "",
        "## 2. OCR vs 텍스트 PDF 비교",
        "",
        "| 문서 | 포맷 | Faithfulness | Numerical Acc | ROUGE-L |",
        "|---|---|---|---|---|",
    ]

    # OCR vs 텍스트 — doc명에 'OCR' 또는 이미지 기반 여부로 분류
    ocr_docs  = [r for r in results if '1Q' in r['doc'] or '2Q' in r['doc'] or '3Q' in r['doc']]
    text_docs = [r for r in results if '4Q' in r['doc']]
    if ocr_docs:
        lines.append(f"| 미래에셋 1Q~3Q | OCR | {faithful_rate(ocr_docs)} | {avg([r['num_accuracy'] for r in ocr_docs])} | {avg([r['rougeL'] for r in ocr_docs])} |")
    if text_docs:
        lines.append(f"| 미래에셋 4Q | 텍스트 | {faithful_rate(text_docs)} | {avg([r['num_accuracy'] for r in text_docs])} | {avg([r['rougeL'] for r in text_docs])} |")

    lines += [
        "",
        "---",
        "",
        "## 3. chunk_size 민감도",
        "",
        "| chunk_size | Faithfulness | Completeness | ROUGE-L |",
        "|---|---|---|---|",
    ]

    chunk_sizes = sorted(set(r['chunk_size'] for r in results if r['chunk_size']))
    for cs in chunk_sizes:
        subset = [r for r in results if r['chunk_size'] == cs]
        lines.append(f"| {cs} | {faithful_rate(subset)} | {avg([r['completeness'] for r in subset])} | {avg([r['rougeL'] for r in subset])} |")
    if not chunk_sizes:
        lines.append("| 기본값 | - | - | - |")

    lines += [
        "",
        "---",
        "",
        "## 4. 질문 유형별 성능",
        "",
        "| 유형 | Faithfulness | Numerical Acc | Completeness |",
        "|---|---|---|---|",
    ]

    for t in ['factual', 'numerical', 'summary', 'negative', 'multi_doc']:
        subset = [r for r in results if r['type'] == t]
        if subset:
            lines.append(f"| {t} | {faithful_rate(subset)} | {avg([r['num_accuracy'] for r in subset])} | {avg([r['completeness'] for r in subset])} |")

    lines += [
        "",
        "---",
        "",
        "## 5. 문서별 성능",
        "",
        "| 문서 | QA수 | ROUGE-L | Faithfulness | Num Acc |",
        "|---|---|---|---|---|",
    ]

    docs = sorted(set(r['doc'] for r in results))
    for doc in docs:
        subset = [r for r in results if r['doc'] == doc]
        lines.append(f"| {doc} | {len(subset)} | {avg([r['rougeL'] for r in subset])} | {faithful_rate(subset)} | {avg([r['num_accuracy'] for r in subset])} |")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=str, help="결과 JSON 파일 경로")
    args = parser.parse_args()

    if args.result:
        path = Path(args.result)
        results = json.loads(path.read_text(encoding="utf-8"))
        source  = path.name
    else:
        results, source = load_latest_result()

    content = generate_leaderboard(results, source)
    LEADER_PATH.write_text(content, encoding="utf-8")
    print(f"✅ leaderboard.md 생성 완료: {LEADER_PATH}")
    print(content)


if __name__ == "__main__":
    main()
