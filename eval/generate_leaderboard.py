"""
eval/generate_leaderboard.py
평가 결과 JSON → leaderboard.md 자동 생성

실행:
    uv run python eval/generate_leaderboard.py
    uv run python eval/generate_leaderboard.py --result eval/results/eval_results_xxx.json
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

EVAL_DIR    = Path(__file__).parent
RESULT_DIR  = EVAL_DIR / "results"
LEADER_PATH = EVAL_DIR / "leaderboard.md"


def load_latest_result() -> tuple[list[dict], str]:
    jsons = sorted(RESULT_DIR.glob("eval_results_*.json"), reverse=True)
    if not jsons:
        raise FileNotFoundError("평가 결과 파일 없음. 먼저 run_eval.py를 실행하세요.")
    path = jsons[0]
    return json.loads(path.read_text(encoding="utf-8")), path.name


def avg(values: list) -> str:
    valid = [v for v in values if v is not None]
    if not valid:
        return "-"
    return f"{sum(valid) / len(valid):.4f}"


def faithful_rate(results: list[dict]) -> str:
    if not results:
        return "-"
    n = sum(1 for r in results if r["faithfulness"] == "Faithful")
    return f"{n}/{len(results)} ({n / len(results) * 100:.1f}%)"


def _cs_label(cs) -> str:
    return str(cs) if cs is not None else "기본값"


def generate_leaderboard(results: list[dict], source_file: str) -> str:
    total = len(results)
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# 📊 평가 리더보드",
        "",
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
        f"| Completeness (avg) | {avg([r['completeness'] for r in results])} / 5 |",
        f"| Conciseness (avg) | {avg([r['conciseness'] for r in results])} / 5 |",
        "",
        "---",
        "",
        "## 2. 이미지 기반 vs 텍스트 기반 비교",
        "",
        "| 포맷 | QA수 | ROUGE-L | Faithfulness | Num Acc |",
        "|---|---|---|---|---|",
    ]

    image_based  = [r for r in results if r.get("is_image_based")]
    text_based   = [r for r in results if not r.get("is_image_based")]
    if image_based:
        lines.append(f"| 이미지 기반 (OCR) | {len(image_based)} | {avg([r['rougeL'] for r in image_based])} | {faithful_rate(image_based)} | {avg([r['num_accuracy'] for r in image_based])} |")
    if text_based:
        lines.append(f"| 텍스트 기반 | {len(text_based)} | {avg([r['rougeL'] for r in text_based])} | {faithful_rate(text_based)} | {avg([r['num_accuracy'] for r in text_based])} |")

    lines += [
        "",
        "---",
        "",
        "## 3. 하이퍼파라미터 민감도",
        "",
        "### chunk_size",
        "",
        "| chunk_size | QA수 | Faithfulness | Completeness | ROUGE-L |",
        "|---|---|---|---|---|",
    ]

    chunk_sizes = sorted(
        set(r['chunk_size'] for r in results),
        key=lambda x: (x is not None, x),
    )
    for cs in chunk_sizes:
        subset = [r for r in results if r['chunk_size'] == cs]
        lines.append(
            f"| {_cs_label(cs)} | {len(subset)} | {faithful_rate(subset)} | {avg([r['completeness'] for r in subset])} | {avg([r['rougeL'] for r in subset])} |"
        )

    # chunk_overlap 비교 (결과에 키 있을 경우)
    if any('chunk_overlap' in r for r in results):
        lines += [
            "",
            "### chunk_overlap",
            "",
            "| chunk_overlap | QA수 | Faithfulness | Completeness | ROUGE-L |",
            "|---|---|---|---|---|",
        ]
        overlaps = sorted(set(r.get('chunk_overlap') for r in results), key=lambda x: (x is None, x))
        for co in overlaps:
            subset = [r for r in results if r.get('chunk_overlap') == co]
            lines.append(
                f"| {_cs_label(co)} | {len(subset)} | {faithful_rate(subset)} | {avg([r['completeness'] for r in subset])} | {avg([r['rougeL'] for r in subset])} |"
            )

    # temperature 비교 (결과에 키 있을 경우)
    if any('temperature' in r for r in results):
        lines += [
            "",
            "### temperature",
            "",
            "| temperature | QA수 | Faithfulness | Completeness | ROUGE-L |",
            "|---|---|---|---|---|",
        ]
        temps = sorted(set(r.get('temperature') for r in results), key=lambda x: (x is None, x))
        for t in temps:
            subset = [r for r in results if r.get('temperature') == t]
            lines.append(
                f"| {_cs_label(t)} | {len(subset)} | {faithful_rate(subset)} | {avg([r['completeness'] for r in subset])} | {avg([r['rougeL'] for r in subset])} |"
            )

    lines += [
        "",
        "---",
        "",
        "## 4. 질문 유형별 성능",
        "",
        "| 유형 | QA수 | Faithfulness | Num Acc | Completeness | ROUGE-L |",
        "|---|---|---|---|---|---|",
    ]

    for t in ['factual', 'numerical', 'summary', 'negative', 'multi_doc']:
        subset = [r for r in results if r['type'] == t]
        if subset:
            lines.append(
                f"| {t} | {len(subset)} | {faithful_rate(subset)} | {avg([r['num_accuracy'] for r in subset])} | {avg([r['completeness'] for r in subset])} | {avg([r['rougeL'] for r in subset])} |"
            )

    lines += [
        "",
        "---",
        "",
        "## 5. 문서별 성능",
        "",
        "| 문서 | QA수 | ROUGE-L | Faithfulness | Num Acc |",
        "|---|---|---|---|---|",
    ]

    for doc in sorted(set(r['doc'] for r in results)):
        subset = [r for r in results if r['doc'] == doc]
        lines.append(
            f"| {doc} | {len(subset)} | {avg([r['rougeL'] for r in subset])} | {faithful_rate(subset)} | {avg([r['num_accuracy'] for r in subset])} |"
        )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=str, help="결과 JSON 파일 경로")
    args = parser.parse_args()

    if args.result:
        path    = Path(args.result)
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