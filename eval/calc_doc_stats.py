"""
hyperparam_v3 JSON에서 chunk_size=300, chunk_overlap=100 조합만 필터해
문서별 / 질문유형별 성능 집계

실행:
    uv run python eval/calc_doc_stats.py
"""
import json
from pathlib import Path
from collections import defaultdict

RESULT_PATH = Path(__file__).parent / "results" / "eval_results_hyperparam_v3_20260410_152842.json"

DOC_LABEL = {
    "DS투자증권_시황분석_리포트.pdf":                               "DS투자증권 시황분석 리포트",
    "금융감독원_251125__보도자료__25_10월중_기업의_직접금융_조달실적.docx": "금융감독원 직접금융 조달실적",
    "농협_2022년_9월말_기준_사업보고서.hwp":                        "농협 2022년 사업보고서",
    "미래에셋증권_1분기_실적보고서.pdf":                            "미래에셋증권 1분기 실적보고서",
    "미래에셋증권_2분기_실적보고서.pdf":                            "미래에셋증권 2분기 실적보고서",
    "미래에셋증권_3분기_실적보고서.pdf":                            "미래에셋증권 3분기 실적보고서",
    "미래에셋증권_4분기_실적보고서.pdf":                            "미래에셋증권 4분기 실적보고서",
    "한화투자증권_두산밥캣_기업분석_리포트.pdf":                    "한화투자증권 두산밥캣 리포트",
}

data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
subset = [r for r in data if r["chunk_size"] == 300 and r["chunk_overlap"] == 100]
print(f"필터 결과: {len(subset)}개 (chunk_size=300, chunk_overlap=100)\n")

# ── 문서별 집계 ──────────────────────────────────────
print("[문서별 성능]")
print(f"{'문서명':<32} {'QA':>4}  {'ROUGE-L':>8}  {'Faithfulness':>14}  {'수치정확도':>10}")

doc_groups = defaultdict(list)
for r in subset:
    doc_groups[r["doc"]].append(r)

for doc_key, label in DOC_LABEL.items():
    recs = doc_groups.get(doc_key, [])
    if not recs:
        continue
    n = len(recs)
    rl = sum(r["rougeL"] for r in recs) / n
    na = sum(r["num_accuracy"] for r in recs) / n
    faith = sum(1 for r in recs if r["faithfulness"] == "Faithful")
    print(f"  {label:<30} n={n:>2}  RL={rl:.4f}  Faith={faith}/{n}({faith/n*100:.1f}%)  NA={na:.4f}")

# ── 질문 유형별 집계 ──────────────────────────────────
print("\n[질문 유형별 성능]")
print(f"{'유형':<12} {'QA':>4}  {'ROUGE-L':>8}  {'Faithfulness':>14}  {'수치정확도':>10}")

type_groups = defaultdict(list)
for r in subset:
    type_groups[r["type"]].append(r)

for qtype in ["factual", "numerical", "summary", "negative"]:
    recs = type_groups.get(qtype, [])
    if not recs:
        continue
    n = len(recs)
    rl = sum(r["rougeL"] for r in recs) / n
    na = sum(r["num_accuracy"] for r in recs) / n
    faith = sum(1 for r in recs if r["faithfulness"] == "Faithful")
    comp_vals = [r["completeness"] for r in recs if r.get("completeness") is not None]
    avg_comp = sum(comp_vals) / len(comp_vals) if comp_vals else 0
    print(f"  {qtype:<12} n={n:>3}  RL={rl:.4f}  Faith={faith}/{n}({faith/n*100:.1f}%)  NA={na:.4f}  Comp={avg_comp:.2f}")
