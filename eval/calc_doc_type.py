"""
eval/calc_doc_type.py
hyperparam_v3 JSON에서 chunk_size=300 / chunk_overlap=100 조합만 필터해
문서별 · 질문유형별 성능을 출력한다.

실행:
    uv run python eval/calc_doc_type.py
"""
import json
from pathlib import Path

RESULT_PATH = Path(__file__).parent / "results" / "eval_results_hyperparam_v3_20260410_152842.json"

DOC_LABELS = {
    "DS투자증권_시황분석_리포트.pdf":                              "DS투자증권 시황분석 리포트",
    "금융감독원_251125__보도자료__25_10월중_기업의_직접금융_조달실적.docx": "금융감독원 직접금융 조달실적",
    "농협_2022년_9월말_기준_사업보고서.hwp":                        "농협 2022년 사업보고서",
    "미래에셋증권_1분기_실적보고서.pdf":                            "미래에셋증권 1분기 실적보고서",
    "미래에셋증권_2분기_실적보고서.pdf":                            "미래에셋증권 2분기 실적보고서",
    "미래에셋증권_3분기_실적보고서.pdf":                            "미래에셋증권 3분기 실적보고서",
    "미래에셋증권_4분기_실적보고서.pdf":                            "미래에셋증권 4분기 실적보고서",
    "한화투자증권_두산밥캣_기업분석_리포트.pdf":                     "한화투자증권 두산밥캣 리포트",
}

data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
subset = [r for r in data if r["chunk_size"] == 300 and r["chunk_overlap"] == 100]
print(f"필터된 레코드 수: {len(subset)}개 (chunk=300, overlap=100)\n")

# ── 문서별 ──────────────────────────────────────────────
print("=" * 70)
print("■ 문서별 성능")
print(f"{'문서':30} {'QA':>4} {'ROUGE-L':>8} {'Faithful':>12} {'NumAcc':>8}")
print("-" * 70)

docs = {}
for r in subset:
    docs.setdefault(r["doc"], []).append(r)

for doc, recs in docs.items():
    n   = len(recs)
    rl  = sum(r["rougeL"] for r in recs) / n
    na  = sum(r["num_accuracy"] for r in recs) / n
    fn  = sum(1 for r in recs if r["faithfulness"] == "Faithful")
    lbl = DOC_LABELS.get(doc, doc[:30])
    print(f"{lbl:30} {n:>4} {rl:>8.4f}  {fn}/{n}({fn/n*100:.1f}%)  {na:>8.4f}")

# ── 유형별 ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("■ 질문 유형별 성능")
print(f"{'유형':12} {'QA':>4} {'ROUGE-L':>8} {'Faithful':>12} {'NumAcc':>8} {'Completeness':>13}")
print("-" * 70)

types = {}
for r in subset:
    types.setdefault(r["type"], []).append(r)

for t in sorted(types):
    recs = types[t]
    n    = len(recs)
    rl   = sum(r["rougeL"] for r in recs) / n
    na   = sum(r["num_accuracy"] for r in recs) / n
    fn   = sum(1 for r in recs if r["faithfulness"] == "Faithful")
    comp = [r["completeness"] for r in recs if r.get("completeness") is not None]
    ca   = sum(comp) / len(comp) if comp else 0
    print(f"{t:12} {n:>4} {rl:>8.4f}  {fn}/{n}({fn/n*100:.1f}%)  {na:>8.4f}  {ca:>13.4f}")
