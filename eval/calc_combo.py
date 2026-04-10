import json
from pathlib import Path

result_path = Path(__file__).parent / "results" / "eval_results_hyperparam_v3_20260410_152842.json"
data = json.loads(result_path.read_text(encoding="utf-8"))

combos = [(300,50),(300,100),(500,50),(500,100),(700,50),(700,100)]
for cs, co in combos:
    s = [r for r in data if r["chunk_size"]==cs and r["chunk_overlap"]==co]
    n = len(s)
    rl = sum(r["rougeL"] for r in s) / n
    na = sum(r["num_accuracy"] for r in s) / n
    faith = sum(1 for r in s if r["faithfulness"]=="Faithful")
    print(f"{cs}/{co}: n={n}, RL={rl:.4f}, NA={na:.4f}, Faith={faith}/{n}({faith/n*100:.1f}%)")
