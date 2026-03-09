"""Show best and worst run traces from results/."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

runs = list(Path("results/runs").glob("*.json"))
records = []
for p in runs:
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("metrics") and d["metrics"].get("answer") and d.get("example_id"):
        records.append(d)

records.sort(key=lambda x: x["metrics"]["answer"]["overall_score"] or 0, reverse=True)

print("=== EN IYI 3 RUN ===")
for r in records[:3]:
    m = r["metrics"]["answer"]
    ret = r["metrics"]["retrieval"]
    ans = r.get("generated_answer") or {}
    answer_text = ans.get("answer_text", "")[:90]
    print(f"  [{r['example_id']}] exp={r['experiment_id']}")
    print(f"  Q: {r['query'][:75]}")
    print(f"  A: {answer_text}...")
    print(f"  overall={m['overall_score']:.3f}  groundedness={m['groundedness_score']:.3f}  hit_at_k={ret['hit_at_k']:.1f}")
    print()

print("=== EN KOTU 3 RUN ===")
for r in records[-3:]:
    m = r["metrics"]["answer"]
    ret = r["metrics"]["retrieval"]
    fms = [str(f) for f in r["metrics"].get("failure_modes", [])]
    print(f"  [{r['example_id']}] exp={r['experiment_id']}")
    print(f"  Q: {r['query'][:75]}")
    print(f"  overall={m['overall_score']:.3f}  groundedness={m['groundedness_score']:.3f}  hit_at_k={ret['hit_at_k']:.1f}")
    print(f"  Failure modes: {fms}")
    print()

print("=== ORNEK TRACE (eval_001, baseline) ===")
for r in records:
    if r.get("example_id") == "eval_001" and r.get("experiment_id") == "baseline":
        print(f"  run_id:      {r['run_id']}")
        print(f"  timestamp:   {r['timestamp']}")
        print(f"  query:       {r['query']}")
        print(f"  top_k:       {r['top_k']}")
        print(f"  strategy:    {r['prompt_strategy']}")
        print(f"  n_chunks:    {len(r['retrieved_chunks'])}")
        print(f"  answer:      {(r.get('generated_answer') or {}).get('answer_text','')[:120]}...")
        m = r["metrics"]["answer"]
        ret = r["metrics"]["retrieval"]
        print(f"  --- Metrikler ---")
        print(f"  overall_score:       {m['overall_score']:.4f}")
        print(f"  answer_relevance:    {m['answer_relevance_score']:.4f}")
        print(f"  groundedness:        {m['groundedness_score']:.4f}")
        print(f"  citation_coverage:   {m['citation_coverage_score']:.4f}")
        print(f"  hallucination_risk:  {m['hallucination_risk_score']:.4f}")
        print(f"  hit_at_k:            {ret['hit_at_k']:.4f}")
        print(f"  context_recall:      {ret['context_recall']:.4f}")
        print(f"  latency_ms:          {r['metrics']['latency_ms']:.2f}")
        print(f"  failure_modes:       {r['metrics']['failure_modes']}")
        break
