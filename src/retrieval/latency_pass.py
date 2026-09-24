"""
latency_pass.py

Phase 3: standalone latency capture, separate from eval_agent.py's
scoring pass. Re-runs the same eval set's in-scope, search-path queries
through RealRetriever.search_events() directly (bypassing exact-code
lookups, which never touch dense/BM25/rerank) and records
retriever.last_timing per row, then reports p50/p95 per stage.

Kept separate from eval_agent.py deliberately: that script's current
version doesn't capture timing, and duplicating its exact scoring logic
here isn't necessary - this only needs the search_events() call and the
timing dict it produces as a side effect.

Usage (inside Docker):
    docker compose exec dev python src/retrieval/latency_pass.py \\
        --eval-set data/eval/eval_tune.json
"""
import argparse
import json

from agent.real_retriever import RealRetriever


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="data/eval/eval_tune.json")
    ap.add_argument("--out", default="data/eval/latency_results.json")
    args = ap.parse_args()

    items = json.load(open(args.eval_set))
    # Only in-scope, non-exact-code-shaped queries actually exercise
    # search_events() - exact/typed codes resolve via get_event() and
    # never touch dense/BM25/rerank (see nodes.py's retrieve() guard).
    skip_types = {"out_of_scope", "param_lookup_out_of_scope"}
    items = [i for i in items if i.get("type") not in skip_types]

    retriever = RealRetriever()
    rows = []
    for item in items:
        retriever.search_events(item["query"], k=10)
        rows.append({"query": item["query"], "type": item.get("type"),
                      "timing": dict(retriever.last_timing)})

    stages = sorted({s for r in rows for s in r["timing"].keys()})
    print(f"Latency per stage, {len(rows)} queries through search_events()\n")
    print(f"{'stage':<12} {'p50':>10} {'p95':>10} {'max':>10}")
    summary = {}
    for stage in stages:
        vals = [r["timing"].get(stage, 0) for r in rows]
        p50, p95, mx = percentile(vals, 0.5), percentile(vals, 0.95), max(vals)
        summary[stage] = {"p50_ms": p50, "p95_ms": p95, "max_ms": mx}
        print(f"{stage:<12} {p50:>9.1f}ms {p95:>9.1f}ms {mx:>9.1f}ms")

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()