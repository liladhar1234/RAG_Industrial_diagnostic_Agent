"""
latency_report.py

Phase 3: p50/p95 latency per retrieval stage, aggregated across the eval
set. Reads the per-row timing dicts already captured by eval_agent.py
(retriever.last_timing, logged into eval_agent_results.json) rather than
re-running retrieval - this is purely an aggregation step over data that
already exists.

Usage (inside Docker):
    docker compose exec dev python src/retrieval/eval_agent.py   # if not already run
    docker compose exec dev python src/retrieval/latency_report.py
"""
import argparse
import json


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="data/eval/eval_agent_results.json")
    args = ap.parse_args()

    d = json.load(open(args.infile))
    rows = d["rows"]

    stages = set()
    for r in rows:
        stages.update(r.get("timing", {}).keys())
    stages = sorted(stages)

    if not stages:
        print("No timing data found in", args.infile)
        return

    print(f"Latency per stage, aggregated over {len(rows)} eval rows "
          f"(source: {args.infile})\n")
    print(f"{'stage':<12} {'p50':>10} {'p95':>10} {'max':>10} {'n':>6}")
    for stage in stages:
        vals = [r["timing"].get(stage, 0) for r in rows if stage in r.get("timing", {})]
        p50 = percentile(vals, 0.50)
        p95 = percentile(vals, 0.95)
        mx = max(vals) if vals else None
        print(f"{stage:<12} {p50:>9.1f}ms {p95:>9.1f}ms {mx:>9.1f}ms {len(vals):>6}")

    total_p50 = sum(percentile([r['timing'].get(s, 0) for r in rows if s in r.get('timing', {})], 0.50) or 0 for s in stages)
    total_p95 = sum(percentile([r['timing'].get(s, 0) for r in rows if s in r.get('timing', {})], 0.95) or 0 for s in stages)
    print(f"\n{'sum of per-stage p50s':<25} {total_p50:.1f}ms")
    print(f"{'sum of per-stage p95s':<25} {total_p95:.1f}ms")
    print("(sum of percentiles, not the percentile of end-to-end totals - "
          "note this distinction if reporting)")


if __name__ == "__main__":
    main()