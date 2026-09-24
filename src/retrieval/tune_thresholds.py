"""
tune_thresholds.py

Picks MIN_GAP_RATIO and MIN_TOP_SCORE for agent/nodes.py from data.

It REPLAYS the agent's decision rule (nodes.collapse + nodes.assess) on the
candidate lists that eval_agent.py stored, for a grid of threshold pairs.
For each pair it reports:

    coverage        share of in-scope queries the agent answers confidently
                    (the rest trigger a clarifying question - annoying,
                    but safe)
    precision       share of confident answers that are correct. A confident
                    answer to an out_of_scope query counts as WRONG.
    conf_wrong      confident-and-wrong answers / all queries. This is the
                    failure that matters most for this agent.

Selection rule: maximise coverage subject to precision >= --target, then
take the middle of the plateau of tied grid points (rather than the edge)
so the choice isn't balanced on one question.

Only queries that reach the confidence check are used. Exact-lookup and
unknown-code queries never do, so they are excluded (counts are printed).

Workflow:
    # 1. score the TUNE split (same reranker setting the agent runs with)
    docker compose exec dev python src/retrieval/eval_agent.py \\
        --eval-set data/eval/eval_tune.json --out data/eval/results_tune.json
    # 2. tune
    docker compose exec dev python src/retrieval/tune_thresholds.py \\
        --tune data/eval/results_tune.json
    # 3. paste the two constants into nodes.py, then ONCE, at the end:
    docker compose exec dev python src/retrieval/eval_agent.py \\
        --eval-set data/eval/eval_heldout.json --out data/eval/results_heldout.json
    docker compose exec dev python src/retrieval/tune_thresholds.py \\
        --tune data/eval/results_tune.json --heldout data/eval/results_heldout.json
"""

import argparse
import json
import math

AGENT_K = 5            # nodes.retrieve calls search_events(query, k=5)
CURRENT = (0.15, 0.0)  # MIN_GAP_RATIO, MIN_TOP_SCORE as currently in nodes.py


# ---- replay of the agent's rule (keep in sync with agent/nodes.py) -----------
def collapse(cands):
    """Mirror of nodes.collapse: one entry per (kind, code)."""
    best = {}
    for c in cands:
        key = (c["kind"], c["code"])
        if key not in best or c["score"] > best[key]["score"]:
            best[key] = c
    return sorted(best.values(), key=lambda c: -c["score"])


def decide(cands, min_gap, min_top):
    """Mirror of the confident branch of nodes.assess (search path)."""
    c = collapse(cands[:AGENT_K])
    if not c:
        return False, None
    top = c[0]
    if len(c) == 1:
        gap_ok = True
    else:
        gap = (top["score"] - c[1]["score"]) / max(top["score"], 1e-9)
        gap_ok = gap >= min_gap
    return (gap_ok and top["score"] >= min_top), top


def explain(cands, min_gap, min_top):
    """For printing only: (top, gap, reason) under the same rule as decide()."""
    c = collapse(cands[:AGENT_K])
    if not c:
        return None, None, "no candidates"
    top = c[0]
    gap = None if len(c) == 1 else (top["score"] - c[1]["score"]) / max(top["score"], 1e-9)
    reasons = []
    if top["score"] < min_top:
        reasons.append("low score")
    if gap is not None and gap < min_gap:
        reasons.append("small gap")
    return top, gap, "+".join(reasons) or "passes"


# ---- scoring ------------------------------------------------------------------
def evaluate(rows, g, t):
    c = {"correct": 0, "wrong": 0, "abstain": 0, "oos_accept": 0, "oos_reject": 0}
    detail = []
    for r in rows:
        conf, top = decide(r["cands"], g, t)
        if not r["in_scope"]:
            outcome = "oos_accept" if conf else "oos_reject"
        elif not conf:
            outcome = "abstain"
        elif [top["kind"], top["code"]] == r["expected_key"]:
            outcome = "correct"
        else:
            outcome = "wrong"
        c[outcome] += 1
        detail.append((r, outcome, top))
    return c, detail


def metrics(c, g, t):
    n_in = c["correct"] + c["wrong"] + c["abstain"]
    n_oos = c["oos_accept"] + c["oos_reject"]
    confident = c["correct"] + c["wrong"] + c["oos_accept"]
    n = n_in + n_oos
    return {
        "gap": g, "top": t, "n": n, "n_in": n_in, "n_oos": n_oos,
        "confident": confident, "correct": c["correct"],
        "coverage": (c["correct"] + c["wrong"]) / n_in if n_in else 0.0,
        "precision": c["correct"] / confident if confident else 1.0,
        "conf_wrong": (c["wrong"] + c["oos_accept"]) / n if n else 0.0,
        "oos_reject": c["oos_reject"] / n_oos if n_oos else None,
        "counts": c,
    }


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / d, (centre + half) / d


def build_grid(rows):
    gaps = [round(i * 0.025, 3) for i in range(0, 25)]      # 0.000 .. 0.600
    tops = [round(i * 0.05, 3) for i in range(0, 20)]       # 0.00  .. 0.95
    grid = []
    for g in gaps:
        for t in tops:
            c, _ = evaluate(rows, g, t)
            grid.append(metrics(c, g, t))
    return grid


def centroid_pick(points):
    gm = sum(p["gap"] for p in points) / len(points)
    tm = sum(p["top"] for p in points) / len(points)
    return min(points, key=lambda p: (p["gap"] - gm) ** 2 + (p["top"] - tm) ** 2)


def choose(grid, target):
    ok = [p for p in grid if p["precision"] >= target and p["confident"] > 0]
    if not ok:
        best_prec = max(p["precision"] for p in grid)
        cand = [p for p in grid if p["precision"] == best_prec]
        best_cov = max(p["coverage"] for p in cand)
        tied = [p for p in cand if p["coverage"] == best_cov]
        return centroid_pick(tied), False, tied
    best_cov = max(p["coverage"] for p in ok)
    tied = [p for p in ok if p["coverage"] == best_cov]
    return centroid_pick(tied), True, tied


def frontier(grid):
    """Pareto frontier over (coverage, precision), one representative each."""
    groups = {}
    for p in grid:
        groups.setdefault((round(p["coverage"], 4), round(p["precision"], 4)), []).append(p)
    keys = sorted(groups, key=lambda k: (-k[0], -k[1]))
    out, best_prec = [], -1.0
    for k in keys:
        if k[1] > best_prec:
            out.append(centroid_pick(groups[k]))
            best_prec = k[1]
    return out


# ---- io / printing ------------------------------------------------------------
def load_rows(path):
    with open(path) as f:
        data = json.load(f)
    rows = data["rows"] if isinstance(data, dict) else data
    if rows and "path" not in rows[0]:
        raise SystemExit(
            f"{path} was produced by the old eval_agent.py (no 'path'/'cands' fields). "
            "Re-run the patched eval_agent.py."
        )
    search = [r for r in rows if r["path"] == "search"]
    skipped = {}
    for r in rows:
        if r["path"] != "search":
            skipped[r["path"]] = skipped.get(r["path"], 0) + 1
    return search, skipped


def row_line(label, m):
    lo, hi = wilson(m["correct"], m["confident"])
    oos = "  n/a" if m["oos_reject"] is None else f"{m['oos_reject']:5.0%}"
    return (f"{label:<12} gap>={m['gap']:<5.3f} top>={m['top']:<5.2f} "
            f"coverage={m['coverage']:5.0%}  precision={m['precision']:5.0%} "
            f"[{lo:4.0%}-{hi:4.0%}]  conf_wrong={m['conf_wrong']:5.1%}  oos_rejected={oos}")


def main():
    ap = argparse.ArgumentParser(description="Tune MIN_GAP_RATIO / MIN_TOP_SCORE from eval_agent results.")
    ap.add_argument("--tune", required=True, help="eval_agent results for the TUNE split")
    ap.add_argument("--heldout", help="eval_agent results for the HELD-OUT split (use once, at the end)")
    ap.add_argument("--target", type=float, default=0.95, help="minimum confident-precision")
    ap.add_argument("--all", action="store_true",
                    help="print every query at the chosen thresholds, not just problem cases")
    ap.add_argument("--gap", type=float, help="override: evaluate this MIN_GAP_RATIO on --heldout")
    ap.add_argument("--min-top", type=float, help="override: evaluate this MIN_TOP_SCORE on --heldout")
    args = ap.parse_args()

    rows, skipped = load_rows(args.tune)
    n_in = sum(r["in_scope"] for r in rows)
    n_oos = len(rows) - n_in
    print(f"TUNE: {len(rows)} queries reach the confidence check "
          f"({n_in} in-scope, {n_oos} out-of-scope); "
          f"excluded (never reach it): {skipped or 'none'}")
    if len(rows) < 30:
        print(f"[warn] only {len(rows)} queries: one flipped question moves coverage/precision "
              f"by several points. Treat the chosen thresholds as provisional.")
    if n_oos == 0:
        print("[warn] no out_of_scope queries: MIN_TOP_SCORE cannot be tuned and will look useless.")
    if not rows:
        raise SystemExit("nothing to tune on")

    print(f"\n(replaying agent rule on the top {AGENT_K} candidates, after collapsing "
          f"(kind, code) duplicates)\n")
    print(row_line("current", metrics(evaluate(rows, *CURRENT)[0], *CURRENT)))

    grid = build_grid(rows)
    print("\nPareto frontier (each row is the best coverage available at that precision):")
    for p in frontier(grid)[:15]:
        print(row_line("", p))

    chosen, met, tied = choose(grid, args.target)
    print()
    if not met:
        print(f"[warn] no threshold pair reaches precision >= {args.target:.0%}; "
              f"showing the most precise one instead.")
    print(row_line("CHOSEN", chosen))

    g_lo, g_hi = min(p["gap"] for p in tied), max(p["gap"] for p in tied)
    t_lo, t_hi = min(p["top"] for p in tied), max(p["top"] for p in tied)
    print(f"Plateau of equally good pairs: gap {g_lo:.3f}-{g_hi:.3f}, top {t_lo:.2f}-{t_hi:.2f} "
          f"(grid steps 0.025 / 0.05)")
    if t_hi - t_lo <= 0.05 + 1e-9:
        print("[warn] MIN_TOP_SCORE is pinned to one or two grid points: a handful of queries "
              "(the highest-scoring out-of-scope and the lowest-scoring correct ones, see the table "
              "below) set it. Add in-scope queries before trusting it.")
    if g_hi - g_lo <= 0.05 + 1e-9:
        print("[warn] MIN_GAP_RATIO is pinned to one or two grid points: same caveat.")

    _, detail = evaluate(rows, chosen["gap"], chosen["top"])
    bad = [d for d in detail if d[1] in ("wrong", "oos_accept")]
    ab = [d for d in detail if d[1] == "abstain"]
    print(f"\nConfidently wrong at the chosen thresholds: {len(bad)}")
    print(f"Would ask a clarifying question on {len(ab)} in-scope queries.")

    show = detail if args.all else (bad + ab)
    if show:
        print(f"\n{'outcome':<11} {'top1':>5} {'gap':>6}  {'why not confident':<20} query")
        for r, o, _ in sorted(show, key=lambda d: d[1]):
            top, gap, why = explain(r["cands"], chosen["gap"], chosen["top"])
            t1 = "  n/a" if top is None else f"{top['score']:5.2f}"
            gp = "   n/a" if gap is None else f"{gap:6.3f}"
            ans = "" if top is None else f"  -> {top['kind']} {top['code']}"
            print(f"{o:<11} {t1} {gp}  {why:<20} {r['query'][:50]}{ans}")

    ok_top = [top["score"] for r, o, top in detail if o == "correct"]
    bad_top = [top["score"] for r, o, top in detail if o in ("wrong", "oos_accept")]
    if ok_top and bad_top:
        med = lambda xs: sorted(xs)[len(xs) // 2]
        print(f"Median top-1 score: correct answers {med(ok_top):.2f}, wrong answers {med(bad_top):.2f}")

    print("\nPaste into agent/nodes.py:")
    print(f"MIN_GAP_RATIO = {chosen['gap']}")
    print(f"MIN_TOP_SCORE = {chosen['top']}")

    if args.heldout:
        g = chosen["gap"] if args.gap is None else args.gap
        t = chosen["top"] if args.min_top is None else args.min_top
        hrows, hskipped = load_rows(args.heldout)
        print(f"\n=== HELD-OUT ({len(hrows)} queries reach the check; excluded: {hskipped or 'none'}) ===")
        print("Report this once. If you now change thresholds and re-run it, it is a tuning set.")
        print(row_line("held-out", metrics(evaluate(hrows, g, t)[0], g, t)))
        print(row_line("tune", metrics(evaluate(rows, g, t)[0], g, t)))


if __name__ == "__main__":
    main()