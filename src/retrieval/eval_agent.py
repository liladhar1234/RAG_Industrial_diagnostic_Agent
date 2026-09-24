"""
eval_agent.py

Evaluates retrieval as the AGENT actually sees it: through RealRetriever,
which restricts search to fault/warning chunks only (see real_retriever.py's
docstring). eval.py's numbers are the Phase 2 dense/BM25/hybrid comparison
over ALL chunks (params + events) and do NOT reflect the deployed agent's
scoped pipeline - don't tune MIN_GAP_RATIO/MIN_TOP_SCORE against eval.py's
output, tune against this instead.

Changes in this version (needed for threshold tuning):
  - Code resolution mirrors nodes.parse_symptoms for EVERY item, not just
    type "exact_code". A typed code that exists -> exact lookup; a typed code
    that does not exist -> "unknown_code" (no search), as in the real agent.
  - Items of type "out_of_scope" (expected_id null) and
    "param_lookup_out_of_scope" (expected_id = a parameter chunk) are
    supported. They are excluded from hit@k / MRR and scored by
    tune_thresholds.py instead (the agent should NOT answer them confidently).
  - Search rows store `cands` (chunk_id, kind, code, score for every returned
    candidate) and `expected_key` ([kind, code] of the expected event), so
    tune_thresholds.py can replay nodes.collapse() + nodes.assess() exactly.
    The old gap_ratio was computed on RAW candidates; the agent computes it
    after collapsing (kind, code) duplicates, so the two can differ.

Usage (inside Docker):
    docker compose exec dev python src/retrieval/eval_agent.py \\
        --eval-set data/eval/eval_tune.json \\
        --out data/eval/results_tune.json
"""

import argparse
import json
import os

from agent.nodes import CONTEXT_CODE, BARE_CODE
from agent.real_retriever import RealRetriever


# Types the AGENT should not answer confidently. "param_lookup_out_of_scope" is
# a parameter question (e.g. "parameter 36.09"): its expected_id is a parameter
# chunk, which is valid for eval.py (all chunks) but unreachable by design for
# the agent, whose index only holds fault/warning chunks.
OUT_OF_SCOPE_TYPES = {"out_of_scope", "param_lookup_out_of_scope"}


def load_eval_set(path):
    with open(path) as f:
        return json.load(f)


def resolve_typed(query, retriever):
    """Mirror of nodes.parse_symptoms (single-turn). Keep the two in sync."""
    typed = None
    ctx = CONTEXT_CODE.findall(query)
    if ctx:
        typed = ctx[-1].upper()
    else:
        for tok in reversed(BARE_CODE.findall(query)):
            if retriever.get_event(tok.upper()):
                typed = tok.upper()
                break
    return typed, (retriever.get_event(typed) if typed else None)


def run(eval_items, retriever, k=10):
    rows = []
    for item in eval_items:
        query = item["query"]
        expected = item.get("expected_id")
        in_scope = item.get("type") not in OUT_OF_SCOPE_TYPES and expected is not None

        exp_ev = retriever.index.by_chunk_id.get(expected) if expected else None
        expected_key = [exp_ev["kind"], exp_ev["code"]] if exp_ev else None
        common = {
            "query": query, "type": item.get("type"), "expected_id": expected,
            "expected_key": expected_key, "in_scope": in_scope,
        }

        typed, ev = resolve_typed(query, retriever)

        if typed:                       # exact lookup or unknown code: no search
            path = "exact_lookup" if ev else "unknown_code"
            hit = bool(ev) and in_scope and ev["chunk_id"] == expected
            rows.append({
                **common,
                "rank": 1 if hit else None, "hit@1": hit, "hit@3": hit,
                "hit@5": hit, "rr": 1.0 if hit else 0.0,
                "top1_score": None, "top2_score": None, "gap_ratio": None,
                "top5_ids": [ev["chunk_id"]] if ev else [],
                "path": path, "timing": {"path": path},
            })
            continue

        cands = retriever.search_events(query, k=k)
        ids = [c["chunk_id"] for c in cands]
        rank = next((i + 1 for i, cid in enumerate(ids) if cid == expected), None) if in_scope else None
        top1 = cands[0]["score"] if cands else None
        top2 = cands[1]["score"] if len(cands) > 1 else None
        gap_ratio = (top1 - top2) / max(top1, 1e-9) if (top1 is not None and top2 is not None) else None
        rows.append({
            **common,
            "rank": rank, "hit@1": rank == 1,
            "hit@3": rank is not None and rank <= 3,
            "hit@5": rank is not None and rank <= 5,
            "rr": (1.0 / rank) if rank else 0.0,
            "top1_score": top1, "top2_score": top2, "gap_ratio": gap_ratio,
            "top5_ids": ids[:5],
            "path": "search",
            "cands": [
                {"chunk_id": c["chunk_id"], "kind": c["kind"], "code": c["code"], "score": c["score"]}
                for c in cands
            ],
            "timing": dict(retriever.last_timing),
        })
    return rows


def summarize(rows):
    rows = [r for r in rows if r.get("in_scope", True)]     # out_of_scope has no hit@k
    n = len(rows)
    if n == 0:
        return {"n": 0, "hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "mrr": 0.0}
    return {
        "n": n,
        "hit@1": sum(r["hit@1"] for r in rows) / n,
        "hit@3": sum(r["hit@3"] for r in rows) / n,
        "hit@5": sum(r["hit@5"] for r in rows) / n,
        "mrr": sum(r["rr"] for r in rows) / n,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RealRetriever (the agent's actual pipeline).")
    parser.add_argument("--eval-set", default="data/eval/eval_tune.json")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--out", default="data/eval/results_tune.json")
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder pass")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    eval_items = load_eval_set(args.eval_set)
    print(f"Loaded {len(eval_items)} eval questions from {args.eval_set}")

    retriever = RealRetriever(chunks_path=args.chunks, use_reranker=not args.no_rerank)
    rows = run(eval_items, retriever, k=args.k)

    overall = summarize(rows)
    print(f"\noverall (in-scope only): hit@1={overall['hit@1']:.3f} hit@3={overall['hit@3']:.3f} "
          f"hit@5={overall['hit@5']:.3f} mrr={overall['mrr']:.3f}  (n={overall['n']})")

    search_rows = [r for r in rows if r["path"] == "search" and r["in_scope"]]
    n_lookup = sum(r["in_scope"] and r["path"] != "search" for r in rows)
    if search_rows:
        s_ = summarize(search_rows)
        print(f"search path only:       hit@1={s_['hit@1']:.3f} hit@3={s_['hit@3']:.3f} "
              f"hit@5={s_['hit@5']:.3f} mrr={s_['mrr']:.3f}  (n={s_['n']})")
    print(f"({n_lookup} in-scope queries were resolved by exact code lookup and never touch "
          f"search, so 'overall' flatters the retriever. Quote the search-path line.)")

    by_type = {}
    for r in rows:
        if r["in_scope"]:
            by_type.setdefault(r["type"] or "untyped", []).append(r)
    print(f"\n{'type':<15} {'n':>4} {'hit@1':>8} {'hit@3':>8} {'hit@5':>8} {'mrr':>8}")
    for t, trows in by_type.items():
        s = summarize(trows)
        print(f"{t:<15} {s['n']:>4} {s['hit@1']:>8.3f} {s['hit@3']:>8.3f} {s['hit@5']:>8.3f} {s['mrr']:>8.3f}")

    n_oos = sum(not r["in_scope"] for r in rows)
    if n_oos:
        print(f"\n{n_oos} out-of-scope queries (incl. parameter lookups) not counted above; "
              f"tune_thresholds.py scores them.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"overall": overall, "rows": rows}, f, indent=2)
    print(f"\nPer-question detail (incl. candidates for threshold replay) written to {args.out}")