"""
split_eval.py

Stage 1: merge, validate, and split the eval set into TUNE and HELD-OUT.

Reads one or more JSON lists (your existing eval_set.json plus the filled-in
eval_candidates.json), drops stubs whose `query` is still empty, validates
what's left, and writes:

    data/eval/eval_full.json      everything
    data/eval/eval_tune.json      use freely: tuning thresholds, ablations
    data/eval/eval_heldout.json   do NOT look at until the very end

Item schema:
    {"query": str, "expected_id": <chunk id> | null, "type": str}
    types: exact_code | symptom | symptom_disambiguation | hard_negative
           | out_of_scope | param_lookup_out_of_scope
    out_of_scope items must have expected_id null. param_lookup_out_of_scope
    items keep their parameter chunk id (eval.py scores them over all chunks;
    eval_agent.py treats them as questions the agent must not answer).

The split is decided by a HASH OF THE QUERY TEXT, not by shuffling. That
means adding questions later never moves an existing question from tune to
held-out (which would contaminate the held-out set). The trade-off is that
the split is only approximately stratified, so the type x split table below
is worth a glance.

Usage (inside Docker):
    docker compose exec dev python src/retrieval/split_eval.py \\
        --in data/eval/eval_set.json data/eval/eval_candidates.json
"""

import argparse
import hashlib
import json
import os
import re
import sys

from agent.event_index import EventIndex
from make_eval_candidates import load_jsonl, name_tokens

ALLOWED_TYPES = {
    "exact_code", "symptom", "symptom_disambiguation", "hard_negative",
    "out_of_scope",             # expected_id null: nothing in the manual answers it
    "param_lookup_out_of_scope",  # expected_id = a parameter chunk: valid for eval.py,
                                  # but out of scope for the fault-only agent
}
# symptom_disambiguation is not checked: those queries must name the event to
# separate fault from warning, so naming it is the point, not leakage.
LEAK_CHECKED = {"symptom", "hard_negative"}
MIN_NAME_TOKENS = 3      # names shorter than this ("Overcurrent") match too easily
DROP_KEYS = {"status", "hint"}          # worksheet-only fields


def norm(q):
    return " ".join(q.lower().split())


def bucket(query):
    return int(hashlib.sha1(norm(query).encode()).hexdigest()[:8], 16) % 100


def leakage(query, ev):
    nt = name_tokens(ev)
    if len(nt) < MIN_NAME_TOKENS:
        return 0.0
    qt = set(re.findall(r"[a-z0-9]+", query.lower()))
    return len(nt & qt) / len(nt)


def main():
    ap = argparse.ArgumentParser(description="Merge, validate and split the eval set.")
    ap.add_argument("--in", dest="inputs", nargs="+",
                    default=["data/eval/eval_set.json", "data/eval/eval_candidates.json",
                             "data/eval/hard_oos_queries.json"])
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--out-dir", default="data/eval")
    ap.add_argument("--heldout-pct", type=int, default=30)
    ap.add_argument("--leak-threshold", type=float, default=0.75,
                    help="warn if this share of the target's name words appear in the query")
    args = ap.parse_args()

    have_chunks = os.path.exists(args.chunks)
    all_ids, index = set(), None
    if have_chunks:
        chunks = load_jsonl(args.chunks)
        all_ids = {c["id"] for c in chunks}
        index = EventIndex(chunks)
    else:
        print(f"[warn] {args.chunks} not found: skipping expected_id and leakage checks")

    items, todo = [], 0
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"[warn] {path} not found, skipping")
            continue
        with open(path) as f:
            for it in json.load(f):
                if not (it.get("query") or "").strip():
                    todo += 1
                    continue
                items.append({k: v for k, v in it.items() if k not in DROP_KEYS})

    errors, warnings = [], []
    seen, kept, dupes = set(), [], 0
    for it in items:
        key = norm(it["query"])
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        kept.append(it)

        t, exp = it.get("type"), it.get("expected_id")
        label = f"{it['query']!r}"
        if t not in ALLOWED_TYPES:
            warnings.append(f"unknown type {t!r} on {label} (expected one of {sorted(ALLOWED_TYPES)})")
        if t == "out_of_scope":
            if exp is not None:
                errors.append(f"out_of_scope item must have expected_id null: {label}")
            continue
        if exp is None:
            errors.append(f"missing expected_id: {label}")
            continue
        if have_chunks and exp not in all_ids:
            errors.append(f"expected_id {exp!r} is not a chunk id: {label}")
            continue
        if index is not None and t in LEAK_CHECKED and exp in index.by_chunk_id:
            lk = leakage(it["query"], index.by_chunk_id[exp])
            if lk >= args.leak_threshold:
                warnings.append(f"possible leakage ({lk:.0%} of the fault name is in the query): {label}")

    if errors:
        print("ERRORS (nothing written):")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    tune, held = [], []
    for it in kept:
        (held if bucket(it["query"]) < args.heldout_pct else tune).append(it)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, data in (("eval_full", kept), ("eval_tune", tune), ("eval_heldout", held)):
        with open(os.path.join(args.out_dir, name + ".json"), "w") as f:
            json.dump(data, f, indent=2)

    print(f"{len(kept)} questions kept ({todo} empty stubs skipped, {dupes} duplicates dropped)")
    types = sorted({it.get("type") or "untyped" for it in kept})
    print(f"\n{'type':<15} {'tune':>6} {'heldout':>8}")
    for t in types:
        nt = sum((it.get("type") or "untyped") == t for it in tune)
        nh = sum((it.get("type") or "untyped") == t for it in held)
        flag = "   <- no held-out examples" if nh == 0 else ""
        print(f"{t:<15} {nt:>6} {nh:>8}{flag}")
    print(f"{'total':<15} {len(tune):>6} {len(held):>8}")

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print("  -", w)
    if len(kept) < 60:
        print(f"\n[note] {len(kept)} questions is below the 60-100 target; one question moves "
              f"hit@1 by {100 / max(len(kept), 1):.1f} points.")
    print(f"\nWrote eval_full/eval_tune/eval_heldout.json to {args.out_dir}. "
          f"Leave eval_heldout.json alone until the end.")


if __name__ == "__main__":
    main()