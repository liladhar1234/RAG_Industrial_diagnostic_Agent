"""
make_eval_candidates.py

Stage 1: builds a WORKSHEET of eval-question candidates from chunks.jsonl,
so growing the eval set from 18 to 60-100 questions is mostly filling in
blanks rather than starting from an empty file.

What it emits (data/eval/eval_candidates.json, a JSON list):

  exact_code      status "ready". Query is generated; expected_id is the
                  canonical event chunk for that code (the same chunk
                  EventIndex.get() returns, so the label matches what the
                  agent's exact-lookup path would return).
  symptom         status "todo". `query` is EMPTY. You write it, using the
                  `hint` (name / cause / page) to see what the fault is.
  hard_negative   status "todo". `query` is EMPTY. Pairs of events whose
                  names share many words, found automatically. Write a query
                  for the target that a careless system could confuse with
                  `confusable_with`.
  out_of_scope    status "ready". Starter queries the agent should NOT answer
                  confidently. expected_id is null.

How to write the "todo" queries (this is what makes the eval honest):
  - Describe what a technician SEES or REPORTS ("drive stops and the fan
    is silent"), not the manual's wording. If your query reuses the fault's
    name, retrieval is trivially easy and hit@k is inflated.
    split_eval.py warns when a query overlaps the name too much.
  - Delete any stub you can't write a natural query for.
  - "part_number" is deliberately not a type: parts live in SQLite, not in
    the retrieval index, so there is no chunk to retrieve for them.

Usage (inside Docker):
    docker compose exec dev python src/retrieval/make_eval_candidates.py \\
        --existing data/eval/eval_set.json \\
        --out data/eval/eval_candidates.json
"""

import argparse
import json
import os
import random
import re

from agent.event_index import EventIndex

STOP = {
    "the", "and", "for", "with", "not", "too", "fault", "warning", "error",
    "detected", "failure", "from", "has", "was", "are", "drive", "unit",
}

EXACT_TEMPLATES = [
    "{kind} {code}",
    "the display shows {kind} {code}",
    "what does {kind} {code} mean",
    "{code}",
]

# (query, note). "near_domain" queries sound industrial but are not drive
# faults; before keeping one, sanity-check that no fault in the manual
# actually matches it, otherwise the label "should not be confident" is wrong.
OOS_STARTERS = [
    ("the coffee machine is broken", "unrelated"),
    ("how do I reset my wifi password", "unrelated"),
    ("what is the weather like today", "unrelated"),
    ("my printer keeps jamming paper", "unrelated"),
    ("the conveyor belt is misaligned", "near_domain"),
    ("motor bearings are making a grinding noise", "near_domain"),
    ("the pump is leaking oil at the seal", "near_domain"),
    ("the elevator door will not close", "near_domain"),
]


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def name_tokens(ev):
    return {
        t for t in re.findall(r"[a-z0-9]+", ev["name"].lower())
        if len(t) > 2 and t not in STOP
    }


def confusable_pairs(events, min_jaccard, limit):
    """Pairs of different-code events whose names overlap heavily."""
    toks = [(e, name_tokens(e)) for e in events]
    pairs = []
    for i in range(len(toks)):
        a, ta = toks[i]
        if not ta:
            continue
        for j in range(i + 1, len(toks)):
            b, tb = toks[j]
            if not tb or a["code"] == b["code"]:
                continue
            jac = len(ta & tb) / len(ta | tb)
            if jac >= min_jaccard:
                pairs.append((jac, a, b))
    pairs.sort(key=lambda p: -p[0])
    return pairs[:limit]


def hint_of(ev, cause_chars=160):
    return {
        "code": ev["code"],
        "kind": ev["kind"],
        "name": ev["name"],
        "cause": ev["cause"][:cause_chars],
        "page": ev["page"],
    }


def main():
    ap = argparse.ArgumentParser(description="Build an eval-question worksheet from chunks.jsonl.")
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--existing", default="data/eval/eval_set.json",
                    help="current eval set; events already covered there are skipped")
    ap.add_argument("--out", default="data/eval/eval_candidates.json")
    ap.add_argument("--n-exact", type=int, default=15)
    ap.add_argument("--n-symptom", type=int, default=35)
    ap.add_argument("--n-hard", type=int, default=12)
    ap.add_argument("--min-jaccard", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    index = EventIndex(load_jsonl(args.chunks))
    events = sorted(index.by_code.values(), key=lambda e: (e["code"], e["kind"]))

    covered = set()
    if args.existing and os.path.exists(args.existing):
        with open(args.existing) as f:
            covered = {i.get("expected_id") for i in json.load(f)}

    rng = random.Random(args.seed)
    pool = [e for e in events if e["chunk_id"] not in covered]
    rng.shuffle(pool)
    exact_evs = pool[: args.n_exact]
    sym_evs = pool[args.n_exact: args.n_exact + args.n_symptom]

    out = []

    for k, ev in enumerate(exact_evs):
        tpl = EXACT_TEMPLATES[k % len(EXACT_TEMPLATES)]
        out.append({
            "query": tpl.format(kind=ev["kind"], code=ev["code"]),
            "expected_id": ev["chunk_id"],
            "type": "exact_code",
            "status": "ready",
        })

    for ev in sym_evs:
        out.append({
            "query": "",
            "expected_id": ev["chunk_id"],
            "type": "symptom",
            "status": "todo",
            "hint": hint_of(ev),
        })

    n_pairs = 0
    for jac, a, b in confusable_pairs(events, args.min_jaccard, args.n_hard * 3):
        if n_pairs >= args.n_hard:
            break
        target, other = (a, b) if rng.random() < 0.5 else (b, a)
        if target["chunk_id"] in covered:
            continue
        out.append({
            "query": "",
            "expected_id": target["chunk_id"],
            "type": "hard_negative",
            "status": "todo",
            "hint": {
                "target": hint_of(target),
                "confusable_with": hint_of(other),
                "name_jaccard": round(jac, 2),
            },
        })
        n_pairs += 1

    for q, note in OOS_STARTERS:
        out.append({
            "query": q,
            "expected_id": None,
            "type": "out_of_scope",
            "status": "ready",
            "note": note,
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    by_type = {}
    for it in out:
        by_type.setdefault((it["type"], it["status"]), 0)
        by_type[(it["type"], it["status"])] += 1
    print(f"{len(events)} distinct event codes; {len(covered)} chunk ids already in {args.existing}")
    for (t, s), n in sorted(by_type.items()):
        print(f"  {t:<14} {s:<6} {n}")
    if index.collisions:
        print(f"\n{len(index.collisions)} code(s) appear on more than one event chunk "
              f"(fault+warning collision or parent+aux chunks): {sorted(set(index.collisions))[:10]}")
    print(f"\nWrote {args.out}. Fill in every empty 'query' (delete stubs you can't), "
          f"then run split_eval.py.")


if __name__ == "__main__":
    main()
