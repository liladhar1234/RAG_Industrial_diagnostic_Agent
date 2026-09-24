"""
apply_drafts.py

Fills the empty `query` fields in data/eval/eval_candidates.json from
data/eval/drafted_queries.json, matching stubs on (expected_id, type).

- Only stubs whose query is EMPTY are touched, so queries you have already
  written or edited are never overwritten. Safe to re-run.
- Filled stubs get status "drafted". Deliberately skipped stubs keep an empty
  query and get a `skip_reason` (split_eval.py ignores empty stubs).
- A backup of the original is written next to it (.bak).

Review workflow: edit drafted_queries.json BEFORE applying, or edit the query
directly in eval_candidates.json afterwards. Delete a whole stub to drop it.

Usage (standard library only):
    python src/retrieval/apply_drafts.py
"""

import argparse
import json
import shutil


def main():
    ap = argparse.ArgumentParser(description="Apply drafted queries to the eval worksheet.")
    ap.add_argument("--candidates", default="data/eval/eval_candidates.json")
    ap.add_argument("--drafts", default="data/eval/drafted_queries.json")
    args = ap.parse_args()

    with open(args.candidates) as f:
        stubs = json.load(f)
    with open(args.drafts) as f:
        drafts = json.load(f)

    fill = {(d["expected_id"], d["type"]): d["query"] for d in drafts["fill"]}
    skip = {(d["expected_id"], d["type"]): d["reason"] for d in drafts["skip"]}

    shutil.copyfile(args.candidates, args.candidates + ".bak")

    filled = skipped = 0
    used_fill, used_skip = set(), set()
    untouched = []
    for s in stubs:
        if (s.get("query") or "").strip():
            continue                                    # already written: leave alone
        key = (s.get("expected_id"), s.get("type"))
        if key in fill:
            s["query"] = fill[key]
            s["status"] = "drafted"
            used_fill.add(key)
            filled += 1
        elif key in skip:
            s["skip_reason"] = skip[key]
            used_skip.add(key)
            skipped += 1
        else:
            untouched.append(key)

    with open(args.candidates, "w") as f:
        json.dump(stubs, f, indent=2)

    print(f"filled {filled} stubs, marked {skipped} as skipped, {len(untouched)} still empty")
    for key in untouched:
        print("  still empty (no draft):", key)
    for key in sorted(set(fill) - used_fill):
        print("  draft matched no empty stub (already filled, or id not in worksheet):", key)
    print(f"backup: {args.candidates}.bak")


if __name__ == "__main__":
    main()
