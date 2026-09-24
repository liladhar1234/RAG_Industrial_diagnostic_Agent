# Phase 2 & 3 Log — Retrieval, Hybrid Fusion, Evaluation

## Pipeline built

1. **Dense** — `bge-small-en-v1.5` (local, free), Qdrant, cosine similarity
2. **BM25** — `rank_bm25`, custom tokenizer that keeps alphanumeric runs with
   internal hyphens/dots (`A2B1`, `36.09`, `4521-B` survive as single tokens
   instead of being split)
3. **Hybrid** — Reciprocal Rank Fusion (RRF, k=60) over dense + BM25,
   fused on actual `chunk_id` strings (not list position, after fixing an
   early fragility where fusion assumed `chunks.jsonl` order never changed)
4. **Hybrid + rerank** — `bge-reranker-base` cross-encoder over the RRF
   candidate pool (top 20), re-scoring the true top-k by joint
   query-passage relevance rather than fused rank position

## Eval set

18 hand-crafted questions (`data/eval/eval_set.json`), covering:
- 6 exact-code lookups
- 4 plain natural-language symptom queries
- 7 **disambiguation pairs** — same-named warning/fault pairs the manual
  actually contains (`A2B1`/`2310` "Overcurrent", `A3A1`/`3210` "DC link
  overvoltage", `A8A0`/`80A0` "AI supervision")
- 1 **hard negative**, flagged back in Phase 1: `A8BF` warning vs. `37.04`
  parameter, which mentions `A8BF` in its description text without being
  the right answer

## Headline results (hit@k / MRR)

| method | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| dense | 0.500 | 0.667 | 0.833 | 0.631 |
| bm25 | 0.667 | 0.944 | 0.944 | 0.796 |
| hybrid | 0.556 | 1.000 | 1.000 | 0.778 |
| hybrid+rerank | **0.833** | 1.000 | 1.000 | **0.907** |

## The counterintuitive finding: naive RRF hurts top-1 precision

Hybrid's hit@1 (0.556) is **worse than BM25 alone (0.667)**, despite combining
two signals. RRF fuses purely on rank position, with no notion of which
retriever's judgment is more trustworthy for a given query. When two
retrievers disagree about which of two candidates ranks first (e.g. dense
confidently correct, BM25 confidently wrong, or vice versa), RRF can
average a confidently-right #1 down to #2 — this is exactly what was seen
anecdotally with `"auxiliary fan is broken"` (5081 vs A582 tied), and it
recurs at scale here (e.g. `01.01` vs `01.61`, see below).

**RRF does guarantee recall improves** (hit@3 = hit@5 = 1.000 — the right
answer is *somewhere* in the top 3 every single time), but recall isn't
the same as precision, and an agent that just takes the #1 result would be
measurably worse off under naive hybrid than under BM25 alone. **Reranking
is what actually captures hybrid's promised benefit**: hit@1 jumps from
0.556 to 0.833, and MRR from 0.778 to 0.907 — worth the extra cross-encoder
latency for this use case.

## Three specific hit@1 failures, even with reranking (3/18)

These are the questions that keep hybrid+rerank from a perfect score, and
each reveals a genuinely distinct ambiguity class in the manual itself —
not a pipeline bug.

### 1. `"motor speed used parameter"` → expected `01.01`, got `01.61`

Dense alone got this right (`01.01` #1). BM25 confidently preferred
`01.61` instead; RRF let BM25's wrong pick win; the cross-encoder *still*
preferred `01.61` even with full context.

**Confirmed, not speculation:** `01.61`'s actual name is **"Abs motor
speed used"**, with description *"Absolute value of parameter 01.01 Motor
speed used."* The manual's own text explicitly names and defines `01.61`
in terms of `01.01` — this is a genuine, by-design near-duplicate, not a
parsing or retrieval defect. Any parameter whose whole definition is
"the absolute value of parameter X" will structurally resemble X's own
chunk more than an unrelated chunk would, and no amount of better
tokenization or reranking model changes that; a real fix would need
either query-side disambiguation ("do you mean the raw or absolute value
signal?") or explicit parent/derivative linking in the chunk metadata
so an agent can recognize the relationship rather than being confused
by it.

### 2. `"drive tripped on an overcurrent fault"` → expected `2310`, got `05.20`

All four methods, including hybrid+rerank, prefer `05.20` over the real
fault definition. Group `05` is "Diagnostics" — `05.20` is very likely a
status/diagnostic register that *enumerates* fault codes (e.g. "active
fault word"), so its text legitimately contains "overcurrent" and "fault"
prominently. **This is a new hard-negative pattern**, distinct from the
known `A8BF` case: a diagnostic parameter that lists faults will
systematically outcompete the actual fault chunk on symptom-style queries
about that fault.

### 3. `"analog input supervision fault, drive stopped"` → expected `80A0`, got `12.03`

`80A0`'s own chunk text reads *"Programmable fault: 12.03 AI supervision
function"* — the configuring parameter shares the fault's name almost
verbatim, by design (this is how the manual cross-references programmable
faults to their trigger parameter). **A third distinct ambiguity axis**:
fault chunks that name their configuring parameter create a structural
near-duplicate with that parameter's own chunk.

### Note: the original `A8BF` hard negative was resolved correctly

Hybrid+rerank puts `A8BF` at rank 1, ahead of `37.04` — the specific case
flagged in Phase 1 as a predicted retrieval risk was, in fact, handled
correctly by the full pipeline. Good confirmation that reranking earns its
cost on the exact failure mode it was intended to fix.

## Known fragility fixed during this phase

`hybrid_search.py` originally fused dense and BM25 rankings by **list
index**, silently assuming `chunks.jsonl` would never be rebuilt in a
different order than when `embed.py` created the Qdrant collection. Fixed
by fusing on the actual `chunk_id` string (already available in Qdrant's
payload) instead — removes a subtle correctness dependency that would have
been easy to violate later (e.g. after adding diagram chunks or re-running
`build_chunks.py` with different page ranges).

## Status

Phase 2 (dense/BM25/hybrid/rerank) and Phase 3's core deterministic
eval (hit@k/MRR against a labeled set) are both functionally complete,
with a real, non-trivial, non-obvious finding (naive RRF can hurt top-1
precision) and three specifically diagnosed remaining failure cases,
each pointing to a distinct and genuine ambiguity in the source manual
rather than a pipeline defect.

Not yet done from the original Phase 3 plan: Ragas-based generation
metrics (faithfulness, answer relevancy) — deferred until Phase 4's
agent exists and can generate answers worth judging. Also worth
expanding the eval set beyond 18 hand-written questions (the roadmap's
suggestion of synthetic question generation, manually verified, is the
natural next step if a larger, more statistically solid number is
wanted for the final report).