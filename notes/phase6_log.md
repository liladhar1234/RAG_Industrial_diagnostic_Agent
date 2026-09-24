### Attempted optimization: reduce rerank cost by narrowing candidate pool (reverted)

**Hypothesis:** rerank is ~97% of search-path latency; if fewer candidates
needed reranking, cost would drop roughly proportionally with no accuracy
loss.

**First attempt (pool_k 10 → 3, narrowing fetch+fusion+rerank together):**
A retroactive test sliced results_tune.json's stored `cands` field to its
first 3 entries and found hit@1 unchanged (0.839 at pool_k=3, 5, and 10).
Applied pool_k=3 live; hit@1 on the search path actually dropped to 0.774,
and hard_negative collapsed from 1.000 to 0.667. Root cause: `cands` in
results_tune.json is the reranker's own output order, not the raw RRF-fused
order - the retroactive test implicitly assumed a full-width fetch/fusion
pass, but pool_k=3 narrowed fetch and fusion too, denying RRF the width it
needed to surface hard-negative candidates in the first place.

**Second attempt (pool_k reverted to 10, added separate rerank_k=3 to
narrow only the reranker's input, not fetch/fusion width):** Live re-test
still regressed: search-path hit@1 fell further to 0.742, hard_negative
stayed at 0.667. Root cause, now correctly diagnosed: the stored `cands`
data is POST-rerank, so no retroactive test built from it can validate a
change to what the reranker is allowed to see - that information (the
pre-rerank RRF order) was never captured. Reranking exists specifically to
promote candidates RRF fused-ranked 4th-10th; narrowing the reranker's own
input to 3 removes exactly the candidates it exists to rescue.

**Reverted to pool_k=10, rerank_k=10** (functionally identical to the
original, pre-optimization pipeline). Rerank latency (p50 ~1.8s) is not
reduced by this line of attack. A real fix would need either a faster/
smaller reranker model (a genuine architecture change, not a parameter
tweak) or capturing pre-rerank RRF order in the eval harness to properly
validate any future pool-narrowing attempt before applying it live.

**Methodological lesson:** an offline ablation is only valid if it uses
the same intermediate data the live change will actually operate on.
Slicing a final, already-processed result list to approximate an earlier
pipeline stage's behavior is not equivalent to actually changing that
stage - this cost two live regressions to learn, both caught by re-running
eval_agent.py rather than trusting the offline number, which is exactly
why that verification step exists.