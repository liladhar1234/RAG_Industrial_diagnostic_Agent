# Agent Confidence Threshold Tuning

## Eval set (tune split)
46 queries reach the confidence check (31 in-scope, 15 out-of-scope).
11 queries never reach it (`exact_lookup` - resolved by exact code match,
bypassing the confidence gate entirely).

## Before / after

| | current (0.15 / 0.0) | chosen (0.1 / 0.4) |
|---|---|---|
| confident answers correct | 58% | **95%** (21 of 22) |
| confident-and-wrong (of all queries) | 28.3% | **2.2%** (1 of 46) |
| out-of-scope queries rejected | 20% | **93%** (14 of 15) |
| in-scope queries answered (coverage) | 61% | 68% |

**Chosen: `MIN_GAP_RATIO = 0.1`, `MIN_TOP_SCORE = 0.4`.** Verified by
`tune_thresholds.py`'s own plateau computation (gap 0.075-0.125, top
0.30-0.55), not hand-estimated - 0.1/0.4 sits in the middle of both ranges.
"If a threshold is off, it should err upward" - a false abstain only costs
coverage, a false confident answer costs correctness.

## The 10 abstentions, categorized

**5 - correct twin-ambiguity abstentions.** A genuine warning/fault twin
exists for the same underlying issue (`A581` vs its fault, `A6D1` vs fault
`65A1`, `A2B3`/`A2B4` vs their faults, `3220` vs its warning twin).
Abstaining is the right call - `prepare_question` lists both options so the
agent asks "fault or warning?" instead of guessing. This is the intended
behavior, not a gap.

**3 - reranker ignored the fault/warning cue.** "Tripped, braking resistor
shorted" → `A792` instead of fault `7184`; "brake chopper IGBT got too hot"
→ `A79C` instead of fault `7192`; the brake-chopper-transistor query → `A793`
instead of `A79C`. Small gaps meant abstaining still caught these safely,
but the underlying cross-encoder didn't reliably use "tripped" vs "warning"
as a fault/warning signal.

**2 - genuine vocabulary misses.** "Control board can't talk to the power
stage" and "fieldbus adapter... not supported" scored 0.26 and 0.20 - the
manual says "power unit" and "option module" instead. These fail safe
(abstain, ask a question) rather than answering wrong, but they're real
recall gaps worth noting.

## Known, accepted limitation

`"how do I wire the emergency stop"` scores 0.66 and passes confidently,
returning `warning AFE1` - a genuine out-of-scope confident-wrong (this
query isn't really asking about a fault code). Raising `MIN_TOP_SCORE` to
0.7 would fix this one case but would reject 4 correct in-scope answers
(`AFF8`, `64B3`, `A4A0`, `A6A6`, scoring 0.60-0.66) - a bad trade. Documented
as an accepted limitation rather than tuned away.

## Two cautions for the report

1. **The fault/warning disambiguation queries partly bake in their own
   answer.** Words like "tripped" (implying fault) or "warning" were
   written into the query text deliberately to test disambiguation. A rule
   that specifically boosted on those words would score artificially well
   here; real technicians may not phrase things that consistently. Any
   future cue-based rule must be disclosed as trained/tested on
   self-referential phrasing.
2. **Held-out stays sealed** until the retrieval pipeline is frozen. These
   thresholds are fit to the current score distribution; if later ablations
   (router logic, weighted RRF, raw-logit gap) change that distribution,
   re-tune on the TUNE split first, and only run held-out once, at the end,
   on the final frozen pipeline.

## Status

Thresholds tuned and verified against the tool's own plateau computation
(not hand-estimated). Pasted into `agent/nodes.py`. Manual CLI check
pending to confirm no regression on the original ambiguous fieldbus
symptom test case.
