"""
eval_ragas.py

Phase 3: Ragas generation-quality metrics (faithfulness, answer_relevancy,
context_precision, context_recall) over the agent's ACTUAL generated
answers - not retrieval alone. Complements eval_agent.py's deterministic
hit@k/MRR, which only checks whether the right chunk was retrieved, not
whether the final answer text is faithful to it.

CAVEAT, worth repeating in your report: the judge here is a local
qwen2.5:3b via Ollama, not a frontier model. Ragas is typically validated
against GPT-4-class judges; a 3B model is a meaningfully weaker judge.
Treat these numbers as a rough internal signal for catching regressions,
not a benchmark result to defend under scrutiny without this caveat.

Scope: only eval items that resolve to a confident, single-turn answer
(no clarifying question needed) are scored - Ragas expects a finished
question/answer pair, not a graph paused mid-dialogue. Items that
correctly abstain are skipped and counted separately; abstaining is
correct behavior there, not a generation failure - eval_agent.py's
coverage metric already covers that decision.

ground_truth is the LABELED fault/warning's own cause+action text (from
expected_id), not whatever fault the agent actually picked - a wrong
pick is scored against the correct answer, as it should be, not against
itself.

Setup: needs an Ollama LangChain chat wrapper. Try running as-is first;
if you get an ImportError for langchain_ollama or langchain_community,
run ONE of:
    pip install langchain-ollama --break-system-packages
    pip install langchain-community --break-system-packages
(inside the container - see package_management notes if running elsewhere)

Usage (inside Docker):
    docker compose exec dev python src/retrieval/eval_ragas.py \\
        --eval-set data/eval/eval_tune.json
    # to also use the LLM-phrased (not just deterministic template) answers:
    docker compose exec dev python src/retrieval/eval_ragas.py \\
        --eval-set data/eval/eval_tune.json --llm
"""

import argparse
import json
import os
import uuid

from sentence_transformers import SentenceTransformer
from langgraph.types import Command

from agent.graph import build_graph
from agent.real_retriever import RealRetriever

OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


class LocalEmbeddings:
    """Minimal ragas BaseRagasEmbeddings-compatible wrapper around the
    project's existing bge-small-en-v1.5 model, so answer_relevancy (the
    one Ragas metric that needs embeddings) doesn't require a separate
    OpenAI embeddings key or a new dependency."""

    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self._st = SentenceTransformer(model_name)  # not ".model": ragas logs that attribute and needs a str

    def embed_query(self, text):
        return self._st.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts):
        return self._st.encode(texts, normalize_embeddings=True).tolist()

    async def aembed_query(self, text):
        return self.embed_query(text)

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)


def build_judge_llm():
    """Wrap local Ollama (qwen2.5:3b) as the Ragas judge LLM."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.chat_models import ChatOllama

    from ragas.llms import LangchainLLMWrapper

    chat = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0, num_ctx=8192)
    return LangchainLLMWrapper(chat)


def build_judge_embeddings():
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # LangchainEmbeddingsWrapper expects a langchain-style embeddings
    # object; LocalEmbeddings above implements the same method names by
    # hand rather than depending on an extra langchain integration package.
    return LangchainEmbeddingsWrapper(LocalEmbeddings())


def summarize_scores(result):
    """Mean of the non-NaN scores per metric, plus how many rows the judge
    failed on. A NaN means the judge produced no usable output (timeout or
    malformed JSON), which is common with a small local judge, so it must be
    reported rather than silently averaged away."""
    import math

    summary = {}
    for name in result.scores[0].keys():
        vals = [s[name] for s in result.scores]
        ok = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        summary[name] = {
            "mean": (sum(ok) / len(ok)) if ok else None,
            "n_scored": len(ok),
            "n_failed": len(vals) - len(ok),
        }
    return summary


def print_summary(summary):
    print("\nRagas results (mean over the rows the judge actually scored):")
    for name, s in summary.items():
        mean = "n/a" if s["mean"] is None else format(s["mean"], ".3f")
        print(f"  {name:20s} {mean:>6}   scored={s['n_scored']}  judge_failed={s['n_failed']}")


def ground_truth_for(fault):
    return f"{fault['name']}. Cause: {fault['cause']} What to do: {fault['action']}"


def parts_result_to_text(parts_result):
    """Renders the DB tool's result as text so the judge can see the
    spare-parts facts the answer draws on. Without this, faithfulness
    penalizes the answer for stating stock levels/prices that are real
    (came from a verified DB call) but absent from the manual-chunk-only
    context, which mislabels a correct DB fact as unsupported."""
    if not parts_result or not parts_result.get("mapped"):
        return "No replacement part is mapped to this code in the parts database."
    lines = ["Spare parts from the database:"]
    for p in parts_result.get("parts", []):
        if not p.get("found"):
            continue
        part = p["part"]
        stock = "in stock" if p.get("in_stock") else "out of stock"
        lines.append(f"- {part['part_number']} ({part['description']}, ${part['unit_price']:.2f}): {stock}")
    return "\n".join(lines)


def build_rows(items, retriever, graph):
    rows = []
    skipped_ambiguous = skipped_no_fault = skipped_no_label = 0
    picked_labeled = 0
    for item in items:
        expected_id = item.get("expected_id")
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        out = graph.invoke({"user_turns": [item["query"]], "clarify_count": 0}, cfg)

        if graph.get_state(cfg).next:  # paused on a clarifying question
            skipped_ambiguous += 1
            continue

        fault = out.get("fault")
        if not fault:  # unknown_code / give_up / unverified
            skipped_no_fault += 1
            continue

        # Ground truth from the LABEL, not from whatever the agent picked -
        # a wrong pick must be scored against the correct fault, not itself.
        label_event = retriever.index.by_chunk_id.get(expected_id) if expected_id else None
        if label_event is None:
            skipped_no_label += 1
            continue

        if fault.get("chunk_id") == expected_id:
            picked_labeled += 1

        # Real contexts: the retrieved candidates (what retrieval actually
        # surfaced, in rank order) plus the DB result as text - this is
        # what the answer was actually built from, not just the one chosen
        # chunk in isolation.
        contexts = []
        seen_ids = set()
        for c in (out.get("candidates") or [])[:3]:
            cid = c.get("chunk_id")
            text = retriever.text_by_id.get(cid, "")
            if text and cid not in seen_ids:
                contexts.append(text[:2000])
                seen_ids.add(cid)
        chosen_text = retriever.text_by_id.get(fault.get("chunk_id"), "")
        if chosen_text and fault.get("chunk_id") not in seen_ids:
            contexts.insert(0, chosen_text[:2000])
            seen_ids.add(fault.get("chunk_id"))
        contexts.append(parts_result_to_text(out.get("parts_result")))

        rows.append({
            "question": item["query"],
            "answer": out["answer"],
            "contexts": contexts,
            "ground_truth": ground_truth_for(label_event),
            "expected_id": expected_id,
            "chosen_id": fault.get("chunk_id"),
        })
    return rows, skipped_ambiguous, skipped_no_fault, skipped_no_label, picked_labeled


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ragas generation-quality eval over real agent answers.")
    ap.add_argument("--eval-set", default="data/eval/eval_tune.json")
    ap.add_argument("--llm", action="store_true", help="use qwen2.5:3b to phrase the agent's answers too")
    ap.add_argument("--out", default="data/eval/ragas_results.json")
    ap.add_argument("--timeout", type=int, default=900, help="seconds allowed per judge job")
    ap.add_argument("--workers", type=int, default=1, help="parallel judge jobs (keep 1 for local Ollama)")
    args = ap.parse_args()

    items = json.load(open(args.eval_set))
    items = [i for i in items if i.get("type") not in ("out_of_scope", "param_lookup_out_of_scope")]
    print(f"{len(items)} in-scope eval items loaded from {args.eval_set}")

    retriever = RealRetriever()
    graph = build_graph(retriever, use_llm=args.llm)

    rows, skipped_ambiguous, skipped_no_fault, skipped_no_label, picked_labeled = build_rows(items, retriever, graph)
    print(f"{len(rows)} scored, {skipped_ambiguous} skipped (agent correctly asked to clarify), "
          f"{skipped_no_fault} skipped (unknown_code / give_up / unverified), "
          f"{skipped_no_label} skipped (label wasn't a fault/warning chunk)")
    if rows:
        print(f"picked the labeled fault in {picked_labeled}/{len(rows)} rows")
    if not rows:
        raise SystemExit("nothing to score")

    from datasets import Dataset
    import ragas_compat
    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    judge_llm = build_judge_llm()
    judge_embeddings = build_judge_embeddings()

    ds = Dataset.from_list(rows)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(timeout=args.timeout, max_workers=args.workers, max_retries=2),
    )
    print(f"\nJudge: local {OLLAMA_MODEL} via Ollama - treat as a rough signal, "
          f"not a frontier-model-grade benchmark result.\n")
    print(result)

    with open(args.out, "w") as f:
        summary = summarize_scores(result)
        json.dump({"summary": summary, "rows": rows, "per_row_scores": result.scores}, f, indent=2)
    print_summary(summary)
    print(f"\nWritten to {args.out}")