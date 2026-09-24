"""
eval.py

Phase 3: quantitative retrieval evaluation. Runs the labeled eval set
against all four Phase 2 retrieval methods (dense, BM25, hybrid,
hybrid+rerank) and reports hit@1/hit@3/hit@5 and MRR for each - the
headline numbers for the report's retrieval comparison table.

hit@k / MRR are DETERMINISTIC (did retrieval find the exact right
chunk id?), which is a stronger, more honest signal here than an
LLM-judged metric like Ragas's context_precision, since the eval set
already states exactly which chunk id is correct for each question.
Ragas-style generation metrics belong later, once Phase 4's agent can
generate answers worth judging.

Usage (inside Docker):
    docker compose exec dev python src/retrieval/eval.py \\
        --chunks data/processed/chunks.jsonl \\
        --eval-set data/eval/eval_set.json \\
        --collection acs580

    # skip the slow cross-encoder pass while iterating:
    docker compose exec dev python src/retrieval/eval.py --skip-rerank
"""

import argparse
import json
import os

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from bm25_search import tokenize
from bm25_search import search as bm25_search_fn
from hybrid_search import hybrid_search, rerank

MAX_K = 10  # depth checked for hit@k / MRR


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_eval_set(path):
    with open(path) as f:
        return json.load(f)


def dense_search_ids(client, model, collection, query, top_k):
    query_vector = model.encode([query], normalize_embeddings=True)[0].tolist()
    response = client.query_points(collection_name=collection, query=query_vector, limit=top_k)
    return [hit.payload["chunk_id"] for hit in response.points]


def bm25_search_ids(bm25, chunks, query, top_k):
    results = bm25_search_fn(bm25, chunks, query, top_k)
    return [c["id"] for c, _ in results]


def hybrid_search_ids(client, model, bm25, chunks, collection, query, top_k, candidate_k):
    results = hybrid_search(client, model, bm25, chunks, collection, query,
                             top_k=top_k, candidate_k=candidate_k)
    return [c["id"] for c, _ in results]


def hybrid_rerank_ids(client, model, bm25, chunks, collection, query, top_k, candidate_k, reranker_model):
    pool = hybrid_search(client, model, bm25, chunks, collection, query,
                          top_k=candidate_k, candidate_k=candidate_k)
    reranked = rerank(query, pool, model_name=reranker_model, top_k=top_k)
    return [c["id"] for c, _ in reranked]


def score_method(ranked_ids_per_query, expected_ids):
    """ranked_ids_per_query: list of ranked chunk-id-lists, one per eval
    question. Returns mean hit@1/hit@3/hit@5/MRR across all questions."""
    n = len(expected_ids)
    hit1 = hit3 = hit5 = 0
    rr_sum = 0.0
    for ranked_ids, expected in zip(ranked_ids_per_query, expected_ids):
        rank = None
        for i, cid in enumerate(ranked_ids):
            if cid == expected:
                rank = i + 1  # 1-indexed
                break
        if rank is not None:
            rr_sum += 1.0 / rank
            if rank <= 1:
                hit1 += 1
            if rank <= 3:
                hit3 += 1
            if rank <= 5:
                hit5 += 1
    return {"hit@1": hit1 / n, "hit@3": hit3 / n, "hit@5": hit5 / n, "mrr": rr_sum / n}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval methods against a labeled eval set.")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--eval-set", default="data/eval/eval_set.json")
    parser.add_argument("--collection", default="acs580")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--out", default="data/eval/eval_results.json")
    parser.add_argument("--skip-rerank", action="store_true",
                         help="skip hybrid+rerank - one cross-encoder pass per question, slow")
    args = parser.parse_args()

    chunks = load_jsonl(args.chunks)
    eval_items = load_eval_set(args.eval_set)
    queries = [item["query"] for item in eval_items]
    expected_ids = [item["expected_id"] for item in eval_items]

    print(f"Loaded {len(chunks)} chunks, {len(eval_items)} eval questions\n")

    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    model = SentenceTransformer(args.model)

    dense_ids = [dense_search_ids(client, model, args.collection, q, MAX_K) for q in queries]
    bm25_ids = [bm25_search_ids(bm25, chunks, q, MAX_K) for q in queries]
    hybrid_ids = [
        hybrid_search_ids(client, model, bm25, chunks, args.collection, q, MAX_K, args.candidate_k)
        for q in queries
    ]

    results_by_method = {
        "dense": score_method(dense_ids, expected_ids),
        "bm25": score_method(bm25_ids, expected_ids),
        "hybrid": score_method(hybrid_ids, expected_ids),
    }

    rerank_ids = None
    if not args.skip_rerank:
        rerank_ids = [
            hybrid_rerank_ids(client, model, bm25, chunks, args.collection, q, MAX_K,
                               args.candidate_k, args.reranker_model)
            for q in queries
        ]
        results_by_method["hybrid+rerank"] = score_method(rerank_ids, expected_ids)

    print(f"{'method':<15} {'hit@1':>8} {'hit@3':>8} {'hit@5':>8} {'mrr':>8}")
    for m, r in results_by_method.items():
        print(f"{m:<15} {r['hit@1']:>8.3f} {r['hit@3']:>8.3f} {r['hit@5']:>8.3f} {r['mrr']:>8.3f}")

    detail = []
    for i, item in enumerate(eval_items):
        row = {"query": item["query"], "expected_id": item["expected_id"], "type": item.get("type")}
        row["dense_top5"] = dense_ids[i][:5]
        row["bm25_top5"] = bm25_ids[i][:5]
        row["hybrid_top5"] = hybrid_ids[i][:5]
        if rerank_ids is not None:
            row["hybrid_rerank_top5"] = rerank_ids[i][:5]
        detail.append(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(detail, f, indent=2)
    print(f"\nPer-question detail written to {args.out}")
