"""
hybrid_search.py

Phase 2, step 3: hybrid retrieval via Reciprocal Rank Fusion (RRF).

Combines the dense ranking (Qdrant, semantic) and the BM25 ranking
(exact lexical match) into one fused ranking. RRF is used instead of a
weighted score average because dense cosine scores and BM25 scores
live on completely different, incomparable scales - RRF sidesteps that
by fusing on RANK POSITION instead of raw score:

    rrf_score(chunk) = sum over retrievers r that returned this chunk of
                        1 / (k + rank_r(chunk))

where rank_r is the chunk's 1-indexed position in retriever r's ranked
list, and k=60 is the standard RRF constant (dampens the influence of
any single very-high rank).

Depends on chunks.jsonl and the Qdrant collection built by embed.py -
both must be built from the exact same chunks.jsonl (same order), since
this script relies on Qdrant point IDs matching chunk list indices.

Usage (inside Docker):
    docker compose exec dev python src/retrieval/hybrid_search.py \\
        --chunks data/processed/chunks.jsonl \\
        --collection acs580 \\
        --search "fault 5081"
"""

import argparse
import json
import os

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from bm25_search import tokenize  # reuse the same code-aware tokenizer

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
RRF_K = 60


def load_chunks(path):
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def dense_ranking(client, model, collection, query, top_k):
    query_vector = model.encode([query], normalize_embeddings=True)[0].tolist()
    response = client.query_points(collection_name=collection, query=query_vector, limit=top_k)
    return [hit.payload["chunk_id"] for hit in response.points]


def bm25_ranking(bm25, chunks, query, top_k):
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i]["id"] for i in ranked]


def rrf_fuse(rankings, k=RRF_K):
    """rankings: list of ranked chunk-id-lists, one per retriever.
    Returns {chunk_id: fused_score}, higher is better. Fusing on the
    chunk's actual id string (rather than a list-index position) means
    this doesn't depend on chunks.jsonl staying in the exact same order
    it was in when embed.py built the Qdrant collection."""
    fused = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):  # rank is 0-indexed here
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return fused


def hybrid_search(client, model, bm25, chunks, collection, query, top_k=5, candidate_k=20):
    dense_ids = dense_ranking(client, model, collection, query, candidate_k)
    bm25_ids = bm25_ranking(bm25, chunks, query, candidate_k)

    fused = rrf_fuse([dense_ids, bm25_ids])
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    id_to_chunk = {c["id"]: c for c in chunks}
    return [(id_to_chunk[cid], score) for cid, score in ranked]


def rerank(query, candidates, model_name="BAAI/bge-reranker-base", top_k=5):
    """
    Rerank a pool of RRF-fused candidates with a cross-encoder, which
    scores each (query, passage) pair jointly rather than comparing
    independently-computed embeddings/scores. This directly answers
    "how relevant is THIS chunk to THIS query" instead of averaging
    rank positions, which is exactly what RRF can't do - it's how the
    "auxiliary fan is broken" tie between 5081 (broken) and A582
    (missing) gets broken by actual meaning instead of rank symmetry.

    `candidates` is a list of (chunk, rrf_score) tuples, typically the
    larger candidate_k pool from hybrid_search(), not just its already-
    truncated top_k - reranking needs a wider pool to be worth doing.
    """
    cross_encoder = CrossEncoder(model_name)
    pairs = [[query, chunk["text"]] for chunk, _ in candidates]
    scores = cross_encoder.predict(pairs)

    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [(chunk, float(score)) for (chunk, _rrf), score in reranked][:top_k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid (dense + BM25, RRF-fused) search.")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--collection", default="acs580")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--search", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20,
                         help="how many results each retriever contributes before fusion")
    parser.add_argument("--rerank", action="store_true",
                         help="apply a cross-encoder rerank on top of the RRF-fused pool")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    client = QdrantClient(url=QDRANT_URL)
    model = SentenceTransformer(args.model)

    # When reranking, fetch a WIDER pool (candidate_k) from fusion first,
    # then let the cross-encoder pick the true top_k out of that pool -
    # reranking a pool of 5 defeats the purpose, since the correct
    # answer might have been fused-ranked 6th-20th.
    pool_k = args.candidate_k if args.rerank else args.top_k

    results = hybrid_search(
        client, model, bm25, chunks, args.collection, args.search,
        top_k=pool_k, candidate_k=args.candidate_k,
    )

    label = "hybrid+reranked" if args.rerank else "hybrid"
    if args.rerank:
        results = rerank(args.search, results, model_name=args.reranker_model, top_k=args.top_k)

    print(f"Top {args.top_k} {label} results for: {args.search!r}\n")
    for chunk, score in results:
        print(f"score={score:.5f}  {chunk['id']}")
        print(f"  {chunk['text'][:200].replace(chr(10), ' ')}...")
        print()