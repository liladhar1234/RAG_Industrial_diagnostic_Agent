"""
embed.py

Phase 2, step 1: dense-only baseline.

Reads data/processed/chunks.jsonl (from build_chunks.py), embeds each
chunk's text with a local sentence-transformers model (bge-small-en by
default - free, runs on CPU, no API key needed), and upserts everything
into a Qdrant collection.

This is deliberately the simplest possible retrieval path - no BM25, no
fusion, no reranking. Phase 2's plan is to measure this in isolation
first, then add BM25, then hybrid, and compare all three.

Usage (inside Docker):
    docker compose exec dev python src/retrieval/embed.py \\
        --chunks data/processed/chunks.jsonl \\
        --collection acs580 \\
        --model BAAI/bge-small-en-v1.5

    docker compose exec dev python src/retrieval/embed.py --search "fault 5081"
"""

import argparse
import json
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def load_chunks(path):
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_collection(client, collection, dim):
    """(Re)create the collection with the right vector size. Recreating
    is fine here - Phase 2 is iterative, we expect to re-embed often as
    chunking/model choices change, and this is a small dataset (~1100
    chunks), so a full rebuild each time is cheap and avoids stale
    leftover points from earlier experiments.

    Uses exists-check + delete + create rather than the older
    recreate_collection() convenience method, since that's deprecated
    in newer qdrant-client versions (this project's client is 1.19.x
    against a pinned 1.12.5 server - see the version-mismatch warning
    printed at runtime, which is safe to ignore but worth noting)."""
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def embed_and_upsert(chunks, model, client, collection, batch_size=64):
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                            normalize_embeddings=True)

    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={"chunk_id": c["id"], "text": c["text"], **c["metadata"]},
        )
        for i, c in enumerate(chunks)
    ]

    client.upsert(collection_name=collection, points=points)
    return len(points)


def search(client, model, collection, query, top_k=5):
    query_vector = model.encode([query], normalize_embeddings=True)[0].tolist()
    response = client.query_points(
        collection_name=collection, query=query_vector, limit=top_k
    )
    return response.points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed chunks and upsert into Qdrant, or search.")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--collection", default="acs580")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--search", default=None,
                         help="if given, skip embedding and just search the existing collection")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    client = QdrantClient(url=QDRANT_URL)
    model = SentenceTransformer(args.model)

    if args.search:
        hits = search(client, model, args.collection, args.search, args.top_k)
        print(f"Top {args.top_k} results for: {args.search!r}\n")
        for h in hits:
            print(f"score={h.score:.4f}  {h.payload['chunk_id']}")
            print(f"  {h.payload['text'][:200].replace(chr(10), ' ')}...")
            print()
    else:
        chunks = load_chunks(args.chunks)
        print(f"Loaded {len(chunks)} chunks from {args.chunks}")

        dim = model.get_embedding_dimension()
        build_collection(client, args.collection, dim)

        n = embed_and_upsert(chunks, model, client, args.collection)
        print(f"Upserted {n} points into collection '{args.collection}' (dim={dim})")