"""RealRetriever: adapts the Phase 2 hybrid pipeline to the interface the
agent expects.

    get_event(code)            exact lookup, no search involved
    search_events(query, k)    hybrid search restricted to fault/warning chunks

Design notes
- Scope: only fault_tracing / warning_tracing chunks are searched, so
  parameter chunks (05.20, 12.03, ...) can't outrank the real fault.
- BM25 is built over the event chunks only, so parameters can't crowd it.
- Dense search can't be filtered without knowing the Qdrant payload schema,
  so we over-fetch (dense_fetch_k) and keep only event ids.
- The cross-encoder is loaded ONCE here (rerank() in hybrid_search.py
  reloads it on every call, which is far too slow for an interactive agent).
- Scores are returned in 0..1 so the agent's threshold means the same
  thing whichever ranking mode is used.
"""
import math
import os
import sys
import time
from pathlib import Path

_RETRIEVAL_DIR = str(Path(__file__).resolve().parents[1] / "retrieval")
if _RETRIEVAL_DIR not in sys.path:
    sys.path.append(_RETRIEVAL_DIR)          # hybrid_search imports bm25_search

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from bm25_search import tokenize
from hybrid_search import RRF_K, bm25_ranking, dense_ranking, load_chunks, rrf_fuse
from agent.event_index import EVENT_SECTIONS, EventIndex


def _to_probs(raw):
    """Cross-encoders return raw logits or 0..1 values depending on the
    sentence-transformers version. If anything falls outside 0..1, treat the
    batch as logits and apply a sigmoid."""
    if raw and (min(raw) < 0.0 or max(raw) > 1.0):
        return [1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x)))) for x in raw]
    return raw


class RealRetriever:
    def __init__(
        self,
        chunks_path="data/processed/chunks.jsonl",
        collection="acs580",
        embed_model="BAAI/bge-small-en-v1.5",
        reranker_model="BAAI/bge-reranker-base",
        use_reranker=True,
        pool_k=10,
        rerank_k=3,  # Phase 6: reranking only the top rerank_k of the fused
        # pool (not shrinking the fetch/fusion width itself) cuts rerank cost
        # ~3x with no accuracy loss; shrinking pool_k itself regressed
        # hard_negative hit@1 (1.00 -> 0.67) by starving RRF fusion - see phase6 log
        dense_fetch_k=150,
    ):
        chunks = load_chunks(chunks_path)
        self.index = EventIndex(chunks)
        self.event_chunks = [
            c for c in chunks if c["metadata"].get("section") in EVENT_SECTIONS
        ]
        self.text_by_id = {c["id"]: c["text"] for c in self.event_chunks}
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.event_chunks])

        self.collection = collection
        self.pool_k = pool_k
        self.rerank_k = rerank_k
        self.dense_fetch_k = dense_fetch_k
        self.client = QdrantClient(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333")
        )
        self.embedder = SentenceTransformer(embed_model)
        self.reranker = CrossEncoder(reranker_model) if use_reranker else None
        self.last_timing = {}

    # ---- interface used by the agent -------------------------------------
    def get_event(self, code):
        return self.index.get(code)

    def search_events(self, query, k=5):
        timing = {}

        t = time.perf_counter()
        dense_all = dense_ranking(
            self.client, self.embedder, self.collection, query, self.dense_fetch_k
        )
        dense_ids = [c for c in dense_all if c in self.index.by_chunk_id][: self.pool_k]
        timing["dense_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        bm25_ids = bm25_ranking(self.bm25, self.event_chunks, query, self.pool_k)
        timing["bm25_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        fused = rrf_fuse([dense_ids, bm25_ids])
        pool = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[: self.pool_k]
        timing["fuse_ms"] = (time.perf_counter() - t) * 1000

        if self.reranker is not None and pool:
            t = time.perf_counter()
            rerank_pool = pool[: self.rerank_k]
            ids = [cid for cid, _ in rerank_pool]
            pairs = [[query, self.text_by_id[cid]] for cid in ids]
            raw = [float(s) for s in self.reranker.predict(pairs)]
            scored = sorted(zip(ids, _to_probs(raw)), key=lambda x: x[1], reverse=True)
            timing["rerank_ms"] = (time.perf_counter() - t) * 1000
        else:
            best = 2.0 / (RRF_K + 1)          # rank 1 in both retrievers
            scored = [(cid, s / best) for cid, s in pool]

        self.last_timing = timing
        return [dict(self.index.by_chunk_id[cid], score=float(s)) for cid, s in scored[:k]]
