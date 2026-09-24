"""
bm25_search.py

Phase 2, step 2: BM25-only baseline, to compare against the dense-only
baseline in embed.py. The whole point of this leg is exact lexical
matching - codes like "5081" or "A2B1" should be retrieved exactly,
which dense embedding got wrong for "fault 5081" (see phase2 notes).

Tokenization is the one thing worth getting right here: a naive
word-splitter would break part-number-style tokens like "4521-B" apart
at the hyphen, or lowercase a hex code inconsistently. This tokenizer
keeps alphanumeric runs (including internal hyphens/dots, so "4521-B"
and "36.09" survive as single tokens) and lowercases everything for
case-insensitive matching (so "A2B1" and "a2b1" are the same token).

Usage (inside Docker):
    docker compose exec dev python src/retrieval/bm25_search.py \\
        --chunks data/processed/chunks.jsonl \\
        --search "fault 5081"
"""

import argparse
import json
import re

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\.]*", re.IGNORECASE)


def tokenize(text):
    """Lowercase, keep alphanumeric runs with internal hyphens/dots so
    codes like 'A2B1', '5081', '36.09', and part numbers like '4521-B'
    survive as single tokens instead of being split apart."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def load_chunks(path):
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_index(chunks):
    tokenized = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized)


def search(bm25, chunks, query, top_k=5):
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(chunks[i], scores[i]) for i in ranked]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM25-only search over chunks.jsonl.")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--search", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    bm25 = build_index(chunks)

    results = search(bm25, chunks, args.search, args.top_k)
    print(f"Top {args.top_k} results for: {args.search!r}\n")
    for chunk, score in results:
        print(f"score={score:.4f}  {chunk['id']}")
        print(f"  {chunk['text'][:200].replace(chr(10), ' ')}...")
        print()
