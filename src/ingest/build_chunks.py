"""
build_chunks.py

Bridge between Phase 1 (parsing) and Phase 2 (retrieval): runs both
walkers over their full page ranges and writes every resulting chunk to
a single JSONL file, one JSON object per line. This is the file Phase 2's
embedding step will read.

Usage (inside Docker):
    docker compose exec dev python src/ingest/build_chunks.py \\
        --pdf data/raw/user_manual2.pdf \\
        --manual ACS580 \\
        --param-range 161 370 \\
        --event-range 410 430 \\
        --out data/processed/chunks.jsonl
"""

import argparse
import json
import os

from ingest.params import walk_parameters, parent_to_chunk
from ingest.events import walk_events, event_to_chunk


def build_chunks(pdf_path, manual, param_range, event_range):
    chunks = []

    if param_range:
        first, last = param_range
        params = walk_parameters(pdf_path, first, last)
        for p in params:
            chunks.append(parent_to_chunk(p, manual=manual))

    if event_range:
        first, last = event_range
        events = walk_events(pdf_path, first, last)
        for e in events:
            chunks.append(event_to_chunk(e, manual=manual))

    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a JSONL chunk export from a manual.")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--manual", required=True)
    parser.add_argument("--param-range", nargs=2, type=int, metavar=("FIRST", "LAST"),
                         help="page range for the parameter list, e.g. 161 370")
    parser.add_argument("--event-range", nargs=2, type=int, metavar=("FIRST", "LAST"),
                         help="page range for the fault/warning tables, e.g. 410 430")
    parser.add_argument("--out", default="data/processed/chunks.jsonl")
    args = parser.parse_args()

    chunks = build_chunks(
        args.pdf,
        args.manual,
        tuple(args.param_range) if args.param_range else None,
        tuple(args.event_range) if args.event_range else None,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    print(f"{len(chunks)} chunks written to {args.out}")
    if chunks:
        print("\nExample chunk:")
        print(json.dumps(chunks[0], indent=2))
