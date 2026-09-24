"""
scan_headers.py

For a manual with no TOC/bookmarks (like user_manual2.pdf), find where
each table family lives by scanning every page's first table header
and grouping consecutive pages that share the same header signature.

Usage (inside Docker):
    docker compose exec dev python src/ingest/scan_headers.py data/raw/user_manual2.pdf
"""

import sys

import pdfplumber


def header_signature(table):
    """First row of a table, cleaned up, used to identify which
    table family a page belongs to (e.g. the parameter-list header
    vs the fault/warning header vs an appendix table)."""
    if not table or not table[0]:
        return None
    row = table[0]
    cells = [(c or "").strip().replace("\n", " ") for c in row]
    if not any(cells):
        return None
    return tuple(cells)


def scan(pdf_path):
    ranges = []  # list of [signature, first_page, last_page]
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            pno = i + 1
            tables = page.extract_tables()
            sig = None
            for t in tables:
                sig = header_signature(t)
                if sig:
                    break  # use the first non-empty table header on the page

            if sig is None:
                continue

            if ranges and ranges[-1][0] == sig:
                ranges[-1][2] = pno  # extend current run
            else:
                ranges.append([sig, pno, pno])

    return ranges


if __name__ == "__main__":
    pdf_path = sys.argv[1]
    ranges = scan(pdf_path)
    print(f"{len(ranges)} distinct header runs found:\n")
    for sig, first, last in ranges:
        print(f"pages {first}-{last}: {sig}")
