"""
params.py

Parent-aware walker for the ABB parameter-list tables
(No. Name/Value | Description | Def/FbEq16), e.g. pages 293-309.

A row like '36.09 Reset loggers' is a PARENT (a parameter).
Rows beneath it ('Done', 'All', ...) are its selectable VALUES and
are meaningless on their own, so the walker remembers the current
parent (`current`) and attaches child rows to it. This state has to
survive across pages, since several pages open mid-parameter.

Usage (inside Docker):
    docker compose exec dev python src/ingest/params.py data/raw/<file>.pdf 293 309
"""

import argparse
import json
import re
import sys

import pdfplumber

PARAM_RE = re.compile(r"^(\d{2}\.\d{2})\s+(.+)$", re.S)
GROUP_HEADER_RE = re.compile(r"^\d{2}\s+[A-Za-z]")
HEADER_TEXT = "No. Name/Value"


def clean_cell(c):
    return (c or "").strip()


def walk_parameters(pdf_path, first_page, last_page):
    """
    Parent-aware walker for the parameter tables.

    Returns a list of parent dicts, each with an `options` list of
    the child rows that belong to it:

        {
          "id": "36.09",
          "name": "Reset loggers",
          "description": "...",
          "default": "...",
          "options": [{"value": "Done", "description": "...", "fbeq": "..."}, ...],
          "page": 297,
        }
    """
    parents = []
    current = None
    orphans = []
    group_headers = []

    with pdfplumber.open(pdf_path) as pdf:
        for pno in range(first_page, last_page + 1):
            for table in pdf.pages[pno - 1].extract_tables():
                for row in table:
                    cells = [clean_cell(c) for c in row]
                    if not any(cells) or cells[0].startswith(HEADER_TEXT):
                        continue  # blank row, or a repeated header row

                    name = cells[0]
                    desc = cells[1] if len(cells) > 1 else ""
                    last = cells[-1] if len(cells) > 1 else ""

                    m = PARAM_RE.match(name)
                    if m:  # parent row: starts a new parameter
                        current = {
                            "id": m[1],
                            "name": " ".join(m[2].split()),
                            "description": desc,
                            "default": last,
                            "options": [],
                            "page": pno,
                        }
                        parents.append(current)
                    elif GROUP_HEADER_RE.match(name):
                        # a bare group header like '01 Actual values', not a
                        # parameter and not a child option - log it and move
                        # on, don't attach it to whatever parameter came
                        # before it
                        group_headers.append((pno, name))
                    elif current:  # child row: a selectable value
                        current["options"].append(
                            {"value": name, "description": desc, "fbeq": last}
                        )
                    else:
                        # child row before we've ever seen a parent - log it,
                        # don't silently drop it
                        orphans.append((pno, cells))

    if group_headers:
        print(f"Group headers skipped: {len(group_headers)}", file=sys.stderr)
        for pno, name in group_headers:
            print(f"GROUP page={pno} {name!r}", file=sys.stderr)

    if orphans:
        print(f"Orphan rows (no parent yet seen): {len(orphans)}", file=sys.stderr)
        for pno, cells in orphans:
            print(f"ORPHAN page={pno} {cells}", file=sys.stderr)

    return parents


def parent_to_chunk(parent, manual="unknown-manual", section="parameter_list"):
    """
    Turn one parameter (+ its options) into a single embeddable text
    chunk. Options are folded into the parent chunk here, since an
    option like 'All' or 'Done' is ambiguous without its parameter's
    ID and name attached.

    `manual` and `section` are a breadcrumb: once chunks from several
    manuals (or several chapters of one manual) share a vector store,
    two identical-looking codes (e.g. two different manuals both
    having a "01.01") need something to disambiguate them by. This is
    the same breadcrumb idea from the original roadmap
    ("Manual > Troubleshooting > Error Codes | Code: E-47 | ...").

    This is a design choice, not the only right answer - in Phase 3,
    compare this against splitting options into separate child chunks
    (each repeating the parent's id/name) and see which retrieves
    better for queries like "how do I reset the amplitude logger".
    """
    lines = [
        f"{manual} > {section} > {parent['id']}",
        f"Parameter {parent['id']}: {parent['name']}",
        f"Description: {parent['description']}" if parent["description"] else "",
        f"Default: {parent['default']}" if parent["default"] else "",
    ]
    if parent["options"]:
        lines.append("Options:")
        for opt in parent["options"]:
            fb = f" (fbeq: {opt['fbeq']})" if opt["fbeq"] else ""
            lines.append(f"  - {opt['value']}: {opt['description']}{fb}")

    text = "\n".join(line for line in lines if line)
    return {
        "id": f"{manual}-param-{parent['id']}",
        "text": text,
        "metadata": {
            "manual": manual,
            "section": section,
            "code": parent["id"],
            "page": parent["page"],
        },
    }


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Walk parameter tables in a PDF manual.")
    arg_parser.add_argument("pdf_path")
    arg_parser.add_argument("first_page", type=int, nargs="?", default=293)
    arg_parser.add_argument("last_page", type=int, nargs="?", default=309)
    arg_parser.add_argument("--manual", default="unknown-manual",
                             help="breadcrumb name for this manual, e.g. ACS580")
    args = arg_parser.parse_args()

    ps = walk_parameters(args.pdf_path, args.first_page, args.last_page)
    print(len(ps), "parameters")

    if ps:
        print("\nExample parent parameter:")
        print(json.dumps(ps[0], indent=2))

        print("\nExample chunk:")
        print(json.dumps(parent_to_chunk(ps[0], manual=args.manual), indent=2))