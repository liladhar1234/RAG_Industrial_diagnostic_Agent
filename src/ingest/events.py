"""
events.py

Parent-aware walker for the fault/warning tables in the "Fault
tracing" chapter (e.g. pages 410-430).

A row like:
    ['5081', 'Auxiliary fan broken', 'An auxiliary cooling fan is stuck...', 'Check ...']
is a PARENT (a fault or warning code). Rows beneath it with an empty
first cell:
    ['', '0001', 'Auxiliary fan 1 broken.', '']
    ['', '0002', 'Auxiliary fan 2 broken.', '']
are CHILD sub-codes and are meaningless without the parent code, so
the walker remembers the current parent (`current`) and attaches
child rows to it. Some pages (e.g. 425, 430) open with orphan child
rows whose parent is on the previous page, so `current` has to
survive across pages, not just within one table.

Usage (inside Docker):
    docker compose exec dev python src/ingest/events.py data/raw/<file>.pdf 410 430
"""

import argparse
import json
import re
import sys

import pdfplumber

# ABB fault/warning codes observed so far (5081, A2B1, 80A0, A8BF) are
# all 4-character hex codes. Adjust this if a different manual uses a
# different code format.
CODE_RE = re.compile(r"^[0-9A-Fa-f]{4}$")


def clean_cell(c):
    return (c or "").strip()


def is_header_row(cells):
    """The fault/warning tables repeat their header on every page:
    ['Code\\n(hex)', 'Warning / Aux. code', 'Cause', 'What to do']
    ['Code\\n(hex)', 'Fault / Aux. code',   'Cause', 'What to do']
    These must be skipped, not treated as data - a naive walker would
    otherwise attach the header row itself as a fake child row to
    whatever parent came before it on the previous page."""
    return cells[0].strip().lower().startswith("code")


def section_from_header(cells):
    """The header row's second column names the section directly
    ('Warning / Aux. code' vs 'Fault / Aux. code') - more reliable
    than scanning page text for keywords, since it's exactly the
    boundary where the table itself changes kind."""
    label = cells[1].lower() if len(cells) > 1 else ""
    if "warning" in label:
        return "warning"
    if "fault" in label:
        return "fault"
    return None


def walk_events(pdf_path, first_page, last_page):
    """
    Parent-aware walker for fault/warning tables.

    Returns a list of parent dicts, each with an `aux` list of the
    child sub-codes that belong to it:

        {
          "kind": "fault",
          "code": "5081",
          "name": "Auxiliary fan broken",
          "cause": "...",
          "action": "...",
          "aux": [{"value": "0001", "description": "Auxiliary fan 1 broken."}, ...],
          "page": 423,
        }
    """
    events = []
    current = None
    section = None
    unhandled = []

    with pdfplumber.open(pdf_path) as pdf:
        for pno in range(first_page, last_page + 1):
            page = pdf.pages[pno - 1]

            for table in page.extract_tables():
                for row in table:
                    cells = [clean_cell(c) for c in row]
                    if not any(cells):
                        continue  # blank row

                    if is_header_row(cells):
                        detected = section_from_header(cells)
                        if detected:
                            section = detected
                        continue  # header row is metadata, not a data row

                    code = cells[0]
                    name = cells[1] if len(cells) > 1 else ""
                    cause = cells[2] if len(cells) > 2 else ""
                    action = cells[3] if len(cells) > 3 else ""

                    if CODE_RE.match(code):  # parent row: a fault/warning code
                        current = {
                            "kind": section or "unknown",
                            "code": code,
                            "name": name,
                            "cause": cause,
                            "action": action,
                            "aux": [],
                            "page": pno,
                        }
                        events.append(current)
                    elif current:  # child row: a sub-code beneath the last parent
                        current["aux"].append({"value": name, "description": cause})
                    else:
                        # child row before we've ever seen a parent - log it,
                        # don't silently drop it
                        unhandled.append((pno, cells))

    if unhandled:
        print(f"UNHANDLED rows: {len(unhandled)}", file=sys.stderr)
        for pno, cells in unhandled:
            print(f"UNHANDLED page={pno} {cells}", file=sys.stderr)

    return events


def event_to_chunk(event, manual="unknown-manual"):
    """
    Turn one parent event (+ its aux sub-codes) into a single
    embeddable text chunk. Aux codes are folded into the parent
    chunk here, since a sub-code like '0001' is ambiguous on its own
    (the same aux number can appear under several different parent
    codes) and only makes sense next to its parent.

    `manual` is a breadcrumb (e.g. "ACS580 > fault_tracing > 5081")
    so that once chunks from multiple manuals share a vector store,
    identical-looking codes across different manuals stay
    distinguishable.

    This is a design choice, not the only right answer - in Phase 3,
    test this against parent + separate child chunks and see which
    retrieves better, especially for the hidden hard-negative case
    (e.g. a query for code "A8BF" matching a parameter description
    that merely mentions it, rather than the real fault definition).
    """
    section = f"{event['kind']}_tracing"
    lines = [
        f"{manual} > {section} > {event['code']}",
        f"{event['kind'].title()} {event['code']}: {event['name']}",
        f"Cause: {event['cause']}" if event["cause"] else "",
        f"What to do: {event['action']}" if event["action"] else "",
    ]
    if event["aux"]:
        lines.append("Related sub-codes:")
        for a in event["aux"]:
            lines.append(f"  - {a['value']}: {a['description']}")

    text = "\n".join(line for line in lines if line)
    return {
        "id": f"{manual}-{event['kind']}-{event['code']}",
        "text": text,
        "metadata": {
            "manual": manual,
            "section": section,
            "kind": event["kind"],
            "code": event["code"],
            "page": event["page"],
        },
    }


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Walk fault/warning tables in a PDF manual.")
    arg_parser.add_argument("pdf_path")
    arg_parser.add_argument("first_page", type=int, nargs="?", default=410)
    arg_parser.add_argument("last_page", type=int, nargs="?", default=430)
    arg_parser.add_argument("--manual", default="unknown-manual",
                             help="breadcrumb name for this manual, e.g. ACS580")
    args = arg_parser.parse_args()

    ev = walk_events(args.pdf_path, args.first_page, args.last_page)

    for kind in ("warning", "fault"):
        print(kind, sum(e["kind"] == kind for e in ev))

    if ev:
        print("\nExample parent event:")
        print(json.dumps(ev[0], indent=2))

        print("\nExample chunk:")
        print(json.dumps(event_to_chunk(ev[0], manual=args.manual), indent=2))