"""Turns fault/warning chunks from chunks.jsonl into structured events.

No heavy dependencies on purpose: this module is importable and testable
without Qdrant, embedding models, or the reranker.
"""
import re

EVENT_SECTIONS = {"fault_tracing", "warning_tracing"}

_NAME = re.compile(
    r"^(?:Fault|Warning)\s+\S+?:[ \t]*(.*?)"
    r"(?=\nCause:|\nWhat to do:|\nRelated sub-codes:|\Z)",
    re.S,
)
_CAUSE = re.compile(
    r"\nCause:[ \t]*(.*?)(?=\nWhat to do:|\nRelated sub-codes:|\Z)", re.S
)
_ACTION = re.compile(
    r"\nWhat to do:[ \t]*(.*?)(?=\nRelated sub-codes:|\Z)", re.S
)
_SUBS = re.compile(r"\nRelated sub-codes:\s*(.*)\Z", re.S)


def _clean(s):
    return " ".join(s.split())


def _parse_sub_codes(block):
    subs = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("- "):
            code, _, desc = line[2:].partition(":")
            subs.append({"aux_code": code.strip(), "cause": desc.strip()})
        elif subs:                          # wrapped continuation line
            subs[-1]["cause"] += " " + line
    return subs


def parse_event(chunk):
    md = chunk["metadata"]
    _, _, body = chunk["text"].partition("\n")     # drop the breadcrumb line

    def grab(pattern):
        m = pattern.search(body)
        return _clean(m.group(1)) if m else ""

    sub_match = _SUBS.search(body)
    kind = md.get("kind") or (
        "fault" if md.get("section") == "fault_tracing" else "warning"
    )
    return {
        "chunk_id": chunk["id"],
        "code": str(md["code"]).upper(),
        "kind": kind,
        "name": grab(_NAME),
        "cause": grab(_CAUSE),
        "action": grab(_ACTION),
        "sub_codes": _parse_sub_codes(sub_match.group(1)) if sub_match else [],
        "page": md.get("page"),
    }


class EventIndex:
    def __init__(self, chunks):
        self.events = []
        self.by_code = {}
        self.by_chunk_id = {}
        self.collisions = []       # codes that exist as BOTH a fault and a warning
        for c in chunks:
            if c["metadata"].get("section") not in EVENT_SECTIONS:
                continue
            ev = parse_event(c)
            self.events.append(ev)
            self.by_chunk_id[ev["chunk_id"]] = ev
            prev = self.by_code.get(ev["code"])
            if prev is None:
                self.by_code[ev["code"]] = ev
            else:
                self.collisions.append(ev["code"])
                if ev["kind"] == "fault" and prev["kind"] != "fault":
                    self.by_code[ev["code"]] = ev     # prefer the fault

    def get(self, code):
        return self.by_code.get(code.strip().upper())
