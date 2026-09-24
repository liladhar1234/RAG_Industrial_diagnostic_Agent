import re
from langgraph.types import interrupt
from db.tools import parts_for_fault, FAULT_RE
from agent.render import render_answer
from agent.llm_render import llm_phrase_answer

MAX_CLARIFY = 2
MIN_GAP_RATIO = 0.1   
MIN_TOP_SCORE = 0.4 

CONTEXT_CODE = re.compile(
    r"\b(?:fault|warning|alarm|error|code)\b\s*(?:is|no\.?|number|[:#])?\s*([0-9A-F]{4})\b",
    re.I,
)
BARE_CODE = re.compile(r"\b([0-9A-F]{4})\b", re.I)


def collapse(cands):
    """One entry per (kind, code): parent and aux chunks of one fault
    must not look like competing candidates."""
    best = {}
    for c in cands:
        key = (c["kind"], c["code"])
        if key not in best or c["score"] > best[key]["score"]:
            best[key] = c
    return sorted(best.values(), key=lambda c: -c["score"])


def make_nodes(retriever, use_llm=False):
    def parse_symptoms(state):
        text = " ".join(state["user_turns"])
        typed = None
        ctx = CONTEXT_CODE.findall(text)
        if ctx:
            typed = ctx[-1].upper()
        else:
            for tok in reversed(BARE_CODE.findall(text)):
                if retriever.get_event(tok.upper()):
                    typed = tok.upper()
                    break
        exact = retriever.get_event(typed) if typed else None
        return {"typed_code": typed, "exact": exact}

    def retrieve(state):
        # An exact or typed-but-unknown code was already resolved in
        # parse_symptoms; running search here would only be discarded
        # by assess() below, at the cost of a full dense+BM25+rerank pass.
        if state.get("exact") or state.get("typed_code"):
            retriever.last_timing = {"dense_ms": 0, "bm25_ms": 0, "fuse_ms": 0, "rerank_ms": 0}
            return {"candidates": []}

        # Use only the LATEST turn, not the full accumulated history.
        # user_turns uses operator.add (state.py), so it grows on every
        # clarification round; joining all of it here would let the
        # original symptom text permanently outweigh whatever the user
        # says next, which is why "the drive is overheating" failed to
        # move the candidate list in testing. A new turn is treated as
        # a correction, not an addition, for retrieval purposes.
        query = state["user_turns"][-1]
        return {"candidates": retriever.search_events(query, k=5)}

    def assess(state):
        if state.get("exact"):
            return {"decision": "confident", "fault": state["exact"]}
        if state.get("typed_code"):
            return {"decision": "unknown_code"}
        cands = collapse(state.get("candidates", []))
        if cands:
            top = cands[0]
            if len(cands) == 1:
                gap_ok = True
            else:
                gap = (top["score"] - cands[1]["score"]) / max(top["score"], 1e-9)
                gap_ok = gap >= MIN_GAP_RATIO
            if gap_ok and top["score"] >= MIN_TOP_SCORE:
                return {"decision": "confident", "fault": top}
        if state.get("clarify_count", 0) >= MAX_CLARIFY:
            return {"decision": "give_up"}
        return {"decision": "ambiguous"}

    def prepare_question(state):
        top = collapse(state.get("candidates", []))[:3]
        if top:
            options = "; ".join(
                "{} {} ({})".format(c["kind"], c["code"], c["name"]) for c in top
            )
            q = (
                "I can't pin this down yet. The closest matches are: "
                + options
                + ". Does the drive display a code, or which of these matches what you see?"
            )
        else:
            q = (
                "Can you describe what the drive is doing, "
                "or tell me any code shown on its display?"
            )
        return {"question": q, "clarify_count": state.get("clarify_count", 0) + 1}

    def wait_for_user(state):
        reply = interrupt(state["question"])   # graph pauses here until resumed
        return {"user_turns": [reply]}

    def verify_code(state):
        f = state["fault"]
        ok = bool(FAULT_RE.match(f.get("code", ""))) and bool(f.get("action"))
        if ok:
            return {"fault_valid": True}
        return {"fault_valid": False, "decision": "unverified"}

    def query_parts_db(state):
        return {"parts_result": parts_for_fault(state["fault"]["code"])}

    def compose_answer(state):
        if use_llm:
            return {"answer": llm_phrase_answer(state)}
        return {"answer": render_answer(state)}

    return {
        "parse_symptoms": parse_symptoms,
        "retrieve": retrieve,
        "assess": assess,
        "prepare_question": prepare_question,
        "wait_for_user": wait_for_user,
        "verify_code": verify_code,
        "query_parts_db": query_parts_db,
        "compose_answer": compose_answer,
    }