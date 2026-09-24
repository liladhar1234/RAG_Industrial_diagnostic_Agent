"""
main.py

Phase 5: FastAPI layer over the existing LangGraph agent. Wraps the same
build_graph()/RealRetriever already exercised via cli.py - no new agent
logic here, purely a transport layer.

Session semantics (deliberately strict, not overloaded):
  - POST /diagnose with no session_id  -> starts a NEW session.
  - POST /diagnose with a session_id   -> that session MUST currently be
    paused waiting for a clarification reply (graph.get_state(cfg).next
    is truthy). The message is treated as the resume value.
      * unknown session_id            -> 404
      * session_id exists but not paused (already answered) -> 409, with
        a message telling the caller to start a new session. This avoids
        silently appending an unrelated new question onto a finished
        thread's user_turns history (the same accumulation bug fixed in
        nodes.py earlier in this project - reintroducing it here via a
        different code path would be a real regression).

Known limitation, documented rather than silently accepted: the graph's
checkpointer is MemorySaver, in-process only. All session state is lost
on server restart. A production deployment would swap in
langgraph.checkpoint.sqlite.SqliteSaver or a Postgres-backed checkpointer
- both are drop-in replacements for the `checkpointer=` argument to
build_graph(), no other code changes needed.

Usage (inside Docker):
    docker compose exec dev uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import uuid

from fastapi import FastAPI, HTTPException
from langgraph.types import Command

from agent.graph import build_graph
from agent.real_retriever import RealRetriever
from api.schemas import (
    DiagnoseRequest, DiagnoseResponse, SessionStateResponse,
    SourceRef, HealthResponse,
)

app = FastAPI(title="Industrial Diagnostic & Spare-Part Agent")

# Built once at process startup - model loads happen here, not per-request.
_retriever = RealRetriever()
_graph = build_graph(_retriever, use_llm=False)  # flip to True to use qwen2.5:3b phrasing

# Tracks which session_ids this process has actually created, so we can
# return a clean 404 for a made-up id rather than a confusing state error.
_known_sessions: set[str] = set()


def _cfg(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _sources_from_state(values: dict) -> list[SourceRef] | None:
    fault = values.get("fault")
    if not fault:
        return None
    return [SourceRef(
        chunk_id=fault.get("chunk_id", ""),
        kind=fault.get("kind", ""),
        code=fault.get("code", ""),
        page=fault.get("page"),
    )]


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(req: DiagnoseRequest):
    if req.session_id is None:
        session_id = str(uuid.uuid4())
        _known_sessions.add(session_id)
        cfg = _cfg(session_id)
        _graph.invoke({"user_turns": [req.message], "clarify_count": 0}, cfg)
    else:
        session_id = req.session_id
        if session_id not in _known_sessions:
            raise HTTPException(404, "unknown session_id")
        cfg = _cfg(session_id)
        state = _graph.get_state(cfg)
        if not state.next:
            raise HTTPException(
                409,
                "session is not waiting for a reply (already answered or "
                "never started) - start a new session for a new question",
            )
        _graph.invoke(Command(resume=req.message), cfg)

    state = _graph.get_state(_cfg(session_id))
    if state.next:
        return DiagnoseResponse(
            session_id=session_id,
            status="waiting_for_clarification",
            question=state.values.get("question"),
            clarify_count=state.values.get("clarify_count"),
        )

    values = state.values
    return DiagnoseResponse(
        session_id=session_id,
        status="answered",
        answer=values.get("answer"),
        sources=_sources_from_state(values),
    )


@app.get("/session/{session_id}", response_model=SessionStateResponse)
def get_session(session_id: str):
    if session_id not in _known_sessions:
        raise HTTPException(404, "unknown session_id")
    state = _graph.get_state(_cfg(session_id))
    values = state.values
    if state.next:
        return SessionStateResponse(
            session_id=session_id, status="waiting_for_clarification",
            question=values.get("question"),
            clarify_count=values.get("clarify_count", 0),
        )
    return SessionStateResponse(
        session_id=session_id, status="answered",
        answer=values.get("answer"),
        clarify_count=values.get("clarify_count", 0),
    )


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        _retriever.client.get_collections()
        return HealthResponse(status="ok", qdrant_ok=True)
    except Exception as e:
        return HealthResponse(status="degraded", qdrant_ok=False, detail=str(e))