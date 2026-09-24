"""
schemas.py

Pydantic request/response models for the FastAPI layer. Kept separate
from main.py so the contract is easy to point to on its own.
"""
from typing import Optional
from pydantic import BaseModel


class DiagnoseRequest(BaseModel):
    session_id: Optional[str] = None   # omit to start a new session
    message: str


class SourceRef(BaseModel):
    chunk_id: str
    kind: str
    code: str
    page: Optional[int] = None


class DiagnoseResponse(BaseModel):
    session_id: str
    status: str               # "answered" | "waiting_for_clarification"
    answer: Optional[str] = None
    question: Optional[str] = None
    sources: Optional[list[SourceRef]] = None
    clarify_count: Optional[int] = None


class SessionStateResponse(BaseModel):
    session_id: str
    status: str
    answer: Optional[str] = None
    question: Optional[str] = None
    clarify_count: int


class HealthResponse(BaseModel):
    status: str                # "ok" | "degraded"
    qdrant_ok: bool
    detail: Optional[str] = None