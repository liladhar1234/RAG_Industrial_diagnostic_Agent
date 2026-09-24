import operator
from typing import Annotated, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # Reducer: new user messages are APPENDED, not overwritten
    user_turns: Annotated[list[str], operator.add]
    clarify_count: int
    typed_code: Optional[str]     # code the user typed, if any
    exact: Optional[dict]         # event found by exact code lookup
    candidates: list[dict]        # results of the hybrid search
    decision: str                 # confident | ambiguous | give_up | unknown_code | unverified
    fault: Optional[dict]
    fault_valid: bool
    question: Optional[str]
    parts_result: Optional[dict]
    answer: str
