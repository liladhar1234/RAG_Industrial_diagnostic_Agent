from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from agent.state import AgentState
from agent.nodes import make_nodes


def route_after_assess(state):
    if state["decision"] == "confident":
        return "verify_code"
    if state["decision"] == "ambiguous":
        return "prepare_question"
    return "compose_answer"          # give_up / unknown_code


def route_after_verify(state):
    return "query_parts_db" if state["fault_valid"] else "compose_answer"


def build_graph(retriever, checkpointer=None, use_llm=False):
    g = StateGraph(AgentState)
    for name, fn in make_nodes(retriever, use_llm=use_llm).items():
        g.add_node(name, fn)

    g.add_edge(START, "parse_symptoms")
    g.add_edge("parse_symptoms", "retrieve")
    g.add_edge("retrieve", "assess")
    g.add_conditional_edges(
        "assess",
        route_after_assess,
        {
            "verify_code": "verify_code",
            "prepare_question": "prepare_question",
            "compose_answer": "compose_answer",
        },
    )
    g.add_edge("prepare_question", "wait_for_user")
    g.add_edge("wait_for_user", "parse_symptoms")   # the loop; bounded by MAX_CLARIFY
    g.add_conditional_edges(
        "verify_code",
        route_after_verify,
        {
            "query_parts_db": "query_parts_db",
            "compose_answer": "compose_answer",
        },
    )
    g.add_edge("query_parts_db", "compose_answer")
    g.add_edge("compose_answer", END)

    # interrupt() needs a checkpointer to save state while paused
    return g.compile(checkpointer=checkpointer or MemorySaver())
