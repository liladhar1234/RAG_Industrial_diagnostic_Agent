"""Needs Qdrant running and the embedding models available.

    docker compose exec -e RUN_INTEGRATION=1 dev pytest tests/test_real_retriever_integration.py -v
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1",
    reason="needs Qdrant + models; set RUN_INTEGRATION=1",
)


@pytest.fixture(scope="module")
def retriever():
    from agent.real_retriever import RealRetriever
    return RealRetriever()


def test_exact_lookup(retriever):
    ev = retriever.get_event("5081")
    assert ev["kind"] == "fault" and ev["name"] == "Auxiliary fan broken"


def test_search_returns_only_events_with_scores_in_range(retriever):
    hits = retriever.search_events("auxiliary cooling fan stuck", k=5)
    assert hits
    assert all(h["kind"] in ("fault", "warning") for h in hits)
    assert all(0.0 <= h["score"] <= 1.0 for h in hits)
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)


def test_symptom_finds_fan_fault_in_top5(retriever):
    codes = [h["code"] for h in retriever.search_events("auxiliary fan is broken", k=5)]
    assert "5081" in codes


def test_full_agent_on_real_retriever(retriever):
    from agent.graph import build_graph
    graph = build_graph(retriever)
    cfg = {"configurable": {"thread_id": "real-1"}}
    out = graph.invoke({"user_turns": ["drive shows fault 5081"], "clarify_count": 0}, cfg)
    assert "MK-FAN-1001" in out["answer"]
    assert "Auxiliary fan broken" in out["answer"]
