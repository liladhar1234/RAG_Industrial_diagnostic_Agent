import pytest
from langgraph.types import Command
from agent.graph import build_graph
from agent.render import allowed_parts, part_numbers_in
from fakes import FakeRetriever


@pytest.fixture
def graph():
    return build_graph(FakeRetriever())


def start(graph, text, tid):
    cfg = {"configurable": {"thread_id": tid}}
    out = graph.invoke({"user_turns": [text], "clarify_count": 0}, cfg)
    return cfg, out


def paused(graph, cfg):
    return bool(graph.get_state(cfg).next)


def test_exact_code_goes_straight_to_parts(graph):
    cfg, out = start(graph, "drive shows fault 5081", "t1")
    assert not paused(graph, cfg)
    assert "MK-FAN-1001" in out["answer"]
    assert "out of stock" in out["answer"]           # MK-FAN-1002 has zero stock
    assert part_numbers_in(out["answer"]) <= allowed_parts(out["parts_result"])


def test_ambiguous_symptom_pauses_and_asks(graph):
    cfg, out = start(graph, "the drive is overheating", "t2")
    assert paused(graph, cfg)
    assert graph.get_state(cfg).values["question"]


def test_clarification_with_code_resolves(graph):
    cfg, _ = start(graph, "the drive is overheating", "t3")
    out = graph.invoke(Command(resume="the panel shows fault 4290"), cfg)
    assert not paused(graph, cfg)
    assert "MK-FAN-1003" in out["answer"]


def test_gives_up_after_two_vague_replies(graph):
    cfg, _ = start(graph, "the drive is overheating", "t4")
    graph.invoke(Command(resume="not sure"), cfg)
    assert paused(graph, cfg)                         # asked a second time
    out = graph.invoke(Command(resume="no idea"), cfg)
    assert not paused(graph, cfg)
    assert "escalat" in out["answer"].lower()


def test_unknown_code_is_not_guessed(graph):
    cfg, out = start(graph, "fault 9999", "t5")
    assert "couldn't find code 9999" in out["answer"]
    assert out.get("parts_result") is None


def test_fault_with_no_mapped_parts(graph):
    cfg, out = start(graph, "fault 2330", "t6")
    assert "No replacement part" in out["answer"]


def test_all_parts_out_of_stock(graph):
    cfg, out = start(graph, "fault 7192", "t7")
    assert "out of stock" in out["answer"]
    assert "in stock," not in out["answer"]


def test_year_in_text_is_not_mistaken_for_a_code(graph):
    cfg, out = start(graph, "drive has tripped since 2024", "t8")
    assert out.get("typed_code") is None
