"""Interactive test harness for the agent.

    docker compose exec dev python src/agent/cli.py
    docker compose exec dev python src/agent/cli.py --no-rerank   # lighter on RAM
"""
import argparse
import uuid

from langgraph.types import Command

from agent.graph import build_graph
from agent.real_retriever import RealRetriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--llm", action="store_true", help="use qwen2.5:3b to phrase final answers")
    args = ap.parse_args()

    print("Loading models (first run takes a while)...")
    retriever = RealRetriever(use_reranker=not args.no_rerank)
    graph = build_graph(retriever, use_llm=args.llm)
    print("Ready. Describe a symptom or type a code. Ctrl+C to quit.")

    while True:
        try:
            text = input("\nsymptom> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue

        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        out = graph.invoke({"user_turns": [text], "clarify_count": 0}, cfg)
        while graph.get_state(cfg).next:                    # paused on a question
            print("\nagent> " + graph.get_state(cfg).values["question"])
            reply = input("you> ").strip()
            out = graph.invoke(Command(resume=reply), cfg)

        print("\n" + out["answer"])
        if retriever.last_timing:
            parts = ", ".join(f"{k}={v:.0f}" for k, v in retriever.last_timing.items())
            print(f"\n[retrieval ms: {parts}]")


if __name__ == "__main__":
    main()
