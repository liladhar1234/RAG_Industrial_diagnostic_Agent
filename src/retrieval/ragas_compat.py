"""Compatibility shim: import this BEFORE anything imports ragas.

Problem
    ragas 0.4.3 runs `from langchain_community.chat_models.vertexai import
    ChatVertexAI` at import time. Newer langchain-community releases no longer
    ship that submodule, so `import ragas` dies with ModuleNotFoundError before
    any of your own code runs.

Why a stub is safe
    ragas only puts ChatVertexAI in a list of "models that support multiple
    completions" and checks it with isinstance(). We never use Vertex AI, so a
    placeholder class simply never matches.

Usage (first import line of eval_ragas.py, above any ragas import):
    import ragas_compat  # noqa: F401
"""
import sys
import types
import warnings


def _stub_missing_vertexai():
    name = "langchain_community.chat_models.vertexai"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # langchain-community sunset notice
            __import__(name)                     # real module present? leave it alone
    except ModuleNotFoundError:
        stub = types.ModuleType(name)
        stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[name] = stub


_stub_missing_vertexai()
