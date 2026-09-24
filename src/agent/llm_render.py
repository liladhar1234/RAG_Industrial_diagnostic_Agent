"""
llm_render.py

Phase 4: LLM-phrased final answers, with a hard guardrail.

Design:
- The LLM is given ONLY facts already retrieved and verified (fault
  cause/action, parts_result from the read-only DB tool). It is asked
  to rephrase, never to add information.
- Every generated answer is checked against allowed_parts() /
  part_numbers_in() (render.py) before being trusted. If the LLM
  mentions any part number not actually in parts_result, the generated
  text is discarded entirely and render_answer()'s deterministic
  template is used instead. This is a hard fallback, not a warning -
  an agent giving a technician a wrong part number is worse than one
  that occasionally sounds templated.
- Only the "confident" decision path (a verified fault + optional
  parts) is phrased by the LLM. unknown_code / give_up / unverified
  stay on the deterministic template - there is no benefit to
  "naturalizing" an error message, and less surface area for the LLM
  to invent something in a state that's already an edge case.
- Any LLM/network failure falls back to the template silently; the
  agent must keep working even if Ollama is down.
"""
import json
import urllib.request

from agent.render import render_answer, allowed_parts, part_numbers_in

import os
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
MODEL = "qwen2.5:3b"

SYSTEM_PROMPT = """You are rephrasing a fault diagnosis for a technician.
Rules:
- Only use the facts given below. Do not add any cause, action, or part
  number that is not explicitly listed.
- Keep the fault code and part numbers EXACTLY as given, character for
  character.
- Be concise and clear, as if speaking to a technician on the floor.
- Do not add greetings, sign-offs, or disclaimers."""


def _build_prompt(state):
    f = state["fault"]
    pr = state.get("parts_result") or {}

    parts_lines = []
    if pr.get("mapped"):
        for p in pr["parts"]:
            if p.get("found"):
                part = p["part"]
                stock = "in stock" if p.get("in_stock") else "out of stock"
                parts_lines.append(
                    f"- {part['part_number']} ({part['description']}, "
                    f"${part['unit_price']:.2f}, {stock})"
                )
    parts_block = "\n".join(parts_lines) if parts_lines else "No parts mapped to this fault."

    return f"""{SYSTEM_PROMPT}

Fault code: {f['code']}
Fault name: {f['name']}
Cause: {f['cause']}
Action: {f['action']}
Manual page: {f['page']}

Available parts:
{parts_block}

Write the technician-facing response now."""


def _call_ollama(prompt, timeout=60):
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body.get("response", "").strip()


def llm_phrase_answer(state):
    """Returns LLM-phrased text if it passes the part-number guardrail,
    otherwise falls back to the deterministic template. Never raises -
    any failure degrades to render_answer(state)."""
    fallback = render_answer(state)

    if state.get("decision") != "confident":
        return fallback

    try:
        generated = _call_ollama(_build_prompt(state))
    except Exception as e:
        print(f"[llm_render] Ollama call failed, falling back: {e}")
        return fallback

    if not generated:
        return fallback

    mentioned = part_numbers_in(generated)
    allowed = allowed_parts(state.get("parts_result"))
    if not mentioned <= allowed:
        invented = mentioned - allowed
        print(f"[llm_render] guardrail tripped, invented part(s) {invented}, falling back")
        return fallback

    return generated