"""
chainlit_app.py

Phase 5: Chainlit front end, calling the FastAPI layer over HTTP (not
importing the graph directly) - this actually exercises /diagnose and
/session/{id} rather than bypassing them.

Usage (inside Docker, with the API already running on port 8000):
    docker compose exec dev chainlit run src/ui/chainlit_app.py --host 0.0.0.0 --port 8001
"""
import os

import chainlit as cl
import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000")


@cl.on_chat_start
async def start():
    cl.user_session.set("session_id", None)
    await cl.Message(
        content="Describe a symptom or give a fault/warning code."
    ).send()


async def _post_diagnose(session_id, message):
    async with httpx.AsyncClient(timeout=60) as client:
        return await client.post(
            f"{API_URL}/diagnose",
            json={"session_id": session_id, "message": message},
        )


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    resp = await _post_diagnose(session_id, message.content)

    if resp.status_code == 409:
        # prior session already finished - this is a new question
        resp = await _post_diagnose(None, message.content)
    elif resp.status_code == 404:
        await cl.Message(content="Session expired - starting a new one.").send()
        resp = await _post_diagnose(None, message.content)

    if resp.status_code >= 400:
        await cl.Message(content=f"Error: {resp.text}").send()
        return

    data = resp.json()
    cl.user_session.set("session_id", data["session_id"])

    if data["status"] == "waiting_for_clarification":
        await cl.Message(content=data["question"]).send()
        return

    elements = []
    if data.get("sources"):
        lines = [
            f"- **{s['kind']} {s['code']}** (manual page {s.get('page', '?')})"
            for s in data["sources"]
        ]
        elements.append(cl.Text(name="Sources", content="\n".join(lines), display="side"))

    await cl.Message(content=data["answer"], elements=elements).send()