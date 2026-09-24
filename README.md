# Industrial Equipment Diagnostic & Spare-Part Agent

## What is this?

Imagine a technician standing in front of a broken industrial machine
(an ABB ACS580 motor drive). Something's wrong, but the service manual
is 400+ pages long. Instead of flipping through pages, they type what
they see — either a fault code on the screen, or just a plain
description like *"the auxiliary fan is broken"* — and this system:

1. **Finds the right page** in the manual automatically
2. **Asks a follow-up question** if it's not sure what's wrong yet
3. **Explains the cause and the fix**, in the manual's own words
4. **Tells them which spare part to order**, whether it's in stock,
   and where the nearest warehouse is

It's a chatbot, but one that's only allowed to say things it can
actually prove come from the manual and the parts database — it's not
allowed to make things up, especially not part numbers.

---

## Why this is harder than "just ask ChatGPT"

A general-purpose chatbot doesn't know this specific manual, and even
if you fed it the whole PDF, it could still **hallucinate** — confidently
say something wrong, like inventing a spare part number that doesn't
exist. For an industrial technician, a wrong part number is a real
problem: wasted time, wrong parts ordered, possibly downtime.

So this project isn't really about "make a chatbot." It's about making
one that's **honest about what it doesn't know**, always backs up its
answers with the actual source material, and has a hard safety check
that throws away anything it can't verify.

---

## How it actually works, in plain terms

Think of it as an assembly line with four stations:

**1. Reading the manual (once, ahead of time)**
A script reads the PDF and carefully pulls out every fault code, every
warning, and every spare-part-relevant table — keeping the structure
intact (a fault code and its explanation always travel together, they
never get split apart or jumbled).

**2. Finding the right answer (search)**
When someone types a question, the system searches two different ways
at once:
- **Exact matching** — good for when someone types an exact code
  like "5081"
- **Meaning-based matching** — good for when someone describes a
  *symptom* in their own words, like "the drive keeps overheating"

Then a final, smarter pass double-checks the top few candidates and
picks the single best match — this step alone measurably fixed several
cases where the system would otherwise have picked a close-but-wrong
answer.

**3. Deciding, asking, or looking things up (the "agent")**
This is the decision-making brain. It checks: *am I actually confident
about this fault, or could it be one of a few different things?* If
it's not sure, it asks a clarifying question — just like a real
technician would ask a colleague. If it's confident, it verifies the
fault is well-formed, then looks up matching spare parts in a small
parts database (in stock? which warehouse? how many days to ship?).

**4. Writing the final answer**
A small AI model rewrites the answer so it reads naturally instead of
like a robotic form. But — and this is the important part — **every
single part number it mentions is checked against the real database
first.** If it ever tries to mention a part number that isn't real, the
system throws that answer away completely and falls back to a plain,
guaranteed-accurate version instead. We tested this directly: we forced
it to try to invent a fake part, and confirmed the safety check catches
it every time.

---

## What you can actually do with it

- Type a fault code → get the cause, the fix, and the matching spare
  parts with live stock info
- Describe a symptom in your own words → get asked a quick clarifying
  question if needed, then the same detailed answer
- Use it from a chat window in your browser, or from the command line
- Everything runs **locally on your own machine** — no data is sent to
  any outside company, and there's no API cost, since the AI model
  (`qwen2.5:3b`) runs on your own computer via Ollama

---

## What's under the hood (for the curious)

| Piece | What it does | In plain terms |
|---|---|---|
| PDF parser | Reads the manual | Turns a messy PDF into clean, structured data |
| Qdrant | "Meaning" search | Finds things that *mean* the same thing, even with different words |
| BM25 | "Exact word" search | Finds exact matches, like a fault code |
| Reranker | Picks the best answer | Double-checks the top few guesses and picks the real winner |
| LangGraph | The decision-maker | Runs the whole "ask → decide → verify → answer" flow |
| SQLite | Parts database | A small, safe, read-only spare-parts lookup |
| Ollama + qwen2.5:3b | Writes the final answer | A small AI model that only rephrases, never invents facts |
| FastAPI | The engine room | Lets the chat interface talk to everything above it |
| Chainlit | The chat window | What you actually see and type into |

A full technical write-up — including every design decision, every bug
we found and fixed, what's still not perfect, and what we'd improve
next — is in [`project_report.md`](./project_report.md) (also available
as a Word document).

---

## Running it yourself

**You'll need:** Docker Desktop installed and running.

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd Bruviti_AI

# 2. Add the manual (not included in this repo — see note below)
#    place your ACS580 service manual PDF at:
#    data/raw/user_manual2.pdf

# 3. Start everything
docker compose --profile retrieval --profile serve --profile local-llm up -d

# 4. Pull the local AI model (first time only)
docker compose exec ollama ollama pull qwen2.5:3b

# 5. Open the chat interface
#    http://localhost:8501

# API docs, if you want to see the raw endpoints:
#    http://localhost:8000/docs
```

**Note on the manual:** the ABB ACS580 service manual is ABB's own
copyrighted document, so it isn't included in this repository. You'll
need to supply your own copy of the manual PDF for the ingestion step
to work.

---

## What's still not perfect (and that's on purpose, not hidden)

No project like this is ever "done" — and pretending otherwise isn't
honest. A few known limitations, documented rather than swept under the
rug:

- It sometimes takes a few seconds to answer a symptom-style question
  (the "double-check" step is thorough but not instant)
- A couple of edge-case questions (like "how do I install X" instead of
  "X is broken") can currently confuse it into giving a confident but
  wrong-*type* of answer
- If you restart the server, it forgets any conversation that was
  mid-question (easy to fix, just not done yet)

Full details, plus what we'd do about each of these, are in the project
report.
