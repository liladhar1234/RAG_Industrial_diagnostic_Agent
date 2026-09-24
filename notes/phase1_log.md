# Phase 1 Log — Ingestion & Structure-Aware Chunking

Source: ABB ACS580 manual (`user_manual2.pdf`)
- Parameter list: pages 161–370 (`No. Name/Value | Description | Def/FbEq16`)
- Warning table: pages 410–419 (`Code (hex) | Warning / Aux. code | Cause | What to do`)
- Fault table: pages 420–430 (`Code (hex) | Fault / Aux. code | Cause | What to do`)

Final counts: **966 parameters**, **78 warnings**, **94 faults** (172 fault/warning events total).

## The core concept: parent-aware, state-carrying walkers

PDF tables are visual, not relational — a human sees indentation and a blank
cell and understands "this row belongs to the row above it," but the PDF
stores no such relationship. Both walkers (`walk_parameters`, `walk_events`)
rebuild this relationship with a `current` variable that remembers the last
parent row seen, and — critically — that variable has to survive across
page boundaries, since many pages open mid-parameter or mid-fault (their
parent is on the previous page).

## Three bugs, one underlying lesson

Each of these was found by running the walker on real data and inspecting
the output, not by reasoning about the manual in the abstract.

### 1. Repeated table headers leaking into `events.py`

Every page of the fault/warning chapter repeats its own header row:
```
['Code\n(hex)', 'Warning / Aux. code', 'Cause', 'What to do']
['Code\n(hex)', 'Fault / Aux. code',   'Cause', 'What to do']
```
`"Code\n(hex)"` doesn't match `CODE_RE`, so the walker's first version
treated it as a **child row** and silently attached it as a fake sub-code
to whichever fault/warning happened to be `current` at the end of the
previous page.

**Fix:** detect header rows explicitly (`cells[0].strip().lower().startswith("code")`)
and skip them — while also using the header's own second column
(`"Warning / Aux. code"` vs `"Fault / Aux. code"`) to detect which section
we're in. This is more reliable than scanning page text for keywords like
"warning message," since it keys off the exact boundary where the table
itself changes kind.

### 2. Parameter-group headers leaking into `params.py`

The parameter list is organized into ~38 groups (`01 Actual values`,
`03 Input references`, ... `98 User motor parameters`). A group header row
like `['01 Actual values', 'Basic signals for monitoring the drive...', '']`
doesn't match `PARAM_RE` (which requires `\d{2}\.\d{2}`, e.g. `01.01`), so
it fell into the same "child row" branch — meaning **every one of the 38
group boundaries silently attached itself as a bogus option** to the last
real parameter of the previous group (e.g. `01.68 -> "03 Input references"`).

Only the very first group header (page 161) was visible as an orphan,
because `current` was still `None` at that point. All 37 others were
invisible corruption — they didn't error, they just quietly poisoned the
data.

**Fix:** a second regex, `GROUP_HEADER_RE = r"^\d{2}\s+[A-Za-z]"`, checked
*before* the child-row branch, logs and skips bare group headers instead of
attaching them.

**Confirmed via a targeted check** (searching all parsed options for values
matching the group-header pattern): before the fix, 30 leaks found; after,
zero (`leaks: []`), with `Group headers skipped: 38` logged instead.

### 3. Table-boundary collapse on 10 pages

pdfplumber's table detection breaks on 10 pages within the parameter range
(169, 189, 202, 204–205, 212, 263–265, 278, 303, 356, 362) — likely because
these pages mix in inline bit-definition tables or diagrams that break the
normal 3-column layout. On these pages, the entire table collapses into a
single row where the header text and the first parameter's data are mashed
into one long string, e.g. on page 169:
```
"No. Name/Value Description Def/FbEq16 06.11 Main status word Main status
word of the drive. - For the bit descriptions see page 467. ..."
```
This string starts with `"No. Name/Value"`, so it gets caught by the
existing header-skip check and dropped entirely — meaning **any parameters
that live only on these 10 pages are currently missing from the output**,
not corrupted, just silently absent.

**Status: known gap, not fixed.** Recovering this would mean falling back
to `page.extract_text()` + regex for just these 10 pages, or trying
different `pdfplumber` table-detection settings. Deferred — flagging it
honestly here is more valuable right now than the extra parsing work,
since it doesn't block Phase 2/3.

### The unifying insight

All three bugs are the same lesson stated three different ways:
**a parent-aware walker needs an explicit "this row is structural metadata,
not data" check for every kind of non-parameter row a table can contain.**
Parent/child inference by itself isn't enough — you have to enumerate what
"not a parent, but also not a real child" looks like (repeated headers,
group headers, and — separately — that boundary collapse is worth flagging
during ingestion, not just tolerated).

## Naive chunking vs. table-aware chunking (checkpoint demo)

Fault `5081 Auxiliary fan broken` (page 423) has two orphaned aux rows
beneath it:
```
['5081', 'Auxiliary fan broken', 'An auxiliary cooling fan ... is stuck or disconnected.', 'Check the auxiliary code. ...']
['',     '0001',                 'Auxiliary fan 1 broken.',                                   '']
['',     '0002',                 'Auxiliary fan 2 broken.',                                   '']
```

**Naive chunking** (one chunk per row, independently):
```
Code: (empty) | Aux: 0002 | Cause: Auxiliary fan 2 broken.
```
This chunk never mentions `5081`, has no remedy, and is ambiguous — the aux
code `0002` also appears under other faults (e.g. the AI-supervision fault
`80A0`). A technician searching "fault 5081 fan 2" would never match it.

**Table-aware chunking** (`event_to_chunk`, folding aux rows into the parent):
```
Fault 5081: Auxiliary fan broken
Cause: An auxiliary cooling fan (connected to the fan connectors on the
control unit) is stuck or disconnected.
What to do: Check the auxiliary code. Check auxiliary fan(s) and
connection(s). Replace fan if faulty. ...
Related sub-codes:
  - 0001: Auxiliary fan 1 broken.
  - 0002: Auxiliary fan 2 broken.
```
This version retrieves correctly on both "fault 5081" and "aux 0002,"
carries the full remedy text, and disambiguates which parent the sub-code
belongs to.

## Open design question, deferred to Phase 3

Both `parent_to_chunk()` and `event_to_chunk()` currently fold every child
row into a single parent chunk. The alternative — a parent chunk plus
separate child chunks, each repeating the parent's id/name — keeps chunks
smaller and more precise but introduces duplication and near-identical
retrieval hits. No universally right answer here; this is exactly what
Phase 3's hit@k/MRR eval is for. Also worth testing there: the hard
negative already spotted in `A8BF`/`37.04` (a parameter description that
merely *mentions* a warning code, which could outrank the actual warning
definition on a BM25 search for that code).

## Status

Phase 1 checkpoints met: parent-aware walkers built and bug-fixed for both
table families; naive-vs-table-aware demo captured; known gaps logged
honestly rather than hidden. Ready to move to Phase 2 (retrieval).