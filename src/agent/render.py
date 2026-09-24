import re

PART_IN_TEXT = re.compile(r"MK-[A-Z0-9-]{3,20}")


def allowed_parts(parts_result):
    if not parts_result or not parts_result.get("parts"):
        return set()
    return {p["part"]["part_number"] for p in parts_result["parts"] if p.get("found")}


def part_numbers_in(text):
    return set(PART_IN_TEXT.findall(text))


def _part_line(p):
    if not p.get("found"):
        return "- " + str(p.get("part_number", "?")) + ": not found in the parts database."
    part = p["part"]
    head = "- {} - {} (price {:.2f})".format(
        part["part_number"], part["description"], part["unit_price"]
    )
    if p["in_stock"]:
        loc = p["locations"][0]
        stock = "in stock, {} at {}, lead time {} day(s)".format(
            loc["qty_on_hand"], loc["name"], loc["lead_time_days"]
        )
    else:
        stock = "out of stock"
    note = " Note: " + p["note"] if p.get("note") else ""
    return head + ": " + stock + "." + note


def _closest(cands):
    parts = []
    for c in cands[:3]:
        parts.append("{} {} ({})".format(c["kind"], c["code"], c["name"]))
    return "; ".join(parts) or "none"


def render_answer(state):
    """Deterministic answer built ONLY from retrieved facts and DB results."""
    decision = state.get("decision")

    if decision == "unknown_code":
        return (
            "I couldn't find code {} in the manual. "
            "Please double-check the code shown on the display."
        ).format(state.get("typed_code"))

    if decision == "give_up":
        return (
            "I couldn't narrow this down from the information given. "
            "Closest matches: {}. I recommend escalating to a technician."
        ).format(_closest(state.get("candidates", [])))

    if decision == "unverified":
        return (
            "I found a possible match but couldn't verify it. "
            "Please escalate to a technician."
        )

    f = state["fault"]
    lines = [
        "{} {}: {} (manual page {})".format(
            f["kind"].capitalize(), f["code"], f["name"], f["page"]
        ),
        "Cause: " + f["cause"],
        "What to do: " + f["action"],
        "",
    ]
    pr = state.get("parts_result") or {}
    if not pr.get("mapped"):
        lines.append("No replacement part is mapped to this code in the parts database.")
    else:
        lines.append("Related spare parts:")
        for p in pr["parts"]:
            lines.append(_part_line(p))
    return "\n".join(lines)
