import sqlite3, os, re
from contextlib import closing

DB_PATH = os.getenv("MOCK_DB_PATH", "data/db/parts.sqlite")
FAULT_RE = re.compile(r"^[0-9A-F]{4}$")
PART_RE = re.compile(r"^MK-[A-Z0-9-]{3,20}$")


def _connect():
    # mode=ro makes writes impossible at the SQLite level
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def check_part_availability(part_number: str) -> dict:
    if not PART_RE.match(part_number):
        return {"ok": False, "error": "invalid_part_number_format"}
    with closing(_connect()) as con:
        part = con.execute(
            "SELECT * FROM parts WHERE part_number = ?", (part_number,)
        ).fetchone()
        if part is None:
            return {"ok": True, "found": False, "part_number": part_number}
        stock = con.execute(
            """SELECT w.name, w.city, w.lead_time_days, i.qty_on_hand
               FROM inventory i JOIN warehouses w USING (warehouse_id)
               WHERE i.part_number = ? AND i.qty_on_hand > 0
               ORDER BY w.lead_time_days""",
            (part_number,),
        ).fetchall()
    return {
        "ok": True,
        "found": True,
        "part": dict(part),
        "in_stock": len(stock) > 0,
        "locations": [dict(r) for r in stock],
    }


def parts_for_fault(fault_code: str) -> dict:
    fault_code = fault_code.strip().upper()
    if not FAULT_RE.match(fault_code):
        return {"ok": False, "error": "invalid_fault_code_format"}
    with closing(_connect()) as con:
        rows = con.execute(
            """SELECT fp.part_number, fp.priority, fp.note
               FROM fault_parts fp JOIN parts p USING (part_number)
               WHERE fp.fault_code = ?
               ORDER BY fp.priority""",
            (fault_code,),
        ).fetchall()
    if not rows:
        return {"ok": True, "fault_code": fault_code, "mapped": False, "parts": []}
    parts = []
    for r in rows:
        avail = check_part_availability(r["part_number"])
        avail["priority"] = r["priority"]
        avail["note"] = r["note"]
        parts.append(avail)
    return {"ok": True, "fault_code": fault_code, "mapped": True, "parts": parts}
