import sqlite3, os

DB_PATH = os.getenv("MOCK_DB_PATH", "data/db/parts.sqlite")

SCHEMA = """
DROP TABLE IF EXISTS fault_parts;
DROP TABLE IF EXISTS inventory;
DROP TABLE IF EXISTS parts;
DROP TABLE IF EXISTS warehouses;

CREATE TABLE parts (
    part_number TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    category    TEXT NOT NULL,
    unit_price  REAL NOT NULL
);
CREATE TABLE warehouses (
    warehouse_id   INTEGER PRIMARY KEY,
    name           TEXT NOT NULL,
    city           TEXT NOT NULL,
    lead_time_days INTEGER NOT NULL
);
CREATE TABLE inventory (
    part_number  TEXT    NOT NULL REFERENCES parts(part_number),
    warehouse_id INTEGER NOT NULL REFERENCES warehouses(warehouse_id),
    qty_on_hand  INTEGER NOT NULL CHECK (qty_on_hand >= 0),
    PRIMARY KEY (part_number, warehouse_id)
);
CREATE TABLE fault_parts (
    fault_code  TEXT NOT NULL,
    part_number TEXT NOT NULL REFERENCES parts(part_number),
    priority    INTEGER NOT NULL,
    note        TEXT,
    PRIMARY KEY (fault_code, part_number)
);
"""

PARTS = [
    ("MK-FAN-1001", "Auxiliary cooling fan, frame R1-R3", "cooling", 42.50),
    ("MK-FAN-1002", "Auxiliary cooling fan, frame R4-R6", "cooling", 58.00),
    ("MK-FAN-1003", "Main cooling fan, drive module", "cooling", 96.00),
    ("MK-FLT-3001", "Air filter kit", "cooling", 18.00),
    ("MK-BRK-2001", "Brake resistor 100 ohm", "braking", 120.00),
    ("MK-BRK-2002", "Brake chopper unit", "braking", 310.00),
    ("MK-COM-6001", "Fieldbus adapter module", "communication", 145.00),
    ("MK-STO-7001", "Safety relay for STO circuit", "safety", 88.00),
    ("MK-4521-A", "Control panel, basic", "control", 64.00),
    ("MK-4521-B", "Control panel, assistant", "control", 92.00),
    ("MK-OLD-0001", "Legacy fan assembly (discontinued)", "cooling", 30.00),
]
WAREHOUSES = [
    (1, "Pune Central", "Pune", 1),
    (2, "Mumbai Hub", "Mumbai", 2),
    (3, "Frankfurt DC", "Frankfurt", 6),
]
INVENTORY = [
    ("MK-FAN-1001", 1, 12), ("MK-FAN-1001", 3, 4),
    ("MK-FAN-1002", 2, 0),                                  # in catalog, zero stock
    ("MK-FAN-1003", 1, 3), ("MK-FAN-1003", 2, 5),
    ("MK-FLT-3001", 1, 40), ("MK-FLT-3001", 2, 25), ("MK-FLT-3001", 3, 10),
    ("MK-BRK-2001", 2, 2), ("MK-BRK-2001", 3, 7),
    ("MK-BRK-2002", 1, 0), ("MK-BRK-2002", 2, 0), ("MK-BRK-2002", 3, 0),  # out everywhere
    ("MK-COM-6001", 1, 6),
    ("MK-STO-7001", 2, 1),
    ("MK-4521-A", 1, 8), ("MK-4521-B", 1, 5),
    # MK-OLD-0001 has NO inventory rows at all (on purpose)
]
FAULT_PARTS = [
    ("5081", "MK-FAN-1001", 1, "Replace fan if stuck or disconnected"),
    ("5081", "MK-FAN-1002", 2, "Use for larger frames"),
    ("5081", "MK-OLD-0001", 3, "Legacy option"),
    ("4290", "MK-FAN-1003", 1, "Check fan operation first"),
    ("4290", "MK-FLT-3001", 2, "Replace if clogged"),
    ("A4A9", "MK-FAN-1003", 1, "Check fan operation first"),
    ("A4A9", "MK-FLT-3001", 2, "Replace if clogged"),
    ("4210", "MK-FAN-1003", 1, None),
    ("4110", "MK-FAN-1001", 1, "Check auxiliary cooling fan"),
    ("A793", "MK-BRK-2001", 1, "Check resistor before replacing"),
    ("7192", "MK-BRK-2002", 1, "Replace chopper if fault persists"),
    ("A7C1", "MK-COM-6001", 1, "Check cabling before replacing"),
    ("FA81", "MK-STO-7001", 1, None),
    ("A5A0", "MK-STO-7001", 1, None),
    # 2330 (earth leakage) is deliberately unmapped: it is a wiring problem, not a part
]

def build():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO parts VALUES (?,?,?,?)", PARTS)
    con.executemany("INSERT INTO warehouses VALUES (?,?,?,?)", WAREHOUSES)
    con.executemany("INSERT INTO inventory VALUES (?,?,?)", INVENTORY)
    con.executemany("INSERT INTO fault_parts VALUES (?,?,?,?)", FAULT_PARTS)
    con.commit()
    con.close()
    print("built", DB_PATH)

if __name__ == "__main__":
    build()