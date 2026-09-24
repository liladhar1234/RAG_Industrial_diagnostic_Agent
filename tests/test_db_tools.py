import sqlite3
import pytest
from db import tools


def test_in_stock_sorted_by_lead_time():
    r = tools.check_part_availability("MK-FAN-1001")
    assert r["found"] and r["in_stock"]
    days = [loc["lead_time_days"] for loc in r["locations"]]
    assert days == sorted(days)


def test_catalog_part_with_zero_stock():
    r = tools.check_part_availability("MK-FAN-1002")
    assert r["found"] is True
    assert r["in_stock"] is False


def test_part_not_in_db_is_not_found():
    r = tools.check_part_availability("MK-NOPE-0000")
    assert r["ok"] is True and r["found"] is False


def test_sql_injection_is_rejected_and_table_survives():
    r = tools.check_part_availability("MK-FAN-1001'; DROP TABLE parts;--")
    assert r == {"ok": False, "error": "invalid_part_number_format"}
    assert tools.check_part_availability("MK-FAN-1001")["found"] is True


def test_connection_is_read_only():
    con = tools._connect()
    with pytest.raises(sqlite3.OperationalError):
        con.execute("INSERT INTO warehouses VALUES (99, 'x', 'y', 1)")
    con.close()


def test_fault_with_no_mapped_parts():
    r = tools.parts_for_fault("2330")
    assert r["ok"] is True and r["mapped"] is False and r["parts"] == []


def test_fault_where_all_parts_out_of_stock():
    r = tools.parts_for_fault("7192")
    assert r["mapped"] is True
    assert all(p["in_stock"] is False for p in r["parts"])


def test_fault_parts_ordered_by_priority_and_include_no_inventory_part():
    r = tools.parts_for_fault("5081")
    assert [p["priority"] for p in r["parts"]] == [1, 2, 3]
    legacy = r["parts"][2]
    assert legacy["part"]["part_number"] == "MK-OLD-0001"
    assert legacy["in_stock"] is False


def test_invalid_fault_code_format():
    assert tools.parts_for_fault("9999Z")["error"] == "invalid_fault_code_format"
