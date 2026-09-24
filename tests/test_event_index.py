from agent.event_index import EventIndex, parse_event

CHUNK_5081 = {
    "id": "ACS580-fault-5081",
    "text": (
        "ACS580 > fault_tracing > 5081\n"
        "Fault 5081: Auxiliary fan broken\n"
        "Cause: An auxiliary cooling fan\n(connected to the fan\n"
        "connectors on the control unit)\nis stuck or disconnected.\n"
        "What to do: Check the auxiliary code.\n"
        "Check auxiliary fan(s) and connection(s).\nReplace fan if faulty.\n"
        "Make sure the front cover of the drive is\nin place and tightened.\n"
        "Reboot the control unit (using parameter\n"
        "96.08 Control board boot) or by cycling\npower.\n"
        "Related sub-codes:\n"
        "  - 0001: Auxiliary fan 1 broken.\n"
        "  - 0002: Auxiliary fan 2 broken."
    ),
    "metadata": {"manual": "ACS580", "section": "fault_tracing",
                 "kind": "fault", "code": "5081", "page": 423},
}

CHUNK_1080 = {
    "id": "ACS580-fault-1080",
    "text": (
        "ACS580 > fault_tracing > 1080\n"
        "Fault 1080: Backup/Restore\ntimeout\n"
        "Cause: Panel or PC tool has failed to\ncommunicate with the drive\n"
        "when backup was being made\nor restored.\n"
        "What to do: Request backup or restore again."
    ),
    "metadata": {"manual": "ACS580", "section": "fault_tracing",
                 "kind": "fault", "code": "1080", "page": 420},
}

CHUNK_PARAM = {
    "id": "ACS580-param-01.01",
    "text": "ACS580 > parameter_list > 01.01\nParameter 01.01: Motor speed used\n",
    "metadata": {"manual": "ACS580", "section": "parameter_list",
                 "code": "01.01", "page": 161},
}


def test_parses_all_fields_of_a_fault_with_sub_codes():
    ev = parse_event(CHUNK_5081)
    assert ev["code"] == "5081" and ev["kind"] == "fault" and ev["page"] == 423
    assert ev["name"] == "Auxiliary fan broken"
    assert ev["cause"].startswith("An auxiliary cooling fan (connected")
    assert "Replace fan if faulty." in ev["action"]
    assert "Related sub-codes" not in ev["action"]
    assert [s["aux_code"] for s in ev["sub_codes"]] == ["0001", "0002"]
    assert ev["sub_codes"][1]["cause"] == "Auxiliary fan 2 broken."


def test_wrapped_name_is_rejoined():
    assert parse_event(CHUNK_1080)["name"] == "Backup/Restore timeout"


def test_missing_cause_gives_empty_string_not_a_crash():
    chunk = {
        "id": "ACS580-fault-9999",
        "text": "ACS580 > fault_tracing > 9999\nFault 9999: Test\nWhat to do: Do X.",
        "metadata": {"section": "fault_tracing", "kind": "fault",
                     "code": "9999", "page": 1},
    }
    ev = parse_event(chunk)
    assert ev["cause"] == "" and ev["action"] == "Do X." and ev["sub_codes"] == []


def test_index_ignores_parameter_chunks_and_looks_up_case_insensitively():
    idx = EventIndex([CHUNK_5081, CHUNK_1080, CHUNK_PARAM])
    assert len(idx.events) == 2
    assert idx.get("5081")["name"] == "Auxiliary fan broken"
    assert idx.get(" 5081 ")["code"] == "5081"
    assert idx.get("01.01") is None
    assert idx.get("ZZZZ") is None


def test_code_shared_by_fault_and_warning_prefers_fault_and_is_reported():
    warn = {
        "id": "ACS580-warning-1080",
        "text": "ACS580 > warning_tracing > 1080\nWarning 1080: Dup\nCause: c\nWhat to do: a",
        "metadata": {"section": "warning_tracing", "kind": "warning",
                     "code": "1080", "page": 5},
    }
    idx = EventIndex([warn, CHUNK_1080])
    assert idx.get("1080")["kind"] == "fault"
    assert idx.collisions == ["1080"]
