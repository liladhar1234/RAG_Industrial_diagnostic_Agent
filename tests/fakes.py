def _ev(code, kind, name, cause, action, page):
    return {
        "code": code,
        "kind": kind,
        "name": name,
        "cause": cause,
        "action": action,
        "page": page,
    }


EVENTS = {
    "5081": _ev(
        "5081", "fault", "Auxiliary fan broken",
        "An auxiliary cooling fan is stuck or disconnected.",
        "Check auxiliary fan(s) and connection(s). Replace fan if faulty.",
        423,
    ),
    "4290": _ev(
        "4290", "fault", "Cooling",
        "Drive module temperature is excessive.",
        "Check ambient temperature and cooling air flow.",
        422,
    ),
    "4210": _ev(
        "4210", "fault", "IGBT overtemperature",
        "Estimated IGBT temperature is excessive.",
        "Check ambient conditions and fan operation.",
        422,
    ),
    "A4A9": _ev(
        "A4A9", "warning", "Cooling",
        "Drive module temperature is excessive.",
        "Check ambient temperature and cooling air flow.",
        412,
    ),
    "2330": _ev(
        "2330", "fault", "Earth leakage",
        "Load unbalance, typically an earth fault.",
        "Check motor cable insulation.",
        421,
    ),
    "7192": _ev(
        "7192", "fault", "BC IGBT excess temperature",
        "Brake chopper IGBT too hot.",
        "Let chopper cool down.",
        427,
    ),
}


class FakeRetriever:
    """Same interface the real retriever must implement."""

    def get_event(self, code):
        return EVENTS.get(code.upper())

    def search_events(self, query, k=5):
        q = query.upper()
        for code, ev in EVENTS.items():
            if code in q:
                others = [e for c, e in EVENTS.items() if c != code][:2]
                hits = [dict(ev, score=0.95)]
                for o, s in zip(others, (0.30, 0.20)):
                    hits.append(dict(o, score=s))
                return hits
        # no code in the query: three near-tied candidates (ambiguous on purpose)
        return [
            dict(EVENTS["4210"], score=0.62),
            dict(EVENTS["4290"], score=0.60),
            dict(EVENTS["A4A9"], score=0.59),
        ]
