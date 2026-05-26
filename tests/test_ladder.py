from quantbots.strategies.ladder import (
    attach_ladder_fields,
    measurable_key,
    parse_threshold,
)


def test_parse_exceeds():
    assert parse_threshold("Will Brent crude exceed $70 on 2026-06-30?") == (70.0, "exceeds")


def test_parse_below():
    assert parse_threshold("Will CPI be below 3.5% in June?") == (3.5, "below")


def test_parse_with_thousands_separator():
    assert parse_threshold("Will the index be above 1,200?") == (1200.0, "exceeds")


def test_parse_none_when_no_threshold():
    assert parse_threshold("Who wins the election?") is None


def test_measurable_key_collapses_strikes():
    q1 = {"question": "Will Brent crude exceed $62 on 2026-06-30?"}
    q2 = {"question": "Will Brent crude exceed $70 on 2026-06-30?"}
    assert measurable_key(q1) == measurable_key(q2)


def test_attach_ladder_fields():
    m = attach_ladder_fields({"id": "x", "question": "Will gold exceed $2500?"})
    assert m["threshold"] == 2500.0
    assert m["direction"] == "exceeds"
