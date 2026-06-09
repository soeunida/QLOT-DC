"""StaticScale equal-budget accept-only selection (public API)."""

from staticscale.selection import equal_budget_accept_only_select


def test_accepts_better_candidate():
    name, clear = equal_budget_accept_only_select(
        {"a": 7.990, "b": 7.995}, sadnd_ppl=8.00, margin=0.001)
    assert name == "a" and clear is True


def test_rejects_within_margin_falls_back():
    name, clear = equal_budget_accept_only_select(
        {"a": 7.9995}, sadnd_ppl=8.00, margin=0.001)        # only 0.0005 better
    assert name == "sadnd" and clear is False


def test_rejects_worse_candidate():
    name, clear = equal_budget_accept_only_select(
        {"a": 8.05}, sadnd_ppl=8.00, margin=0.001)
    assert name == "sadnd" and clear is False
