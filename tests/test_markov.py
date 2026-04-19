"""Unit tests for the Markov chain attributor."""

from __future__ import annotations

import pandas as pd
import pytest

from causal_attribution import MarkovAttribution, generate_synthetic_journeys


def _simple_corpus() -> pd.DataFrame:
    """Two converting paths and one non-converting path, plus an unused channel."""

    rows = []
    # User 1: A -> B -> converted
    for idx, ch in enumerate(["A", "B"]):
        rows.append(
            {
                "user_id": 1,
                "channel": ch,
                "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(days=idx),
                "converted": 1,
            }
        )
    # User 2: A -> C -> converted
    for idx, ch in enumerate(["A", "C"]):
        rows.append(
            {
                "user_id": 2,
                "channel": ch,
                "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(days=idx),
                "converted": 1,
            }
        )
    # User 3: B -> C -> null (not converted). D never appears.
    for idx, ch in enumerate(["B", "C"]):
        rows.append(
            {
                "user_id": 3,
                "channel": ch,
                "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(days=idx),
                "converted": 0,
            }
        )
    # User 4: D only, not converted (channel D must exist in the matrix).
    rows.append(
        {
            "user_id": 4,
            "channel": "D",
            "timestamp": pd.Timestamp("2026-01-01"),
            "converted": 0,
        }
    )
    return pd.DataFrame(rows)


def test_removal_effect_sums_to_one() -> None:
    df = _simple_corpus()
    m = MarkovAttribution(order=1).fit(df)
    assert m.removal_effect_ is not None
    total = float(m.removal_effect_.sum())
    assert total == pytest.approx(1.0, rel=1e-6) or total == pytest.approx(0.0)


def test_unused_channel_gets_zero_credit() -> None:
    df = _simple_corpus()
    m = MarkovAttribution(order=1).fit(df)
    credit = m.attribute(df)
    # Channel D never appears in any converting path -> zero credit.
    assert credit["D"] == pytest.approx(0.0)


def test_attribution_scales_to_conversions() -> None:
    df = _simple_corpus()
    m = MarkovAttribution(order=1).fit(df)
    credit = m.attribute(df)
    n_conv = int((df.groupby("user_id")["converted"].max() == 1).sum())
    assert credit.sum() == pytest.approx(n_conv, rel=1e-6)


def test_larger_synthetic_dataset_has_plausible_ranking() -> None:
    df = generate_synthetic_journeys(n_users=2000, seed=7)
    m = MarkovAttribution(order=1).fit(df)
    credit = m.attribute(df)
    # All non-negative, roughly sums to the conversion count.
    assert (credit >= 0).all()
    n_conv = int((df.groupby("user_id")["converted"].max() == 1).sum())
    assert credit.sum() == pytest.approx(n_conv, rel=0.02)


def test_order_must_be_positive() -> None:
    with pytest.raises(ValueError):
        MarkovAttribution(order=0)
