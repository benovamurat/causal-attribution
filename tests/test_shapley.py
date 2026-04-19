"""Unit tests for Shapley-value attribution."""

from __future__ import annotations

import pandas as pd
import pytest

from causal_attribution import generate_synthetic_journeys, shapley_attribution


def test_shapley_on_trivial_single_channel() -> None:
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "channel": ["A", "A", "A"],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "converted": [1, 1, 0],
        }
    )
    credit = shapley_attribution(df)
    assert credit["A"] == pytest.approx(2.0)


def test_shapley_sums_to_total_conversions() -> None:
    df = generate_synthetic_journeys(n_users=1000, seed=3)
    credit = shapley_attribution(df)
    n_conv = int((df.groupby("user_id")["converted"].max() == 1).sum())
    assert credit.sum() == pytest.approx(n_conv, rel=1e-6)


def test_shapley_channels_all_present() -> None:
    df = generate_synthetic_journeys(n_users=500, seed=11)
    credit = shapley_attribution(df)
    for ch in df["channel"].unique():
        assert ch in credit.index
