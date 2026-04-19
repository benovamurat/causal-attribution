"""Unit tests for heuristic attributors."""

from __future__ import annotations

import pandas as pd
import pytest

from causal_attribution import first_touch, last_touch, linear, time_decay, u_shaped


@pytest.fixture
def trivial_journey() -> pd.DataFrame:
    """A single converting user with path A -> B -> C."""

    return pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "channel": ["A", "B", "C"],
            "timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03"]
            ),
            "converted": [1, 1, 1],
        }
    )


def test_first_touch_credits_first_channel(trivial_journey: pd.DataFrame) -> None:
    credit = first_touch(trivial_journey)
    assert credit["A"] == pytest.approx(1.0)
    assert credit["B"] == pytest.approx(0.0)
    assert credit["C"] == pytest.approx(0.0)


def test_last_touch_credits_last_channel(trivial_journey: pd.DataFrame) -> None:
    credit = last_touch(trivial_journey)
    assert credit["A"] == pytest.approx(0.0)
    assert credit["B"] == pytest.approx(0.0)
    assert credit["C"] == pytest.approx(1.0)


def test_linear_splits_evenly(trivial_journey: pd.DataFrame) -> None:
    credit = linear(trivial_journey)
    for ch in ("A", "B", "C"):
        assert credit[ch] == pytest.approx(1 / 3)


def test_time_decay_weights_recent_higher(trivial_journey: pd.DataFrame) -> None:
    credit = time_decay(trivial_journey, halflife_days=1.0)
    # Credit should rise monotonically from A -> B -> C with 1-day halflife.
    assert credit["A"] < credit["B"] < credit["C"]
    assert credit.sum() == pytest.approx(1.0)


def test_u_shaped_gives_weight_to_endpoints(trivial_journey: pd.DataFrame) -> None:
    credit = u_shaped(trivial_journey, first_weight=0.4, last_weight=0.4)
    assert credit["A"] == pytest.approx(0.4)
    assert credit["C"] == pytest.approx(0.4)
    assert credit["B"] == pytest.approx(0.2)


def test_nonconverting_users_get_no_credit() -> None:
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "channel": ["A", "B", "A", "B"],
            "timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
            ),
            "converted": [1, 1, 0, 0],
        }
    )
    assert first_touch(df).sum() == pytest.approx(1.0)
    assert last_touch(df).sum() == pytest.approx(1.0)
    assert linear(df).sum() == pytest.approx(1.0)


def test_u_shaped_rejects_invalid_weights() -> None:
    df = pd.DataFrame(
        {
            "user_id": [1],
            "channel": ["A"],
            "timestamp": pd.to_datetime(["2026-01-01"]),
            "converted": [1],
        }
    )
    with pytest.raises(ValueError):
        u_shaped(df, first_weight=0.7, last_weight=0.6)


def test_time_decay_rejects_nonpositive_halflife(
    trivial_journey: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):
        time_decay(trivial_journey, halflife_days=0.0)


def test_single_touch_journey_gets_full_credit() -> None:
    df = pd.DataFrame(
        {
            "user_id": [1],
            "channel": ["A"],
            "timestamp": pd.to_datetime(["2026-01-01"]),
            "converted": [1],
        }
    )
    for fn in (first_touch, last_touch, linear, u_shaped):
        credit = fn(df)
        assert credit["A"] == pytest.approx(1.0)
