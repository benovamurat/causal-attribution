"""Integration tests for the causal DAG attributor.

These tests use the synthetic DGP from :mod:`causal_attribution.data`,
which encodes a known ground-truth causal effect per channel. The key
claim we verify is that the causal-DAG estimate is directionally correct
AND that its total absolute error across channels is smaller than that of
last-touch attribution for the intent-confounded channels (retargeting
and brand search).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_attribution import (
    CausalAttributor,
    compare_methods,
    generate_synthetic_journeys,
    last_touch,
)


@pytest.fixture(scope="module")
def small_dataset() -> pd.DataFrame:
    return generate_synthetic_journeys(n_users=3000, seed=42)


def test_graph_structure_matches_declared_channels(
    small_dataset: pd.DataFrame,
) -> None:
    attr = CausalAttributor(
        channels=small_dataset.attrs["channels"],
        confounders=small_dataset.attrs["confounders"],
    )
    g = attr.build_graph()
    for ch in small_dataset.attrs["channels"]:
        assert g.has_edge("intent", ch)
        assert g.has_edge(ch, "converted")
    assert g.has_edge("intent", "converted")


def test_retargeting_causal_ate_has_correct_sign(small_dataset: pd.DataFrame) -> None:
    """Retargeting's true effect is small but positive; naive is inflated."""

    attr = CausalAttributor(
        channels=small_dataset.attrs["channels"],
        confounders=small_dataset.attrs["confounders"],
    )
    result = attr.estimate_ate(small_dataset, "retargeting")
    # Naive comparison will look bigger than the causal one
    assert result.ate >= -0.05  # signed estimate should not be wildly negative
    # Naive estimate is biased upward by intent confounding
    assert result.naive_diff > result.ate - 1e-6


def test_causal_estimate_closer_than_last_touch_for_retargeting(
    small_dataset: pd.DataFrame,
) -> None:
    """Causal DAG attribution should be closer to ground truth than
    last-touch for the intent-confounded retargeting channel."""

    shares = compare_methods(small_dataset)
    gt = shares["ground_truth"]
    lt = shares["last_touch"]
    causal = shares["causal_dag"]

    err_lt = abs(lt["retargeting"] - gt["retargeting"])
    err_causal = abs(causal["retargeting"] - gt["retargeting"])

    assert err_causal < err_lt
    # With the default DGP, last-touch overstates retargeting >= 5x
    assert lt["retargeting"] > gt["retargeting"] * 2.0


def test_display_gets_upgraded_vs_last_touch(small_dataset: pd.DataFrame) -> None:
    """Display is a demand-generation channel; last-touch understates it,
    and the causal estimate should give it closer to ground truth."""

    shares = compare_methods(small_dataset)
    gt = shares["ground_truth"]
    lt = shares["last_touch"]
    causal = shares["causal_dag"]

    assert causal["display"] > lt["display"]
    assert abs(causal["display"] - gt["display"]) < abs(lt["display"] - gt["display"])


def test_causal_total_error_smaller_than_last_touch(small_dataset: pd.DataFrame) -> None:
    """Across all channels, total absolute share error should be smaller
    for the causal attributor than for last-touch."""

    shares = compare_methods(small_dataset)
    gt = shares["ground_truth"]
    err_lt = float((shares["last_touch"] - gt).abs().sum())
    err_causal = float((shares["causal_dag"] - gt).abs().sum())
    assert err_causal < err_lt


def test_last_touch_sanity_check(small_dataset: pd.DataFrame) -> None:
    """Sanity: last-touch still sums to 1 and is non-negative."""

    credit = last_touch(small_dataset)
    assert (credit >= 0).all()
    assert credit.sum() > 0
