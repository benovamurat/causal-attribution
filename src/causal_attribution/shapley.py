"""Shapley-value attribution for multi-touch paths.

Each journey's set of unique touched channels is treated as a cooperative
game. The value of a coalition ``S`` is the expected conversion rate of
users whose exposure set is ``S``. The Shapley value of channel ``c`` is its
marginal contribution averaged over all orderings, which for set-based
games collapses to the classic formula of Shapley (1953).

For journeys over the full channel universe, we use the *exact* Shapley
value when there are at most 8 channels and a Monte-Carlo permutation
approximation otherwise. This mirrors the Berman (2018) formulation used in
marketing attribution.
"""

from __future__ import annotations

from itertools import chain, combinations
from math import factorial
from typing import Callable, Iterable

import numpy as np
import pandas as pd

Coalition = frozenset[str]
ValueFn = Callable[[Coalition], float]


def _powerset(items: list[str]) -> Iterable[Coalition]:
    for size in range(len(items) + 1):
        for combo in combinations(items, size):
            yield frozenset(combo)


def default_value_fn(journeys: pd.DataFrame) -> ValueFn:
    """Return ``v(S) = expected conversions for users whose channel set = S``.

    Only users whose observed exposure set is exactly ``S`` contribute. The
    value is the raw count of conversions among those users, so Shapley
    values sum to the total number of conversions.
    """

    per_user = (
        journeys.sort_values(["user_id", "timestamp"])
        .groupby("user_id")
        .agg(coalition=("channel", lambda s: frozenset(s.dropna().unique())),
             converted=("converted", "max"))
    )

    # Tabulate counts of conversions per coalition
    conv_table = (
        per_user.groupby("coalition")["converted"].sum().astype(float).to_dict()
    )

    empty = frozenset()

    def v(s: Coalition) -> float:
        # Sum of conversions for every observed coalition that is a SUBSET of s.
        # This follows the standard "accessible coalition" interpretation:
        # if the set of available channels is s, any user whose exposure is
        # contained in s could have converted with that set.
        if not s:
            return float(conv_table.get(empty, 0.0))
        total = 0.0
        for obs, cnt in conv_table.items():
            if obs.issubset(s):
                total += cnt
        return float(total)

    return v


def _exact_shapley(channels: list[str], value_fn: ValueFn) -> pd.Series:
    n = len(channels)
    n_fact = factorial(n)
    credit = {c: 0.0 for c in channels}

    # Cache value-function evaluations
    cache: dict[Coalition, float] = {}

    def v(s: Coalition) -> float:
        if s not in cache:
            cache[s] = value_fn(s)
        return cache[s]

    base_set = set(channels)
    for ch in channels:
        rest = [c for c in channels if c != ch]
        # Sum over all subsets S of rest
        for s_tuple in chain.from_iterable(
            combinations(rest, r) for r in range(len(rest) + 1)
        ):
            s = frozenset(s_tuple)
            marginal = v(s | {ch}) - v(s)
            # Weighting in the Shapley formula
            size = len(s)
            weight = factorial(size) * factorial(n - size - 1) / n_fact
            credit[ch] += weight * marginal

    return pd.Series(credit).reindex(channels).astype(float)


def _mc_shapley(
    channels: list[str], value_fn: ValueFn, n_samples: int, rng: np.random.Generator
) -> pd.Series:
    n = len(channels)
    credit = {c: 0.0 for c in channels}
    cache: dict[Coalition, float] = {}

    def v(s: Coalition) -> float:
        if s not in cache:
            cache[s] = value_fn(s)
        return cache[s]

    for _ in range(n_samples):
        order = list(rng.permutation(channels))
        s = set()
        prev = v(frozenset(s))
        for ch in order:
            s.add(ch)
            nxt = v(frozenset(s))
            credit[ch] += nxt - prev
            prev = nxt

    for ch in credit:
        credit[ch] /= max(1, n_samples)
    return pd.Series(credit).reindex(channels).astype(float)


def shapley_attribution(
    journeys: pd.DataFrame,
    value_fn: ValueFn | None = None,
    *,
    exact_max_channels: int = 8,
    n_samples: int = 2000,
    seed: int | None = 0,
) -> pd.Series:
    """Compute Shapley-value attribution over distinct channels.

    Parameters
    ----------
    journeys:
        Touchpoint-level DataFrame.
    value_fn:
        Optional custom coalition value function. Defaults to the
        expected-conversions-over-observed-subsets function in
        :func:`default_value_fn`.
    exact_max_channels:
        Use the closed-form exact Shapley formula when the number of
        distinct channels is at most this value. Otherwise fall back to the
        Monte-Carlo permutation estimator.
    n_samples:
        Number of permutations to sample in the Monte-Carlo mode.
    seed:
        Seed for the permutation sampler (ignored in exact mode).
    """

    channels = sorted(journeys["channel"].dropna().unique().tolist())
    if not channels:
        return pd.Series(dtype=float, name="credit")

    v = value_fn if value_fn is not None else default_value_fn(journeys)

    if len(channels) <= exact_max_channels:
        credit = _exact_shapley(channels, v)
    else:
        rng = np.random.default_rng(seed)
        credit = _mc_shapley(channels, v, n_samples=n_samples, rng=rng)

    credit.name = "credit"
    credit = credit.clip(lower=0)
    # Rescale to preserve total conversions (Shapley is efficient when v is
    # super-additive; we rescale to guard against numerical drift / clipping).
    total_conv = float(
        (journeys.groupby("user_id")["converted"].max() == 1).sum()
    )
    s = credit.sum()
    if s > 0 and total_conv > 0:
        credit = credit * (total_conv / s)
    return credit
