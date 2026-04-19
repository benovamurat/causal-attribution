"""Heuristic multi-touch attribution baselines.

Each function accepts a journeys :class:`pandas.DataFrame` with at minimum
the columns ``user_id``, ``channel``, ``timestamp``, and ``converted`` and
returns a :class:`pandas.Series` of per-channel credit that sums to the
total number of conversions in the input.

The journeys frame represents touchpoint-level data. Each row is a single
touchpoint. The ``converted`` column is ``1`` for every touchpoint in a
user's path if that user eventually converted, and ``0`` otherwise. This
mirrors how attribution vendors encode outcome labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _converted_journeys(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the touchpoints of converted users, sorted by time."""

    if df.empty:
        return df
    converted = df[df["converted"] == 1].copy()
    converted = converted.sort_values(["user_id", "timestamp"])
    return converted


def _all_channels(df: pd.DataFrame) -> list[str]:
    return sorted(df["channel"].dropna().unique().tolist())


def _empty_credit(channels: list[str]) -> pd.Series:
    return pd.Series(0.0, index=channels, name="credit")


def first_touch(journeys: pd.DataFrame) -> pd.Series:
    """Assign full credit to the first touchpoint in each converting journey."""

    channels = _all_channels(journeys)
    if not channels:
        return pd.Series(dtype=float, name="credit")
    conv = _converted_journeys(journeys)
    if conv.empty:
        return _empty_credit(channels)
    first = conv.groupby("user_id", sort=False).first()
    credit = first["channel"].value_counts().reindex(channels, fill_value=0.0)
    credit.name = "credit"
    return credit.astype(float)


def last_touch(journeys: pd.DataFrame) -> pd.Series:
    """Assign full credit to the last touchpoint in each converting journey."""

    channels = _all_channels(journeys)
    if not channels:
        return pd.Series(dtype=float, name="credit")
    conv = _converted_journeys(journeys)
    if conv.empty:
        return _empty_credit(channels)
    last = conv.groupby("user_id", sort=False).last()
    credit = last["channel"].value_counts().reindex(channels, fill_value=0.0)
    credit.name = "credit"
    return credit.astype(float)


def linear(journeys: pd.DataFrame) -> pd.Series:
    """Divide credit evenly across all touchpoints in each converting journey."""

    channels = _all_channels(journeys)
    if not channels:
        return pd.Series(dtype=float, name="credit")
    conv = _converted_journeys(journeys)
    if conv.empty:
        return _empty_credit(channels)
    sizes = conv.groupby("user_id", sort=False).size().rename("path_len")
    frame = conv.join(sizes, on="user_id")
    frame["weight"] = 1.0 / frame["path_len"]
    credit = frame.groupby("channel")["weight"].sum()
    credit = credit.reindex(channels, fill_value=0.0).astype(float)
    credit.name = "credit"
    return credit


def time_decay(journeys: pd.DataFrame, halflife_days: float = 7.0) -> pd.Series:
    """Give more credit to recent touchpoints using exponential decay.

    Parameters
    ----------
    halflife_days:
        Time (in days) for a touchpoint's weight to halve as it moves further
        from the conversion timestamp. The last touchpoint has weight ``1``.
    """

    if halflife_days <= 0:
        raise ValueError("halflife_days must be positive")
    channels = _all_channels(journeys)
    if not channels:
        return pd.Series(dtype=float, name="credit")
    conv = _converted_journeys(journeys)
    if conv.empty:
        return _empty_credit(channels)

    # ``timestamp`` may be datetime or numeric.
    ts = pd.to_datetime(conv["timestamp"], errors="coerce")
    if ts.isna().any():
        # Fall back to treating timestamps as numeric day offsets.
        numeric = pd.to_numeric(conv["timestamp"], errors="raise")
        conv = conv.assign(_t=numeric.astype(float))
    else:
        # Convert to days since the epoch for subtraction.
        conv = conv.assign(_t=ts.astype("int64") / (1e9 * 60 * 60 * 24))

    last_t = conv.groupby("user_id", sort=False)["_t"].transform("max")
    delta = (last_t - conv["_t"]).clip(lower=0)
    lam = np.log(2) / float(halflife_days)
    raw = np.exp(-lam * delta)
    conv = conv.assign(_raw=raw)

    totals = conv.groupby("user_id", sort=False)["_raw"].transform("sum")
    conv = conv.assign(_w=conv["_raw"] / totals.replace(0, np.nan))
    conv["_w"] = conv["_w"].fillna(0.0)

    # Each converting user contributes exactly 1 conversion in total.
    n_conv = conv["user_id"].nunique()
    per_user_sum = conv.groupby("user_id", sort=False)["_w"].transform("sum")
    # Normalize per-user weight to 1 so total credit equals total conversions.
    safe = per_user_sum.replace(0, np.nan)
    conv["_w"] = (conv["_w"] / safe).fillna(0.0)

    credit = conv.groupby("channel")["_w"].sum()
    credit = credit.reindex(channels, fill_value=0.0).astype(float)

    # Sanity: total credit should equal n_conv (up to numerical tolerance).
    total = credit.sum()
    if total > 0:
        credit *= n_conv / total
    credit.name = "credit"
    return credit


def u_shaped(
    journeys: pd.DataFrame,
    first_weight: float = 0.4,
    last_weight: float = 0.4,
) -> pd.Series:
    """Position-based attribution (a.k.a. U-shaped / bathtub).

    ``first_weight`` and ``last_weight`` control the fraction of credit given
    to the first and last touchpoints respectively. The remainder is split
    evenly across the middle touches. If a journey has only one touchpoint it
    receives the full conversion credit. For two-touch journeys the split is
    ``first_weight`` and ``last_weight`` renormalized to sum to 1.
    """

    if first_weight < 0 or last_weight < 0:
        raise ValueError("first_weight and last_weight must be non-negative")
    if first_weight + last_weight > 1.0 + 1e-9:
        raise ValueError("first_weight + last_weight must not exceed 1")
    channels = _all_channels(journeys)
    if not channels:
        return pd.Series(dtype=float, name="credit")
    conv = _converted_journeys(journeys)
    if conv.empty:
        return _empty_credit(channels)

    middle_weight = max(0.0, 1.0 - first_weight - last_weight)
    frame = conv.reset_index(drop=True)
    frame["_idx"] = frame.groupby("user_id").cumcount()
    sizes = frame.groupby("user_id")["_idx"].transform("max") + 1
    frame["_n"] = sizes

    weights = np.zeros(len(frame), dtype=float)
    idx = frame["_idx"].to_numpy()
    n = frame["_n"].to_numpy()

    # Single-touch
    single_mask = n == 1
    weights[single_mask] = 1.0

    # Two-touch: split according to first/last, renormalized.
    two_mask = n == 2
    denom = first_weight + last_weight
    if denom <= 0:
        # Degenerate: all weight to middle (which is empty) -> fall back to linear
        weights[two_mask] = 0.5
    else:
        weights[(two_mask) & (idx == 0)] = first_weight / denom
        weights[(two_mask) & (idx == 1)] = last_weight / denom

    # >=3 touch
    long_mask = n >= 3
    mids = n - 2
    per_mid = np.where(mids > 0, middle_weight / np.maximum(mids, 1), 0.0)
    weights[(long_mask) & (idx == 0)] = first_weight
    weights[(long_mask) & (idx == n - 1)] = last_weight
    middle_touch = (long_mask) & (idx > 0) & (idx < n - 1)
    weights[middle_touch] = per_mid[middle_touch]

    frame["_w"] = weights
    credit = frame.groupby("channel")["_w"].sum()
    credit = credit.reindex(channels, fill_value=0.0).astype(float)
    credit.name = "credit"
    return credit
