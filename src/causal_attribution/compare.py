"""Side-by-side comparison of heuristic, algorithmic, and causal attributors."""

from __future__ import annotations

import pandas as pd

from .causal_dag import CausalAttributor
from .heuristics import first_touch, last_touch, linear, time_decay, u_shaped
from .markov import MarkovAttribution
from .shapley import shapley_attribution


def _normalize_share(series: pd.Series) -> pd.Series:
    total = series.sum()
    if total <= 0:
        return series * 0.0
    return series / total


def compare_methods(
    journeys: pd.DataFrame,
    *,
    causal_method: str = "dr",
    include_causal: bool = True,
) -> pd.DataFrame:
    """Run every attributor on the same dataset and return a comparison frame.

    Columns:
        - ``first_touch``, ``last_touch``, ``linear``, ``time_decay``,
          ``u_shaped``: heuristic baselines.
        - ``markov``: Shao-Li removal-effect Markov chain.
        - ``shapley``: exact Shapley-value attribution.
        - ``causal_dag``: DoWhy/EconML doubly-robust ATE-weighted credit.
        - ``ground_truth``: the DGP's true per-channel effect on
          conversion probability (when available via ``df.attrs``).
    Each column is expressed as the **share** of total conversions credited
    to that channel (sums to 1 across rows per column). The ground-truth
    column is the share of incremental conversions implied by the true
    treatment effects and exposure counts.
    """

    channels = sorted(journeys["channel"].dropna().unique().tolist())

    results: dict[str, pd.Series] = {}
    results["first_touch"] = first_touch(journeys).reindex(channels).fillna(0.0)
    results["last_touch"] = last_touch(journeys).reindex(channels).fillna(0.0)
    results["linear"] = linear(journeys).reindex(channels).fillna(0.0)
    results["time_decay"] = time_decay(journeys).reindex(channels).fillna(0.0)
    results["u_shaped"] = u_shaped(journeys).reindex(channels).fillna(0.0)

    markov = MarkovAttribution(order=1).fit(journeys)
    results["markov"] = markov.attribute(journeys).reindex(channels).fillna(0.0)

    results["shapley"] = shapley_attribution(journeys).reindex(channels).fillna(0.0)

    if include_causal:
        confounders = journeys.attrs.get("confounders", ["intent"])
        attr = CausalAttributor(channels=channels, confounders=confounders)
        results["causal_dag"] = (
            attr.attribute(journeys, method=causal_method).reindex(channels).fillna(0.0)
        )

    frame = pd.DataFrame(results)
    # Convert each column to a share for interpretability.
    shares = frame.apply(_normalize_share, axis=0)

    # Ground truth: each channel's contribution to total incremental
    # conversions, estimated as (true_effect_c * exposure_count_c).
    if "true_effects" in journeys.attrs and "per_user" in journeys.attrs:
        per_user = journeys.attrs["per_user"]
        truths = journeys.attrs["true_effects"]
        gt = pd.Series(dtype=float)
        for ch in channels:
            col = f"x_{ch}"
            exposure = int(per_user[col].sum()) if col in per_user.columns else 0
            gt[ch] = truths.get(ch, 0.0) * exposure
        shares["ground_truth"] = _normalize_share(gt.reindex(channels).fillna(0.0))

    shares.index.name = "channel"
    return shares.round(4)
