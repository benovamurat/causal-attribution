"""Causal DAG attribution using backdoor adjustment and doubly-robust learners.

The attribution problem is recast as a per-channel causal estimation
problem: for each channel ``c``, treat ``X_c`` (was the user exposed) as
the treatment, ``Y`` (did the user convert) as the outcome, and the set of
confounders as the backdoor adjustment set. The remaining channels act as
additional covariates so that channel-to-channel correlations are
partialled out.

We use DoWhy's four-step workflow (model, identify, estimate, refute) with
EconML's Doubly Robust Learner as the numerical estimator. This combines a
propensity model (``X ~ Z``) with an outcome model (``Y ~ X, Z``); the
estimator is consistent if either one is correctly specified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

# DoWhy / EconML imports are heavy; keep them optional at import time so
# the rest of the package stays lightweight.
try:  # pragma: no cover - exercised in tests via CausalAttributor
    from dowhy import CausalModel
except Exception as exc:  # noqa: BLE001
    CausalModel = None  # type: ignore[assignment]
    _DOWHY_IMPORT_ERROR: Exception | None = exc
else:
    _DOWHY_IMPORT_ERROR = None

try:
    from econml.dr import LinearDRLearner
except Exception as exc:  # noqa: BLE001
    LinearDRLearner = None  # type: ignore[assignment]
    _ECONML_IMPORT_ERROR: Exception | None = exc
else:
    _ECONML_IMPORT_ERROR = None


@dataclass
class ATEResult:
    """Container for a single-channel causal estimate."""

    channel: str
    ate: float
    method: str
    naive_diff: float
    confounders: list[str] = field(default_factory=list)
    adjustment_set: list[str] = field(default_factory=list)
    confidence_interval: tuple[float, float] | None = None


class CausalAttributor:
    """Per-channel causal attribution on a touchpoint DataFrame.

    Parameters
    ----------
    channels:
        List of channel names to estimate ATEs for.
    confounders:
        Names of observed variables that act as backdoor adjustment
        variables. In the synthetic DGP these are ``["intent"]``; in a
        production setting they would include pre-exposure behavioral
        features (historical purchases, days since last visit, etc.).
    other_channels_as_covariates:
        If ``True`` (default) other channel exposure indicators are added
        to the adjustment set when estimating a given channel's effect.
        This partially isolates a channel's effect from that of correlated
        channels.
    """

    def __init__(
        self,
        channels: Iterable[str],
        confounders: Iterable[str],
        *,
        other_channels_as_covariates: bool = True,
    ):
        self.channels = list(channels)
        self.confounders = list(confounders)
        self.other_channels_as_covariates = other_channels_as_covariates
        self.graph_: nx.DiGraph | None = None
        self._per_user_cache_: pd.DataFrame | None = None

    # ---------------------------------------------------------- graph build

    def build_graph(self) -> nx.DiGraph:
        """Return the DAG as a :class:`networkx.DiGraph`.

        Edges: every confounder points at every channel and at the outcome.
        Every channel points at the outcome. Channels have no direct edges
        among each other - they share confounders but are not direct causes
        of each other in the default specification.
        """

        g = nx.DiGraph()
        outcome = "converted"
        for u in self.confounders:
            g.add_node(u, kind="confounder")
            for ch in self.channels:
                g.add_edge(u, ch)
            g.add_edge(u, outcome)
        for ch in self.channels:
            g.add_node(ch, kind="treatment")
            g.add_edge(ch, outcome)
        g.add_node(outcome, kind="outcome")
        self.graph_ = g
        return g

    def graph_as_gml(self) -> str:
        """Return a DoWhy-compatible GML string for the DAG."""

        g = self.graph_ or self.build_graph()
        lines = ["digraph {"]
        for src, dst in g.edges():
            lines.append(f"    {src} -> {dst};")
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------- helpers

    def _prepare_per_user(self, journeys: pd.DataFrame) -> pd.DataFrame:
        """Return a user-level DataFrame with one exposure indicator per channel."""

        if "per_user" in journeys.attrs:
            # Data generator provides the per-user snapshot directly.
            per_user = journeys.attrs["per_user"].copy()
            # Confirm all required columns exist.
            missing = [
                c
                for c in self.confounders
                if c not in per_user.columns
            ]
            if missing:
                raise KeyError(
                    "Confounders missing in per_user table: " + ", ".join(missing)
                )
            for ch in self.channels:
                col = f"x_{ch}"
                if col not in per_user.columns:
                    per_user[col] = 0
            return per_user

        # Fall back to building it from the touchpoint frame. In this case
        # we can't recover the true intent without it being on the frame.
        rows = journeys.copy()
        # Collapse to user level
        users = rows["user_id"].unique()
        per_user = pd.DataFrame({"user_id": users})
        conv = (
            rows.groupby("user_id")["converted"].max().reindex(users).fillna(0).astype(int)
        )
        per_user["converted"] = conv.to_numpy()
        for ch in self.channels:
            flag = (
                rows.assign(_is=(rows["channel"] == ch).astype(int))
                .groupby("user_id")["_is"]
                .max()
                .reindex(users)
                .fillna(0)
                .astype(int)
            )
            per_user[f"x_{ch}"] = flag.to_numpy()
        for c in self.confounders:
            if c in rows.columns:
                per_user[c] = (
                    rows.groupby("user_id")[c].first().reindex(users).to_numpy()
                )
            else:
                raise KeyError(
                    f"Confounder '{c}' not found in journeys frame. Attach it "
                    f"either as a column on the touchpoint frame or via "
                    f"df.attrs['per_user']."
                )
        return per_user

    # --------------------------------------------------------- estimation

    def estimate_ate(
        self,
        journeys: pd.DataFrame,
        treatment_channel: str,
        *,
        method: str = "dr",
    ) -> ATEResult:
        """Estimate the average treatment effect of one channel on conversion.

        Parameters
        ----------
        journeys:
            Journey-level DataFrame; typically produced by
            :func:`causal_attribution.data.generate_synthetic_journeys`.
        treatment_channel:
            Channel to estimate the ATE for.
        method:
            ``"dr"`` uses EconML's Linear DR Learner. ``"backdoor"`` uses
            DoWhy's builtin propensity score stratification.
        """

        if treatment_channel not in self.channels:
            raise ValueError(
                f"Channel '{treatment_channel}' not in declared channels."
            )
        self.build_graph()

        per_user = self._prepare_per_user(journeys)
        treatment_col = f"x_{treatment_channel}"
        outcome_col = "converted"
        common_causes = list(self.confounders)
        if self.other_channels_as_covariates:
            common_causes += [
                f"x_{c}"
                for c in self.channels
                if c != treatment_channel
            ]

        naive = float(
            per_user.loc[per_user[treatment_col] == 1, outcome_col].mean()
            - per_user.loc[per_user[treatment_col] == 0, outcome_col].mean()
        )

        if method == "dr":
            ate, ci = self._estimate_dr(per_user, treatment_col, outcome_col, common_causes)
            method_name = "EconML LinearDRLearner (doubly robust)"
        elif method == "backdoor":
            ate, ci = self._estimate_backdoor(per_user, treatment_col, outcome_col, common_causes)
            method_name = "DoWhy backdoor.propensity_score_stratification"
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'dr' or 'backdoor'.")

        return ATEResult(
            channel=treatment_channel,
            ate=float(ate),
            method=method_name,
            naive_diff=naive,
            confounders=list(self.confounders),
            adjustment_set=list(common_causes),
            confidence_interval=ci,
        )

    # ---------------------------------------------------- internal methods

    def _estimate_dr(
        self,
        per_user: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        common_causes: list[str],
    ) -> tuple[float, tuple[float, float] | None]:
        if LinearDRLearner is None:  # pragma: no cover
            raise ImportError(
                "EconML is required for the 'dr' method. "
                f"Original import error: {_ECONML_IMPORT_ERROR}"
            )
        y = per_user[outcome_col].to_numpy()
        t = per_user[treatment_col].to_numpy()
        if len(common_causes) == 0:
            x = np.zeros((len(per_user), 1))
        else:
            x = per_user[common_causes].to_numpy().astype(float)

        learner = LinearDRLearner(random_state=0)
        learner.fit(y, t, X=x, W=None)
        ate = float(learner.ate(X=x, T0=0, T1=1))
        ci: tuple[float, float] | None = None
        try:
            lo, hi = learner.ate_interval(X=x, T0=0, T1=1, alpha=0.05)
            ci = (float(lo), float(hi))
        except Exception:  # noqa: BLE001
            ci = None
        return ate, ci

    def _estimate_backdoor(
        self,
        per_user: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        common_causes: list[str],
    ) -> tuple[float, tuple[float, float] | None]:
        if CausalModel is None:  # pragma: no cover
            raise ImportError(
                "DoWhy is required for the 'backdoor' method. "
                f"Original import error: {_DOWHY_IMPORT_ERROR}"
            )
        # Build a DAG string listing only the variables we pass to DoWhy.
        lines = ["digraph {"]
        for z in common_causes:
            lines.append(f"    {z} -> {treatment_col};")
            lines.append(f"    {z} -> {outcome_col};")
        lines.append(f"    {treatment_col} -> {outcome_col};")
        lines.append("}")
        graph = "\n".join(lines)

        model = CausalModel(
            data=per_user,
            treatment=treatment_col,
            outcome=outcome_col,
            common_causes=common_causes,
            graph=graph,
        )
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified,
            method_name="backdoor.propensity_score_stratification",
        )
        return float(estimate.value), None

    # ---------------------------------------------- batch attribution utils

    def attribute(
        self,
        journeys: pd.DataFrame,
        *,
        method: str = "dr",
    ) -> pd.Series:
        """Return per-channel credit proportional to estimated ATEs.

        Credits are scaled so the total equals the number of converting
        users in ``journeys``. Negative ATE estimates are floored at zero
        before scaling because a negative attribution is not a meaningful
        budget share.
        """

        ates = {}
        for ch in self.channels:
            ates[ch] = self.estimate_ate(journeys, ch, method=method).ate

        per_user = self._prepare_per_user(journeys)
        n_converters = int(per_user["converted"].sum())

        raw = pd.Series(ates).clip(lower=0.0)
        # Weight each channel's ATE by the number of exposed users: that's
        # the channel's contribution to total conversions.
        exposure_count = pd.Series(
            {ch: int(per_user[f"x_{ch}"].sum()) for ch in self.channels}
        )
        weighted = raw * exposure_count
        total = weighted.sum()
        if total > 0:
            credit = weighted * (n_converters / total)
        else:
            credit = weighted
        credit.name = "credit"
        return credit.reindex(self.channels).astype(float)
