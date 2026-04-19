"""Markov chain multi-touch attribution with the removal-effect estimator.

Shao & Li (2011) introduced the removal-effect view of attribution: build a
first-order Markov chain over channels with absorbing ``converted`` and
``null`` states, then estimate each channel's contribution by measuring the
drop in conversion probability when that channel is removed from the graph.

This implementation supports order-1 transitions. Higher orders are
supported by collapsing the last ``order`` touches into a composite state
label.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

_START = "__start__"
_CONVERTED = "__converted__"
_NULL = "__null__"


class MarkovAttribution:
    """First-order (or higher) Markov-chain attribution.

    Parameters
    ----------
    order:
        Number of previous channels aggregated into the state label. ``1``
        recovers the textbook Shao-Li estimator.
    """

    def __init__(self, order: int = 1):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.channels_: list[str] = []
        self.transition_: pd.DataFrame | None = None
        self.base_conv_rate_: float | None = None
        self.removal_effect_: pd.Series | None = None

    # ------------------------------------------------------------------ build

    def _state(self, history: list[str]) -> str:
        tail = history[-self.order :]
        return ">".join(tail) if tail else _START

    def _iter_paths(self, journeys: pd.DataFrame) -> Iterable[tuple[list[str], bool]]:
        journeys = journeys.sort_values(["user_id", "timestamp"])
        for _, grp in journeys.groupby("user_id", sort=False):
            path = grp["channel"].tolist()
            converted = bool(grp["converted"].max() == 1)
            yield path, converted

    def fit(self, journeys: pd.DataFrame) -> "MarkovAttribution":
        """Estimate the empirical transition matrix from journeys."""

        self.channels_ = sorted(journeys["channel"].dropna().unique().tolist())
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        n_total = 0
        n_converted = 0
        for path, converted in self._iter_paths(journeys):
            n_total += 1
            if converted:
                n_converted += 1
            state = self._state([])  # start state
            history: list[str] = []
            for ch in path:
                next_history = history + [ch]
                next_state = self._state(next_history)
                counts[state][next_state] += 1
                state = next_state
                history = next_history
            terminal = _CONVERTED if converted else _NULL
            counts[state][terminal] += 1

        # Build transition matrix
        states = set(counts.keys())
        for tgt in list(counts.values()):
            states.update(tgt.keys())
        states.update({_CONVERTED, _NULL})
        state_list = sorted(states)

        matrix = pd.DataFrame(0.0, index=state_list, columns=state_list)
        for src, tgt_map in counts.items():
            total = sum(tgt_map.values())
            if total == 0:
                continue
            for tgt, c in tgt_map.items():
                matrix.at[src, tgt] = c / total
        # Absorbing states map to themselves.
        matrix.at[_CONVERTED, _CONVERTED] = 1.0
        matrix.at[_NULL, _NULL] = 1.0

        self.transition_ = matrix
        self.base_conv_rate_ = (
            self._conversion_probability(matrix) if n_total else 0.0
        )
        self.removal_effect_ = self._compute_removal_effects()
        return self

    # ------------------------------------------------------------- absorbing

    @staticmethod
    def _conversion_probability(matrix: pd.DataFrame) -> float:
        """Absorbing-chain probability of reaching ``_CONVERTED`` from start.

        Uses the fundamental matrix ``N = (I - Q)^-1`` on transient states,
        then the absorbing probability matrix ``B = N R``.
        """

        absorbing = [_CONVERTED, _NULL]
        transient = [s for s in matrix.index if s not in absorbing]
        if _START not in transient:
            return 0.0
        q = matrix.loc[transient, transient].to_numpy()
        r = matrix.loc[transient, absorbing].to_numpy()
        i_mat = np.eye(q.shape[0])
        try:
            n_mat = np.linalg.inv(i_mat - q)
        except np.linalg.LinAlgError:
            n_mat = np.linalg.pinv(i_mat - q)
        b_mat = n_mat @ r
        idx_start = transient.index(_START)
        idx_conv = absorbing.index(_CONVERTED)
        return float(b_mat[idx_start, idx_conv])

    def _compute_removal_effects(self) -> pd.Series:
        if self.transition_ is None or self.base_conv_rate_ is None:
            raise RuntimeError("fit() must be called before removal effects.")
        base = self.base_conv_rate_
        effects: dict[str, float] = {}
        for ch in self.channels_:
            removed = self._remove_channel(self.transition_, ch)
            rate = self._conversion_probability(removed)
            effects[ch] = base - rate
        s = pd.Series(effects).clip(lower=0.0)
        total = s.sum()
        if total > 0:
            s = s / total
        return s.reindex(self.channels_).fillna(0.0)

    @staticmethod
    def _remove_channel(matrix: pd.DataFrame, channel: str) -> pd.DataFrame:
        """Redirect transitions through ``channel`` into the ``_NULL`` state.

        Any state whose label contains the channel name is treated as absent.
        """

        def contains(state: str) -> bool:
            if state in (_CONVERTED, _NULL, _START):
                return False
            return channel in state.split(">")

        removed = matrix.copy()
        for st in removed.index:
            if contains(st):
                removed.loc[st] = 0.0
                removed.at[st, _NULL] = 1.0
            else:
                # Zero out probability mass flowing INTO states that contain the channel.
                bad_cols = [c for c in removed.columns if contains(c)]
                if bad_cols:
                    mass = removed.loc[st, bad_cols].sum()
                    removed.loc[st, bad_cols] = 0.0
                    if mass > 0:
                        removed.at[st, _NULL] += mass
        # Maintain absorbing states
        removed.at[_CONVERTED, _CONVERTED] = 1.0
        removed.at[_NULL, _NULL] = 1.0
        return removed

    # ------------------------------------------------------------- attribute

    def attribute(self, journeys: pd.DataFrame) -> pd.Series:
        """Return per-channel credit scaled to the number of conversions."""

        if self.removal_effect_ is None:
            self.fit(journeys)
        assert self.removal_effect_ is not None
        n_conv = int((journeys.groupby("user_id")["converted"].max() == 1).sum())
        credit = self.removal_effect_ * n_conv
        credit.name = "credit"
        return credit
