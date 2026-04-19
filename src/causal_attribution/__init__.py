"""Causal multi-touch attribution.

This package contrasts heuristic attribution baselines with a causal
DAG-based estimator, using a synthetic data-generating process where the
true per-channel effects are known.

Public API
----------
- Heuristics: ``first_touch``, ``last_touch``, ``linear``, ``time_decay``,
  ``u_shaped``.
- Markov chain with removal effect: :class:`MarkovAttribution`.
- Shapley-value attribution: :func:`shapley_attribution`.
- Causal DAG estimator: :class:`CausalAttributor`.
- Synthetic journey generator: :func:`generate_synthetic_journeys`.
- Side-by-side comparison: :func:`compare_methods`.
"""

from .heuristics import (
    first_touch,
    last_touch,
    linear,
    time_decay,
    u_shaped,
)
from .markov import MarkovAttribution
from .shapley import shapley_attribution
from .causal_dag import CausalAttributor
from .data import generate_synthetic_journeys
from .compare import compare_methods

__all__ = [
    "first_touch",
    "last_touch",
    "linear",
    "time_decay",
    "u_shaped",
    "MarkovAttribution",
    "shapley_attribution",
    "CausalAttributor",
    "generate_synthetic_journeys",
    "compare_methods",
]

__version__ = "0.1.0"
