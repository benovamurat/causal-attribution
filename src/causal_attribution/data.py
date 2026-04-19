"""Synthetic data-generating process for causal attribution experiments.

The goal of this module is to produce a journey-level dataset where the
true per-channel causal effects are known. A confounder - ``intent`` -
drives both channel exposure probability and conversion probability. This
is the textbook selection-bias setup that causes standard multi-touch
attribution to overstate the value of channels correlated with intent
(typically retargeting and brand search).

Data-generating process
-----------------------
For each synthetic user:

1. Draw ``intent ~ Beta(a, b)``. Shape parameters are chosen so that the
   distribution is right-skewed (few high-intent users, many low-intent).
2. For each channel ``c``, exposure ``X_c`` is sampled from a Bernoulli
   whose probability is ``logistic(alpha_c + beta_c * intent)``.
   ``beta_c`` is large and positive for retargeting and brand search (they
   target high-intent users), small for display and social, zero-ish for
   email and direct.
3. Conversion ``Y`` is sampled from a Bernoulli whose probability is
   ``logistic(delta_0 + delta_u * intent + sum_c tau_c * X_c)``.
   ``tau_c`` is the **true causal effect** of channel ``c`` on the
   log-odds of conversion; we expose it as ``df.attrs["true_effects"]``.
4. For converting users we synthesize a plausible timestamp sequence. The
   ordering encodes funnel position: upper-funnel channels (display,
   social, video) land earlier; lower-funnel channels (retargeting, brand
   search, email) land later.

The resulting DataFrame is touchpoint-level. Each converting user has one
row per exposed channel (so the heuristic attributors can distribute
credit), plus one row per channel for non-converting users to keep the
propensity models honest.

The returned DataFrame carries a ``df.attrs`` dictionary containing:

- ``true_effects``: per-channel causal effect on the probability of
  conversion, expressed as the approximate lift in P(Y=1) from setting
  ``X_c`` from 0 to 1 at the population mean of intent.
- ``true_effects_logodds``: raw ``tau_c`` coefficients on the log-odds
  scale.
- ``channels``: the channel universe.
- ``confounders``: names of the confounders (just ``["intent"]``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CHANNELS: tuple[str, ...] = (
    "display",
    "social",
    "video",
    "non_brand_search",
    "brand_search",
    "retargeting",
    "email",
    "direct",
)

# Exposure sensitivity to intent. Larger values mean the channel is MORE
# selected into by high-intent users (i.e. the channel is intent-capturing).
EXPOSURE_BETA: dict[str, float] = {
    "display": 0.2,
    "social": 0.3,
    "video": 0.1,
    "non_brand_search": 1.0,
    "brand_search": 2.2,
    "retargeting": 3.0,
    "email": 1.1,
    "direct": 0.4,
}

# Baseline exposure log-odds (alpha_c) so that average exposure is realistic.
EXPOSURE_ALPHA: dict[str, float] = {
    "display": -1.6,
    "social": -1.5,
    "video": -2.2,
    "non_brand_search": -1.8,
    "brand_search": -2.2,
    "retargeting": -2.6,
    "email": -1.4,
    "direct": -1.2,
}

# True causal effect on log-odds of conversion (tau_c).
# The story here: demand-generation channels have real but modest effects,
# demand-capture channels have SMALL true effects but their *observed*
# effects look huge because of confounding by intent.
TRUE_EFFECT_LOGODDS: dict[str, float] = {
    "display": 0.55,
    "social": 0.60,
    "video": 0.50,
    "non_brand_search": 0.45,
    "brand_search": 0.20,
    "retargeting": 0.15,
    "email": 0.30,
    "direct": 0.10,
}

# Funnel position: lower values place the channel earlier in the journey.
FUNNEL_POSITION: dict[str, float] = {
    "video": 0.1,
    "display": 0.2,
    "social": 0.3,
    "non_brand_search": 0.5,
    "email": 0.6,
    "direct": 0.7,
    "brand_search": 0.8,
    "retargeting": 0.9,
}

INTENT_BETA_A = 1.5
INTENT_BETA_B = 4.5  # right-skewed
CONV_INTERCEPT = -2.3
CONV_INTENT_COEF = 2.6


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _true_effects_probability_scale(
    effects_logodds: dict[str, float],
    intent_mean: float,
) -> dict[str, float]:
    """Approximate lift in P(Y=1) from flipping X_c at the population mean of intent.

    Uses the formula Delta_p = sigmoid(eta + tau) - sigmoid(eta) evaluated
    at a baseline log-odds corresponding to an average user with no other
    channel exposure. This is a reasonable proxy for 'incremental
    conversion rate' for small baseline rates.
    """

    baseline = CONV_INTERCEPT + CONV_INTENT_COEF * intent_mean
    out = {}
    for ch, tau in effects_logodds.items():
        out[ch] = float(_sigmoid(baseline + tau) - _sigmoid(baseline))
    return out


def generate_synthetic_journeys(
    n_users: int = 10_000,
    *,
    seed: int | None = 42,
    channels: tuple[str, ...] = CHANNELS,
) -> pd.DataFrame:
    """Generate a synthetic journey dataset with known ground-truth effects.

    Parameters
    ----------
    n_users:
        Number of users to simulate.
    seed:
        Seed for the numpy random generator.
    channels:
        Subset of the default channel universe to simulate. Unknown
        channels raise ``KeyError``.

    Returns
    -------
    pandas.DataFrame
        Touchpoint-level journeys with columns
        ``user_id, channel, timestamp, converted, intent, n_touches``.
        Ground-truth effects are attached under ``df.attrs``.
    """

    for ch in channels:
        if ch not in TRUE_EFFECT_LOGODDS:
            raise KeyError(f"Unknown channel '{ch}'")

    rng = np.random.default_rng(seed)
    n = int(n_users)

    intent = rng.beta(INTENT_BETA_A, INTENT_BETA_B, size=n)

    # Exposure matrix
    alpha = np.array([EXPOSURE_ALPHA[c] for c in channels])
    beta = np.array([EXPOSURE_BETA[c] for c in channels])
    logits = alpha[None, :] + beta[None, :] * intent[:, None]
    probs = _sigmoid(logits)
    exposure = (rng.uniform(size=probs.shape) < probs).astype(int)

    # Conversion outcome
    tau = np.array([TRUE_EFFECT_LOGODDS[c] for c in channels])
    conv_logit = (
        CONV_INTERCEPT
        + CONV_INTENT_COEF * intent
        + exposure @ tau
    )
    conv_probs = _sigmoid(conv_logit)
    converted = (rng.uniform(size=n) < conv_probs).astype(int)

    # Build touchpoint-level frame
    user_ids = np.arange(n)
    rows = []
    base_time = pd.Timestamp("2026-01-01")
    for i in range(n):
        exposed = [c for c, flag in zip(channels, exposure[i]) if flag]
        if not exposed:
            # Keep the user in the frame with a "none" placeholder for
            # downstream propensity tables; we skip them for heuristic
            # attribution by excluding them here. However we still want them
            # in the causal model, so we emit a single row with channel=None.
            continue
        # Sort exposed by funnel position with a small random jitter.
        order = sorted(
            exposed,
            key=lambda c: FUNNEL_POSITION.get(c, 0.5) + rng.normal(0, 0.05),
        )
        for j, ch in enumerate(order):
            ts = base_time + pd.Timedelta(hours=j * 24 + rng.integers(0, 23))
            rows.append(
                {
                    "user_id": int(user_ids[i]),
                    "channel": ch,
                    "timestamp": ts,
                    "converted": int(converted[i]),
                    "intent": float(intent[i]),
                    "n_touches": len(exposed),
                }
            )

    df = pd.DataFrame(rows)
    # Include a per-user summary via attrs for convenience.
    df.attrs["true_effects"] = _true_effects_probability_scale(
        {c: TRUE_EFFECT_LOGODDS[c] for c in channels},
        intent_mean=float(intent.mean()),
    )
    df.attrs["true_effects_logodds"] = {c: float(TRUE_EFFECT_LOGODDS[c]) for c in channels}
    df.attrs["channels"] = list(channels)
    df.attrs["confounders"] = ["intent"]

    # Stash per-user snapshot as a separate attribute so causal estimators
    # don't need to reconstruct it.
    per_user = pd.DataFrame(
        {
            "user_id": user_ids,
            "intent": intent,
            "converted": converted,
        }
    )
    for idx, ch in enumerate(channels):
        per_user[f"x_{ch}"] = exposure[:, idx]
    df.attrs["per_user"] = per_user

    return df
