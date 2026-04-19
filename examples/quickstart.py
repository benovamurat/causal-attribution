"""End-to-end comparison of heuristic, algorithmic, and causal attributors.

Run this script from the repository root::

    python examples/quickstart.py

It generates a synthetic journey dataset with a known ground-truth effect
per channel, then runs every attributor in the package side-by-side. The
resulting table makes the article's claim concrete: last-touch and
Markov-chain attribution systematically overcredit intent-capturing
channels (retargeting, brand search) and undercredit demand-generation
channels (display, social, video).
"""

from __future__ import annotations

import pandas as pd

from causal_attribution import compare_methods, generate_synthetic_journeys


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")

    print("Generating synthetic journey dataset...")
    df = generate_synthetic_journeys(n_users=8000, seed=42)

    n_users = df["user_id"].nunique()
    n_conv = int((df.groupby("user_id")["converted"].max() == 1).sum())
    n_touch = len(df)
    conv_rate = n_conv / n_users if n_users else 0
    print(
        f"Users:         {n_users:>6d}\n"
        f"Touchpoints:   {n_touch:>6d}\n"
        f"Conversions:   {n_conv:>6d}\n"
        f"Conv. rate:    {conv_rate:>6.2%}"
    )
    print()

    print("Running attributors (this calls DoWhy + EconML once per channel)...")
    shares = compare_methods(df)

    print()
    print("Share of conversions credited to each channel, by method:")
    print(shares.to_string())

    print()
    print("Error vs. ground truth (sum of absolute share deviations):")
    gt = shares["ground_truth"]
    errors = (shares.drop(columns=["ground_truth"]).sub(gt, axis=0)).abs().sum().sort_values()
    print(errors.to_string())

    print()
    print("Ranked finding: the causal DAG method should have the lowest total error.")
    print(
        "Last-touch is expected to overstate retargeting and brand_search, and "
        "understate display/social/video - mirroring the published research."
    )


if __name__ == "__main__":
    main()
