"""Walk through the causal DAG attribution for a single channel.

This script prints the DAG used by :class:`CausalAttributor`, explains the
backdoor adjustment set, and estimates the ATE for ``retargeting`` using
two different methods: doubly-robust learning (EconML) and propensity
stratification (DoWhy). It also shows the *naive* conditional difference
for contrast - that is the quantity standard MTA models report.
"""

from __future__ import annotations

import pandas as pd

from causal_attribution import CausalAttributor, generate_synthetic_journeys


STEP_BANNER = "\n" + "=" * 78 + "\n"


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")

    print(STEP_BANNER + "Step 1: Build the DAG" + STEP_BANNER)
    df = generate_synthetic_journeys(n_users=8000, seed=42)
    channels = df.attrs["channels"]
    confounders = df.attrs["confounders"]
    attr = CausalAttributor(channels=channels, confounders=confounders)
    attr.build_graph()
    print(attr.graph_as_gml())

    print(STEP_BANNER + "Step 2: Identify the backdoor adjustment set" + STEP_BANNER)
    print(
        "The only confounder in this synthetic DGP is 'intent' - it causes\n"
        "both channel exposure (by design of targeting) and conversion\n"
        "(because high-intent users convert more often regardless of ads).\n"
        "The backdoor criterion says: condition on intent and we block\n"
        "every non-causal path from channel to conversion."
    )
    print()
    print("Backdoor adjustment set for each channel:")
    for ch in channels:
        print(f"  {ch}: {confounders}")

    print(STEP_BANNER + "Step 3: Estimate ATE - naive vs. causal" + STEP_BANNER)
    rows = []
    for ch in channels:
        result = attr.estimate_ate(df, ch, method="dr")
        truth = df.attrs["true_effects"][ch]
        rows.append(
            {
                "channel": ch,
                "naive_diff": result.naive_diff,
                "causal_ATE": result.ate,
                "ci_low": result.confidence_interval[0]
                if result.confidence_interval
                else float("nan"),
                "ci_high": result.confidence_interval[1]
                if result.confidence_interval
                else float("nan"),
                "ground_truth": truth,
            }
        )
    table = pd.DataFrame(rows).set_index("channel")
    print(table.to_string())

    print()
    print(
        "Read columns left-to-right. 'naive_diff' is what a standard MTA\n"
        "dashboard would approximate if you treated a channel exposure as\n"
        "the treatment and compared means. 'causal_ATE' is the backdoor-\n"
        "adjusted estimate. 'ground_truth' is what the DGP actually does."
    )

    print(STEP_BANNER + "Step 4: Zoom in on retargeting" + STEP_BANNER)
    r = attr.estimate_ate(df, "retargeting", method="dr")
    print(f"Method:               {r.method}")
    print(f"Confounders:          {r.confounders}")
    print(f"Adjustment set:       {r.adjustment_set}")
    print(f"Naive difference:     {r.naive_diff:+.4f}")
    print(f"Causal ATE:           {r.ate:+.4f}")
    if r.confidence_interval:
        print(f"95% CI:               [{r.confidence_interval[0]:+.4f}, "
              f"{r.confidence_interval[1]:+.4f}]")
    print(f"Ground-truth effect:  {df.attrs['true_effects']['retargeting']:+.4f}")

    print()
    print(
        "Interpretation: the naive difference overstates retargeting's effect\n"
        "because high-intent users are both more likely to be retargeted AND\n"
        "more likely to convert regardless of ads. Conditioning on intent\n"
        "closes that backdoor path and recovers a causal estimate that is\n"
        "close to the ground-truth treatment effect."
    )


if __name__ == "__main__":
    main()
