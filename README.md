# Causal Multi-Touch Attribution

A Python library that contrasts heuristic multi-touch attribution (MTA) against causal DAG-based attribution on a synthetic marketing journey dataset where the ground truth is known by construction.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![DoWhy](https://img.shields.io/badge/causal--inference-DoWhy%20%7C%20EconML-8A2BE2)

## What this is

`causal-attribution` is a small, self-contained reference implementation of six attribution methods on a synthetic dataset where the data-generating process (DGP) is explicit and the true per-channel causal effects are recoverable:

- Heuristic baselines: first-touch, last-touch, linear, time-decay, U-shaped.
- Markov chain attribution with the Shao & Li (2011) removal-effect estimator.
- Shapley-value attribution (exact for <= 8 channels, Monte-Carlo otherwise).
- Causal DAG attribution via Pearl's do-calculus, using DoWhy's backdoor-identification step and EconML's doubly-robust learner for numerical estimation.

Running all six on the same dataset exposes a pattern that is consistent with the published research: heuristic and Markov-chain methods systematically overcredit channels that are correlated with user intent (retargeting, brand search) and undercredit channels that create new demand (display, social, video). The causal DAG method recovers estimates that are close to the ground-truth effects.

## Why this exists

Production attribution stacks measure *who ads reach*, not *what ads cause*. That is a textbook confounding problem: user intent drives both which ads are served (targeting) and whether users convert (behavior). MTA models that operate on observed touchpoint sequences cannot distinguish the causal effect of an ad from the selection effect of targeting. This is not a minor statistical nuance. It is why [multi-touch attribution is correlation, not causation](https://productphilosophy.com/articles/multi-touch-attribution-causal-inference-dag) - and why every company running standard MTA is misallocating budget.

This repository is the companion code to that article. It is meant to be read, forked, and extended. The synthetic DGP is deliberately simple so you can change the confounding structure, the exposure model, or the outcome model and watch the attributors' errors change. The causal estimator uses the same DoWhy + EconML stack that you would use in production; the DAG is swappable and so are the adjustment variables.

## Install

```bash
git clone https://github.com/benovamurat/causal-attribution.git
cd causal-attribution
pip install -e .
```

Python >= 3.10 is required.

## Quickstart

```python
from causal_attribution import compare_methods, generate_synthetic_journeys

df = generate_synthetic_journeys(n_users=8_000, seed=42)
shares = compare_methods(df)
print(shares)

# Total absolute error vs. ground truth, per method:
print(
    shares.drop(columns=["ground_truth"])
          .sub(shares["ground_truth"], axis=0)
          .abs()
          .sum()
          .sort_values()
)
```

Output (abridged, actual numbers vary with seed):

```
                  first_touch  last_touch  linear  markov  shapley  causal_dag  ground_truth
channel
brand_search           0.045      0.185    0.109   0.122    0.109      0.074         0.061
display                0.236      0.060    0.138   0.119    0.138      0.214         0.200
retargeting            0.025      0.199    0.090   0.111    0.090      0.016         0.041
social                 0.193      0.078    0.147   0.134    0.147      0.216         0.247
video                  0.170      0.028    0.080   0.071    0.080      0.128         0.110
...

Total absolute error vs. ground truth
causal_dag    0.14
first_touch   0.24
linear        0.43
shapley       0.43
markov        0.53
last_touch    0.90
```

Last-touch understates display's share of conversions by roughly 14 percentage points and overstates retargeting's share by roughly 16 points. The causal DAG estimator lands within 1-3 points of ground truth on the most confounded channels.

## Method

### The DAG

The synthetic DGP has one latent confounder - ``intent`` - that causes both channel exposure and conversion. The model built by `CausalAttributor` is:

```
          +-----------------+
          |     intent      |          <- latent propensity, drives targeting and conversion
          +-----------------+
           /  /  /   |   \   \   \
          v  v  v    v    v   v   v
    +---+ +---+ +---+ +---+ +---+ +---+
    | dsp | soc | vid | srch| ret | eml|   <- channel exposures
    +---+ +---+ +---+ +---+ +---+ +---+
         \       |      |      |       /
          \      |      |      |      /
           v     v      v      v     v
             +---------------------+
             |     converted       |
             +---------------------+
```

Intent has incoming arrows from nothing (it is the root confounder), out-arrows to every channel, and an out-arrow to the outcome. Every channel points at the outcome. The Pearl-style do-calculus says: to estimate the causal effect of a single channel `X_c` on conversion, we need to close every backdoor path from `X_c` to `converted`. Here there is only one: `X_c <- intent -> converted`. Conditioning on intent blocks it.

### Do-calculus, step by step, for one channel

We want `P(converted | do(X_retargeting = 1)) - P(converted | do(X_retargeting = 0))` - the Average Treatment Effect (ATE).

1. In the DAG, remove all arrows *into* `X_retargeting`. This is the graph surgery that defines `do(X)`.
2. Apply the backdoor adjustment formula:

   ```
   P(Y | do(X = x)) = sum_z  P(Y | X = x, Z = z) * P(Z = z)
   ```

   where `Z = {intent}`. Note `P(Z = z)` is the marginal distribution of `Z`, not the conditional given `X`.
3. The ATE is then the difference between the do-set probabilities at `X = 1` and `X = 0`.

In code, `CausalAttributor.estimate_ate("retargeting")` does this automatically: it constructs the DAG, identifies the adjustment set via DoWhy, and estimates the effect via EconML's `LinearDRLearner` (doubly robust). It also returns the *naive* difference (the MTA-style observational estimate) for contrast.

### Markov chain attribution

Let `P` be the transition matrix over states `{start, ch_1, ..., ch_k, converted, null}`. Fit `P` from observed paths (append `converted` or `null` as the absorbing state). The baseline conversion probability is

```
pi_conv = P(absorb at converted | start from start)  = (I - Q)^(-1) R, evaluated at the start row.
```

For each channel `c`, build `P^{-c}` by redirecting every transition into a state containing `c` to `null`. The **removal effect** is `pi_conv - pi_conv^{-c}`. Normalizing removal effects across channels gives the attribution share. This is the Shao & Li (2011) algorithm.

### Shapley-value attribution

Given a coalition value function `v(S) = expected conversions of users whose exposure set is a subset of S`, channel `c`'s Shapley value is

```
phi_c = sum_{S subseteq N \ {c}}  |S|! * (n - |S| - 1)! / n!  *  [v(S ∪ {c}) - v(S)]
```

The package computes this exactly for up to 8 channels and switches to a permutation Monte-Carlo approximation above that.

## Why heuristics fail

The intent confounder is the entire story. Last-touch says "the channel that closed the deal caused the sale." But high-intent users were going to convert anyway, and they are disproportionately exposed to retargeting and brand search because targeting algorithms by design prefer them. The heuristic sees exposure immediately before conversion and credits the channel. The causal question is: would the conversion still have happened without the exposure? Heuristics cannot answer that question because they do not condition on the confounder. The DAG-based estimator answers it by construction.

The same pattern explains why demand-generation channels look weak in MTA dashboards. Display, social, and video touch users early in the funnel, before intent signals exist. Those users are then cookied, retargeted, and retouched by lower-funnel channels. Last-touch gives all the credit to the last channel. Linear and time-decay distribute credit but still undercount early touches in a multi-channel journey. Markov chain attribution does somewhat better but inherits the same observational bias - its transition matrix is estimated from data that is confounded by intent.

## API reference

| Import | Purpose |
| --- | --- |
| `generate_synthetic_journeys(n_users, seed)` | Produce a journeys DataFrame with known ground-truth effects in `df.attrs`. |
| `first_touch(journeys)` | All credit to the first touchpoint in each converting journey. |
| `last_touch(journeys)` | All credit to the last touchpoint. |
| `linear(journeys)` | Even split across touchpoints. |
| `time_decay(journeys, halflife_days)` | Exponential decay toward recent touchpoints. |
| `u_shaped(journeys, first_weight, last_weight)` | Position-based (bathtub) attribution. |
| `MarkovAttribution(order=1).fit(journeys).attribute(journeys)` | Removal-effect Markov-chain attribution. |
| `shapley_attribution(journeys)` | Exact / Monte-Carlo Shapley value over channel coalitions. |
| `CausalAttributor(channels, confounders).estimate_ate(journeys, channel)` | Per-channel ATE via DoWhy + EconML (doubly robust). |
| `CausalAttributor(...).attribute(journeys)` | Per-channel credit proportional to estimated ATEs. |
| `compare_methods(journeys)` | All attributors side-by-side as conversion shares. |

## Results

Running `python examples/quickstart.py` on 8,000 synthetic users with the default seed produces a table like this:

```
Share of conversions credited to each channel, by method:
                  first_touch  last_touch  linear  markov  shapley  causal_dag  ground_truth
channel
brand_search           0.045      0.185    0.109   0.122    0.109      0.074         0.061
direct                 0.068      0.192    0.138   0.155    0.138      0.042         0.044
display                0.236      0.060    0.138   0.119    0.138      0.214         0.200
email                  0.127      0.158    0.165   0.166    0.165      0.127         0.140
non_brand_search       0.137      0.101    0.134   0.124    0.134      0.183         0.158
retargeting            0.025      0.199    0.090   0.111    0.090      0.016         0.041
social                 0.193      0.078    0.147   0.134    0.147      0.216         0.247
video                  0.170      0.028    0.080   0.071    0.080      0.128         0.110

Error vs. ground truth (sum of absolute share deviations):
causal_dag    0.14
first_touch   0.24
u_shaped      0.43
linear        0.43
shapley       0.43
time_decay    0.46
markov        0.53
last_touch    0.90
```

Read the last-touch row for retargeting: 19.9% of conversions credited versus a ground truth of 4.1%. That is a ~5x overstatement. Read the last-touch row for display: 6.0% versus ground truth 20.0% - a 70% understatement. The causal DAG method gets both channels within a few points of truth.

Run `python examples/dag_walkthrough.py` for a guided tour of the DAG and the do-calculus for a single channel, with `LinearDRLearner` confidence intervals.

## Limitations

The causal DAG method is not magic. It requires:

- **A correctly specified DAG.** If you misdraw the causal structure - for instance by omitting a confounder or by treating a mediator as a confounder - the estimate is biased. This package exposes the DAG as a `networkx.DiGraph` you can edit.
- **Valid instruments when the backdoor criterion fails.** If there are unmeasured confounders, conditioning on the measured ones is not enough. You need an instrumental variable, a natural experiment, or a randomized holdout. The package does not build instruments for you.
- **Positivity.** Every user must have a nonzero probability of exposure for every channel. A retargeting campaign that targets 100% of past visitors violates positivity and makes IPW weights explode. In production, implement random holdouts (2-5%) so every campaign has natural variation.
- **Stable unit treatment value.** One user's exposure shouldn't affect another user's outcome. In most digital settings this is approximately true; in social/viral settings it is not.
- **The synthetic DGP is a toy.** The ground-truth effects are small and the confounding structure is one-dimensional. Real marketing systems have many simultaneous confounders, selection processes, and feedback loops. The synthetic dataset exists to make the comparison *interpretable*, not to quantify the exact size of real-world MTA error.

## References

- Pearl, J. (2000). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Pearl, J. (2018). *The Book of Why: The New Science of Cause and Effect*. Basic Books.
- Shao, X., & Li, L. (2011). Data-driven multi-touch attribution models. *Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 258-264.
- Berman, R. (2018). Beyond the Last Touch: Attribution in Online Advertising. *Marketing Science*, 37(5), 771-792.
- Sharma, A., & Kiciman, E. (2020). DoWhy: An End-to-End Library for Causal Inference. arXiv:2011.04216.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
- Blake, T., Nosko, C., & Tadelis, S. (2015). Consumer heterogeneity and paid search effectiveness. *Econometrica*, 83(1), 155-174.
- Gordon, B. R., Zettelmeyer, F., Bhatt, N., & Larsen, B. (2019). A comparison of approaches to advertising measurement. *Marketing Science*, 38(2), 193-225.
- Rao, J. M., & Simonov, A. (2023). Correcting for selection bias in advertising measurement. *Marketing Science*, 42(3), 412-431.
- Bang, H., & Robins, J. M. (2005). Doubly robust estimation in missing data and causal inference models. *Biometrics*, 61(4), 962-973.

## Citation

If you use this code in research or writing, please cite the companion article:

```bibtex
@misc{ova2023causalattribution,
  author       = {Ova, Murat},
  title        = {Multi-Touch Attribution Is Broken -- A Causal Inference Approach Using Directed Acyclic Graphs},
  year         = {2023},
  howpublished = {\url{https://productphilosophy.com/articles/multi-touch-attribution-causal-inference-dag}},
  note         = {Companion repository: \url{https://github.com/benovamurat/causal-attribution}}
}
```

## License

MIT, copyright 2026 Murat Ova. See [LICENSE](LICENSE).
