# ADM1_RL — a constrained control benchmark for anaerobic digestion

Code and per-run results for

> *Algorithm Choice Matters More Than Penalty-Weight Tuning in Constrained
> Anaerobic Digestion Control: A 24-Family Benchmark*

Release [`paper-benchmark-v1`](https://github.com/radiuson/ADM1_RL/releases/tag/paper-benchmark-v1)
is the version the paper's tables and figures are computed from.

An earlier study of this reactor under thermal stress — a 40-state
temperature-extended solver with a heating action and its own scenario set — is
preserved at the tag
[`thermal-stress-study`](https://github.com/radiuson/ADM1_RL/releases/tag/thermal-stress-study).
It is a different environment and its numbers do not correspond to this paper.

---

## What this benchmark does

It compares reinforcement learning against conventional control on a
plant-anchored ADM1 digester, on the axis operators actually trade along:
methane production against the share of time volatile fatty acids (VFA) sit
above a soft limit.

- **346 conventional configurations** (PI, constant feed, rule-based) are swept
  and reduced to an **82-point Pareto upper envelope** that reaches zero
  violations. Every RL run is scored against this envelope interpolated at the
  run's own violation rate, so nothing is compared against a single arbitrary
  baseline tuning.
- **24 RL configurations** are evaluated: ten under a reward-penalty
  formulation (four penalty weights × ten seeds each) and fourteen under a
  constrained (CMDP) formulation (four cost limits × ten seeds each), where the
  permitted excursion rate is a specification rather than a tuning outcome.

## Reproducing the paper's tables and figures

The per-run evaluation records ship with the repository, so the tables and
figures rebuild from a clean clone with no training and no GPU:

```bash
git clone https://github.com/radiuson/ADM1_RL.git
cd ADM1_RL
pip install -e .

python3 scripts/build_tables.py <outdir>   # table_results.tex, table_cmdp.tex, numbers.tex
python3 scripts/plot_figures.py <outdir>   # fig_frontier.pdf, fig_tracking.pdf
```

Rebuilding the tables and figures needs only numpy and matplotlib; the rest of
the dependency set is for training. Run the scripts from the clone — they are
not installed onto the path.

`build_tables.py` prints the envelope summary it derives, which is the quickest
check that a clone is intact:

```
envelope 346 configurations -> 82 points, 0.00-46.3%, best at zero violation 1551
8/10 on-policy constrained families above the envelope with intervals excluding zero
```

Both scripts read `paper_data/` and nothing else. Set `ADM1_PAPER_DATA` to point
them at a freshly evaluated set instead.

### `paper_data/`

| Directory | Records | Contents |
|---|---|---|
| `evbase/` | 346 | conventional configurations: the swept grid behind the envelope |
| `evres/` | 605 | reward-penalty runs (the paper uses 400 after the filter below) |
| `evcmdp/` | 563 | constrained runs, 14 families × 4 cost limits × 10 seeds |

Each record carries the run's mean methane flow, its pooled and per-scenario
violation rates, and the excursion-severity fields (`longest_run`,
`excess_area`, `excess_max`, `vfa_max`, `viol_hard`) computed by
`evaluation/severity.py`.

`evres/` holds more runs than the paper reports. Two filters in
`scripts/build_tables.py` select the balanced design, both stated in the
manuscript: only the four penalty weights swept for every family
(`lw0p5, lw1, lw2, lw5`), and, for A2C, DDPG, TD3 and CrossQ, only the runs
under corrected settings — earlier runs of those four used action-noise,
`n_steps` and warm-up values not on the same scale as the other families. Where
a (family, weight, seed) cell was trained more than once, one run per cell is
kept, chosen by model path so the choice does not depend on the result.

## Training

```bash
python3 training/train_sac_std_cur.py \
    --algo ppo --reward-config lw1 --seed 42 \
    --normalize --step-size 1.0 --timesteps 150000 \
    --output-dir models_out
```

`--algo` accepts `sac, tqc, ppo, recurrentppo, trpo, ddpg, td3, a2c, crossq,
ars`. The constrained families are trained through `training/omnisafe_env.py`.
Effective hyperparameters for every family are in the manuscript's appendix and
are exported by `scripts/export_hyperparams.py`.

## Layout

```
env/
  adm1_gym_env_std.py       the benchmark environment (38-state, 2-D action)
  adm1_solver_std.py        ADM1 solver, 15-min operator splitting, LSODA
  scenario_manager.py       scenario definitions and disturbance injection
  scenarios.yaml            the eight scenarios
  data/                     influent series and the scenario initial states
training/
  train_sac_std_cur.py      training entry point for the ten SB3 families
  train_omnisafe.py         training entry point for the constrained families
  omnisafe_env.py           the environment as omnisafe expects it
  baselines.py              PI, constant-feed and rule-based control laws
  reward_configs.py         the penalty weights swept in the paper
evaluation/
  eval_sb3.py               evaluation for reward-penalty runs
  eval_cmdp.py              evaluation for constrained runs
  eval_baseline_ext.py      the conventional sweep behind the 346 configurations
  severity.py               excursion severity: duration, depth, per scenario
scripts/
  build_tables.py           every number the paper quotes
  plot_figures.py           both figures
  stratified_stats.py       class comparison, stratified families/weights/seeds
  plant_block_bootstrap.py  moving-block bootstrap on the plant record
  export_hyperparams.py     the appendix's hyperparameter table
paper_data/                 per-run records behind every table and figure
```

Trained weights are not released; `models_*/` is ignored. The scripts in
`scripts/` read `paper_data/` and are the path from those records to the
paper's numbers.

## Numerical setup

The acid–base equilibrium is solved algebraically outside the ODE, so
integration is interrupted at a 15-minute operator-splitting interval: each
one-day control step is 96 sub-steps. The stiff integrator is LSODA at
`rtol=1e-5`, `atol=1e-7`. Episodes are 60 days at a one-day control interval.

The plant record sets reactor geometry, feed range and the observation set; the
ADM1 parameters are literature values and were not fitted to that plant. The
manuscript states what this does and does not license, and absolute violation
rates are not claimed to transfer to a physical digester.

## Requirements

Python ≥ 3.9 and PyTorch ≥ 2.0 throughout. The two formulations need **two
separate environments**, because omnisafe pins a gymnasium older than the one
stable-baselines3 runs under and the two cannot be installed together:

| | Reward-penalty (ten families) | Constrained (fourteen families) |
|---|---|---|
| Install | `pip install -e .` | `pip install -e ".[cmdp]"` |
| gymnasium | 1.2.2 | 0.28.1 |
| stable-baselines3 | 2.9.0 | — |
| sb3-contrib | 2.9.0 | — |
| omnisafe | — | 0.5.0 |
| numpy | 2.2.6 | 1.26.4 |

The versions above are the ones the reported runs were produced under.
Rebuilding the tables and figures from `paper_data/` works in either
environment.

## Citation

If you use this benchmark, please cite the paper above and the release tag the
results were produced under.

## License

MIT; see `LICENSE`.
