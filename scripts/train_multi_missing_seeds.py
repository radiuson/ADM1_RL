#!/usr/bin/env python3
"""
Train multi-scenario SAC for seeds 789 and 1234 (full + simple obs).

Launches all 4 runs in parallel using multiprocessing.
Each run takes ~85 min; parallel wall time ~85-100 min.

Usage:
    python scripts/train_multi_missing_seeds.py \
        --results-dir /home/ihpc/code/biogas/ADM1/rl/results_repro
"""

import argparse
import json
import multiprocessing as mp
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


SCENARIOS = [
    'nominal', 'high_load', 'shock_load',
    'low_load', 'temperature_drop', 'cold_winter',
]
SEEDS        = [789, 1234]
OBS_MODES    = ['full', 'simple']
REWARD_KEY   = 'safety_first'
TOTAL_STEPS  = 500_000

HYPERPARAMS = dict(
    learning_rate  = 3e-4,
    buffer_size    = 1_000_000,
    batch_size     = 256,
    tau            = 0.005,
    gamma          = 0.99,
    learning_starts= 10_000,
    train_freq     = 1,
    gradient_steps = 1,
    ent_coef       = 'auto',
    policy_kwargs  = dict(net_arch=[256, 256], use_sde=False),
)


def train_one(seed: int, obs_mode: str, output_dir: pathlib.Path) -> None:
    """Train one (seed, obs_mode) run inside a subprocess."""
    # Re-import inside subprocess
    import sys, pathlib as _pl
    sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

    from training.run_experiment import train_single

    obs_suffix = f'_{obs_mode}' if obs_mode != 'full' else ''
    scenario_key = '_'.join(SCENARIOS)
    run_name = f'sac_{scenario_key}_{REWARD_KEY}_seed{seed}{obs_suffix}'

    run_dir = output_dir / 'training' / run_name
    if (run_dir / 'best_model' / 'best_model.zip').exists():
        print(f'[{run_name}] already exists, skipping', flush=True)
        return

    print(f'[{run_name}] starting training ({TOTAL_STEPS:,} steps) ...', flush=True)
    t0 = time.time()

    try:
        train_single(
            scenario           = SCENARIOS,        # list → multi-scenario
            reward_config_name = REWARD_KEY,
            seed               = seed,
            algo               = 'sac',
            total_timesteps    = TOTAL_STEPS,
            output_dir         = output_dir,
            obs_mode           = obs_mode,
            hyperparams_cfg    = {},               # use defaults (same as existing runs)
            eval_freq          = 10_000,
            n_eval_episodes    = 5,
            verbose            = 0,
        )
        elapsed = time.time() - t0
        print(f'[{run_name}] done in {elapsed/60:.1f} min', flush=True)

        # Save run_meta
        meta = {
            'algo':             'sac',
            'scenario':         SCENARIOS,
            'reward_config':    REWARD_KEY,
            'seed':             seed,
            'obs_mode':         obs_mode,
            'total_timesteps':  TOTAL_STEPS,
            'elapsed_seconds':  round(elapsed, 2),
            'hyperparams':      {k: str(v) for k, v in HYPERPARAMS.items()},
        }
        (run_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2))

    except Exception:
        print(f'[{run_name}] ERROR:', flush=True)
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--results-dir', required=True)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.results_dir).resolve() / 'sac_multi_scenario'
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(seed, obs) for seed in SEEDS for obs in OBS_MODES]
    print(f'=== train_multi_missing_seeds.py ===')
    print(f'  output_dir : {output_dir}')
    print(f'  jobs       : {len(jobs)} runs in parallel')
    for seed, obs in jobs:
        print(f'    seed={seed}  obs={obs}')
    print()

    ctx = mp.get_context('spawn')
    procs = []
    for seed, obs in jobs:
        p = ctx.Process(target=train_one, args=(seed, obs, output_dir), daemon=False)
        p.start()
        procs.append((p, seed, obs))
        time.sleep(2)   # stagger starts slightly

    t_start = time.time()
    while True:
        alive = [(p, s, o) for p, s, o in procs if p.is_alive()]
        done  = len(procs) - len(alive)
        elapsed = time.time() - t_start
        print(f'  [{elapsed/60:.0f} min]  {done}/{len(procs)} done, '
              f'{len(alive)} still running ...', flush=True)
        if not alive:
            break
        time.sleep(300)   # check every 5 min

    for p, seed, obs in procs:
        p.join()

    total = time.time() - t_start
    print(f'\n=== All done in {total/60:.1f} min ===')
    print(f'Next: run eval_missing_seeds_cross.py --seeds 789 1234 on multi dir')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
