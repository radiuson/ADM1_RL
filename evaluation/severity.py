"""Severity of soft-limit excursions, per scenario and pooled.

A violation rate counts the steps above the limit and says nothing about how
far above, or for how long.  The same 5 % rate can be a scattering of one-day
excursions a few mg/L over the limit, or a single week-long run approaching
the inhibition threshold, and those are not the same risk.  These are the
quantities that separate the two.

Two scenarios in the suite cannot reach the soft limit even at maximum feed,
so they contribute zero violations by construction and dilute any pooled rate.
Pooled and per-scenario figures are therefore both returned, and
``reachable_rate`` restricts the pooled rate to the scenarios where the limit
is attainable.
"""
from __future__ import annotations

import numpy as np

VFA_SOFT = 0.320
VFA_HARD = 1.600
MG = 1000 / 1.0667


def _runs_above(mask):
    """Lengths of the maximal consecutive True stretches in ``mask``."""
    out, n = [], 0
    for m in mask:
        if m:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def scenario_metrics(vfa):
    """Severity of one scenario's VFA trajectory (kg COD/m3 per control step)."""
    v = np.asarray(vfa, dtype=float)
    over = v > VFA_SOFT
    excess = np.clip(v - VFA_SOFT, 0.0, None)
    spans = _runs_above(over)
    return {
        'steps': int(v.size),
        'viol': float(100 * over.mean()) if v.size else 0.0,
        'viol_hard': float(100 * (v > VFA_HARD).mean()) if v.size else 0.0,
        # Mean excess over the violating steps only: how far past the limit an
        # excursion goes, rather than an average diluted by compliant steps.
        'excess_mean': float(excess[over].mean() * MG) if over.any() else 0.0,
        # Integrated exceedance over the whole trajectory, in mg/L-steps: the
        # area above the limit, which combines depth and duration.
        'excess_area': float(excess.sum() * MG),
        'excess_max': float(excess.max() * MG) if v.size else 0.0,
        'vfa_max': float(v.max() * MG) if v.size else 0.0,
        'longest_run': int(max(spans)) if spans else 0,
        'n_excursions': len(spans),
    }


def summarise(per_scenario, reachable=None):
    """Pool per-scenario metrics and add the worst-case and reachable views.

    ``per_scenario`` maps a scenario name to the dict ``scenario_metrics``
    returns.  ``reachable`` names the scenarios in which the soft limit can be
    attained; when given, ``reachable_*`` is computed over those alone.
    """
    names = list(per_scenario)
    steps = np.array([per_scenario[n]['steps'] for n in names], dtype=float)
    total = steps.sum()

    def weighted(key):
        if not total:
            return 0.0
        vals = np.array([per_scenario[n][key] for n in names], dtype=float)
        return float((vals * steps).sum() / total)

    out = {
        'viol': weighted('viol'),
        'viol_hard': weighted('viol_hard'),
        'excess_area': float(sum(per_scenario[n]['excess_area'] for n in names)),
        'excess_max': float(max(per_scenario[n]['excess_max'] for n in names)),
        'vfa_max': float(max(per_scenario[n]['vfa_max'] for n in names)),
        'longest_run': int(max(per_scenario[n]['longest_run'] for n in names)),
        'n_excursions': int(sum(per_scenario[n]['n_excursions'] for n in names)),
        # The scenario a plant would be judged on: the one that goes worst.
        'worst_viol': float(max(per_scenario[n]['viol'] for n in names)),
        'worst_scenario': max(names, key=lambda n: per_scenario[n]['viol']),
        'per_scenario': per_scenario,
    }
    if reachable:
        keep = [n for n in names if n in reachable]
        if keep:
            rs = np.array([per_scenario[n]['steps'] for n in keep], dtype=float)
            rv = np.array([per_scenario[n]['viol'] for n in keep], dtype=float)
            out['reachable_viol'] = float((rv * rs).sum() / rs.sum())
            out['reachable_scenarios'] = len(keep)
    return out
