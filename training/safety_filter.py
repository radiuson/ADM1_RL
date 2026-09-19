"""Model-based lookahead filter on the feed rate.

Before an action is applied, the same ADM1 model is rolled forward over a short
horizon with that action held constant.  If the predicted VFA crosses the action
level, the feed rate is reduced until the prediction clears it, or to the lowest
admissible rate if no candidate does.

This is a deliberate upper bound rather than a deployable controller: the filter
simulates the very model it is controlling, so it sees the future exactly.  A
plant would face model mismatch, and the gap this filter closes is therefore the
most that lookahead could contribute, not what it would contribute in practice.
"""
import copy

import numpy as np


class LookaheadFilter:
    def __init__(self, env_factory, limit_kgcod_m3, horizon_days=3,
                 candidates=(1.0, 0.8, 0.6, 0.45, 0.3)):
        """
        env_factory  callable returning a fresh env of the same scenario,
                     used as the shadow model for the rollouts
        candidates   multipliers on the proposed feed rate, tried in order
        """
        self.make = env_factory
        self.limit = float(limit_kgcod_m3)
        self.H = int(horizon_days)
        self.cands = tuple(candidates)
        self._shadow = None
        self.n_calls = 0
        self.n_clipped = 0

    def _predict_peak(self, base_env, q, mult):
        """Highest VFA the model reaches over the horizon if (q, mult) is held."""
        if self._shadow is None:
            self._shadow = self.make()
            self._shadow.reset(seed=0)
        s = self._shadow
        s.solver.set_state(copy.deepcopy(base_env.current_state))
        s.current_step = base_env.current_step
        s.current_time_days = base_env.current_time_days
        a = np.array([q, mult], dtype=np.float32)
        peak = 0.0
        for _ in range(self.H):
            s.step(a)
            peak = max(peak, s._calculate_total_vfa(s.current_state))
        return peak

    def __call__(self, env, q, mult):
        self.n_calls += 1
        for k, f in enumerate(self.cands):
            q_try = float(np.clip(q * f, env.Q_AD_MIN_M3D, env.Q_AD_MAX_M3D))
            if self._predict_peak(env, q_try, mult) <= self.limit:
                if k:
                    self.n_clipped += 1
                return q_try, mult
        self.n_clipped += 1
        return float(env.Q_AD_MIN_M3D), mult


class TrendFilter:
    """Lookahead from the measured VFA trajectory alone.

    The recent slope of the measured VFA is extrapolated over the same horizon
    the model-based filter uses, and the feed is cut back when the projection
    crosses the action level.  Nothing beyond the daily VFA record is needed -
    no calibrated model, no state estimate - so this is the variant a plant
    could run as it stands.

    Cutting back is immediate but recovery is rate limited, because a filter
    that restores full feed as soon as the slope turns negative drives the
    reactor between its feed limits: the VFA falls, the projection clears, the
    feed returns, and the cycle repeats.  Rate limiting the recovery is also
    what an operator does after a digester sours - feed is restored gradually,
    not at once.
    """

    def __init__(self, limit_kgcod_m3, horizon_days=3, window=3,
                 candidates=(1.0, 0.8, 0.6, 0.45, 0.3), damping=0.6,
                 recovery_per_day=0.15):
        self.limit = float(limit_kgcod_m3)
        self.H = int(horizon_days)
        self.window = int(window)
        self.cands = tuple(candidates)
        self.damping = float(damping)
        self.recovery = float(recovery_per_day)
        self.hist = []
        self.q_allowed = None
        self.n_calls = 0
        self.n_clipped = 0

    def reset(self):
        self.hist = []
        self.q_allowed = None

    def __call__(self, env, q, mult):
        self.n_calls += 1
        if self.q_allowed is None:
            self.q_allowed = float(env.Q_AD_MAX_M3D)
        v = env._calculate_total_vfa(env.current_state)
        self.hist.append(v)
        if len(self.hist) > self.window:
            self.hist.pop(0)

        slope = ((self.hist[-1] - self.hist[0]) / (len(self.hist) - 1)
                 if len(self.hist) > 1 else 0.0)
        proj = v + self.damping * slope * self.H

        cap = self.q_allowed * (1.0 + self.recovery)          # slow recovery
        if proj > self.limit:
            over = proj / max(self.limit, 1e-9)
            cap = self.q_allowed / max(over, 1.0)             # immediate cut
            self.n_clipped += 1
        self.q_allowed = float(np.clip(cap, env.Q_AD_MIN_M3D, env.Q_AD_MAX_M3D))
        return float(min(q, self.q_allowed)), mult


BATSTONE_PARAMS = dict(k_dis=0.5, k_hyd_ch=10.0, k_hyd_pr=10.0, k_hyd_li=10.0,
                       k_m_su=30.0, K_S_su=0.5, k_m_fa=6.0, K_S_fa=0.4,
                       k_m_c4=20.0, K_S_c4=0.2, k_m_pro=13.0, K_S_pro=0.1,
                       k_m_ac=8.0, K_S_ac=0.15)


def mismatched_env_factory(base_factory, params=BATSTONE_PARAMS):
    """A shadow model calibrated differently from the plant it filters.

    Razaviarani & Buchanan fitted this environment's kinetics to a co-digester
    of matched feed; the ADM1 defaults are a different calibration of the same
    structure, so using them in the filter is the mismatch a plant would face if
    it filtered with a literature model rather than one fitted to itself.
    """
    def make():
        e = base_factory()
        for k, v in params.items():
            if hasattr(e.solver, k):
                setattr(e.solver, k, v)
        return e
    return make
