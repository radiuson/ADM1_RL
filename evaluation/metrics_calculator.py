#!/usr/bin/env python3
"""
Metrics Calculator for ADM1 Controller Evaluation
===================================================

Accumulates per-step data during an episode and computes the metrics used in
the paper: production (CH4 yield), safety (violation rate), stability
(CH4 volatility), and economics (net energy).

Usage:
    from evaluation.metrics_calculator import MetricsCalculator

    calc = MetricsCalculator()
    for step in episode:
        calc.add_step(obs, action, reward, info)
    metrics = calc.compute_metrics()

One MetricsCalculator instance should be used per episode.  To start a new
episode either create a fresh instance or call ``reset()``.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MetricsCalculator:
    """
    Per-episode metrics accumulator for ADM1 controller evaluation.

    Call ``add_step()`` after every ``env.step()``, then ``compute_metrics()``
    at the end of the episode.  The returned dict has the following top-level
    keys: ``episode_info``, ``reward``, ``production``, ``safety``,
    ``stability``, ``control``, ``economics``, ``summary``.
    """

    # ── Tracked data ──────────────────────────────────────────────────────────
    rewards:          List[float] = field(default_factory=list)
    ch4_flows:        List[float] = field(default_factory=list)
    ph_values:        List[float] = field(default_factory=list)
    vfa_values:       List[float] = field(default_factory=list)
    nh3_values:       List[float] = field(default_factory=list)
    q_ad_values:      List[float] = field(default_factory=list)
    feed_mult_values: List[float] = field(default_factory=list)

    # ── Violation tracking: (step_index, exceedance_magnitude) ───────────────
    ph_violations:  List[Tuple[int, float]] = field(default_factory=list)
    vfa_violations: List[Tuple[int, float]] = field(default_factory=list)
    nh3_violations: List[Tuple[int, float]] = field(default_factory=list)

    # ── Episode state ─────────────────────────────────────────────────────────
    step_count:        int           = 0
    terminated_early:  bool          = False
    termination_step:  Optional[int] = None

    # ── Physical constants ────────────────────────────────────────────────────
    # 15-minute control interval converted to days
    step_size_days: float = 15.0 / 1440.0
    # Reactor liquid volume (m³) — BSM2 reference digester
    V_liq: float = 3400.0
    # Nominal feed flow for pump-energy normalisation (m³/d)
    _nominal_flow: float = 178.0

    # ── Public interface ──────────────────────────────────────────────────────

    def add_step(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        info: Dict[str, Any],
    ) -> None:
        """
        Record data from one environment step.

        Args:
            observation: Current observation vector (not used; info is
                         authoritative for state variables).
            action:      3-dim action array [q_ad, feed_mult, Q_HEX].
            reward:      Scalar reward received.
            info:        Info dict returned by ``ADM1Env_v2.step()``.
                         Required keys: q_ch4, pH, total_vfa, q_ad,
                         feed_multiplier.  Optional: S_nh3.
        """
        self.step_count += 1

        self.rewards.append(reward)
        self.ch4_flows.append(info['q_ch4'])
        self.ph_values.append(info['pH'])
        self.vfa_values.append(info['total_vfa'])
        self.nh3_values.append(info.get('S_nh3', 0.0))
        self.q_ad_values.append(info['q_ad'])
        self.feed_mult_values.append(info['feed_multiplier'])

        # Safety thresholds match the paper definition
        # pH safe range: [6.8, 7.8]
        if info['pH'] < 6.8:
            self.ph_violations.append((self.step_count, 6.8 - info['pH']))
        elif info['pH'] > 7.8:
            self.ph_violations.append((self.step_count, info['pH'] - 7.8))

        # VFA safe limit: 0.2 kg COD/m³
        if info['total_vfa'] > 0.2:
            self.vfa_violations.append((self.step_count, info['total_vfa'] - 0.2))

        # NH3 safe limit: 0.002 kmol/m³
        if info.get('S_nh3', 0.0) > 0.002:
            self.nh3_violations.append((self.step_count, info['S_nh3'] - 0.002))

    def set_terminated(self, step: int) -> None:
        """Mark episode as terminated early (safety limit breached)."""
        self.terminated_early = True
        self.termination_step = step

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute all metrics from accumulated data.

        Returns:
            Nested dict with keys:
                episode_info, reward, production, safety,
                stability, control, economics, summary.

        Raises:
            ValueError: If called before any steps have been recorded.
        """
        if self.step_count == 0:
            raise ValueError("No data recorded — call add_step() before compute_metrics().")

        metrics = {
            'episode_info': {
                'steps':            self.step_count,
                'duration_days':    self.step_count * self.step_size_days,
                'terminated_early': self.terminated_early,
                'termination_step': self.termination_step,
            },
            'reward':     self._compute_reward_metrics(),
            'production': self._compute_production_metrics(),
            'safety':     self._compute_safety_metrics(),
            'stability':  self._compute_stability_metrics(),
            'control':    self._compute_control_metrics(),
            'economics':  self._compute_economic_metrics(),
        }
        metrics['summary'] = self._compute_summary_score(metrics)
        return metrics

    def reset(self) -> None:
        """
        Reset all accumulators for reuse across episodes.

        Equivalent to creating a new ``MetricsCalculator()`` instance.
        Prefer creating a fresh instance when clarity matters.
        """
        self.rewards.clear()
        self.ch4_flows.clear()
        self.ph_values.clear()
        self.vfa_values.clear()
        self.nh3_values.clear()
        self.q_ad_values.clear()
        self.feed_mult_values.clear()
        self.ph_violations.clear()
        self.vfa_violations.clear()
        self.nh3_violations.clear()
        self.step_count       = 0
        self.terminated_early = False
        self.termination_step = None

    def print_summary(self, metrics: Dict) -> None:
        """Print a human-readable summary of computed metrics."""
        sep = "=" * 65
        print(sep)
        print("  Metrics Summary")
        print(sep)

        ei = metrics['episode_info']
        print(f"\n  Episode")
        print(f"    Steps:      {ei['steps']}")
        print(f"    Duration:   {ei['duration_days']:.2f} days")
        print(f"    Terminated: {ei['terminated_early']}")

        rw = metrics['reward']
        print(f"\n  Reward")
        print(f"    Total: {rw['total']:.2f}")
        print(f"    Mean:  {rw['mean']:.4f} +/- {rw['std']:.4f}")

        pr = metrics['production']
        print(f"\n  Production")
        print(f"    Total CH4:              {pr['total_ch4_m3']:.1f} m3")
        print(f"    Avg flow:               {pr['avg_ch4_flow']:.1f} m3/d")
        print(f"    Volumetric productivity:{pr['volumetric_productivity']:.4f} m3/m3/d")

        sf = metrics['safety']
        print(f"\n  Safety")
        print(f"    Violation rate: {sf['violation_rate']*100:.1f}%  "
              f"({sf['steps_with_violation']}/{ei['steps']} steps)")
        print(f"    pH violations:  {sf['ph_violation_count']}")
        print(f"    VFA violations: {sf['vfa_violation_count']}  "
              f"(max excess: {sf['vfa_max_magnitude']:.4f})")
        print(f"    NH3 violations: {sf['nh3_violation_count']}")

        st = metrics['stability']
        print(f"\n  Stability")
        print(f"    CH4 volatility: {st['ch4_volatility']:.2f} m3/d")
        print(f"    pH volatility:  {st['ph_volatility']:.4f}")
        print(f"    CH4 CV:         {st['ch4_coefficient_of_variation']:.4f}")

        ec = metrics['economics']
        print(f"\n  Economics")
        print(f"    CH4 energy:   {ec['ch4_energy_kwh']:.1f} kWh")
        print(f"    Pump energy:  {ec['pump_energy_kwh']:.1f} kWh")
        print(f"    Net energy:   {ec['net_energy_kwh']:.1f} kWh")
        print(f"    Efficiency:   {ec['energy_efficiency_ratio']:.1f}x")

        sm = metrics['summary']
        print(f"\n  Summary score (internal ranking only)")
        print(f"    Production: {sm['production_score']:.4f}")
        print(f"    Safety:     {sm['safety_score']:.4f}")
        print(f"    Stability:  {sm['stability_score']:.4f}")
        print(f"    Overall:    {sm['overall_score']:.4f}")

        print(sep)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_reward_metrics(self) -> Dict[str, float]:
        r = np.array(self.rewards)
        return {
            'total':  float(np.sum(r)),
            'mean':   float(np.mean(r)),
            'std':    float(np.std(r)),
            'min':    float(np.min(r)),
            'max':    float(np.max(r)),
            'median': float(np.median(r)),
        }

    def _compute_production_metrics(self) -> Dict[str, float]:
        ch4 = np.array(self.ch4_flows)
        total_ch4 = float(np.sum(ch4) * self.step_size_days)
        duration  = self.step_count * self.step_size_days
        return {
            'total_ch4_m3':           total_ch4,
            'avg_ch4_flow':           float(np.mean(ch4)),
            'std_ch4_flow':           float(np.std(ch4)),
            'min_ch4_flow':           float(np.min(ch4)),
            'max_ch4_flow':           float(np.max(ch4)),
            'volumetric_productivity': total_ch4 / (self.V_liq * duration),
        }

    def _compute_safety_metrics(self) -> Dict[str, Any]:
        ph_array  = np.array(self.ph_values)
        vfa_array = np.array(self.vfa_values)
        nh3_array = np.array(self.nh3_values)

        ph_count  = len(self.ph_violations)
        vfa_count = len(self.vfa_violations)
        nh3_count = len(self.nh3_violations)

        # violation_rate: fraction of steps with at least one constraint
        # violated.  Union over channels so the result is in [0, 1].
        steps_with_violation = len(
            set(s for s, _ in self.ph_violations)
            | set(s for s, _ in self.vfa_violations)
            | set(s for s, _ in self.nh3_violations)
        )

        ph_mag  = [m for _, m in self.ph_violations]
        vfa_mag = [m for _, m in self.vfa_violations]
        nh3_mag = [m for _, m in self.nh3_violations]

        return {
            # Counts
            'total_violations':    ph_count + vfa_count + nh3_count,
            'steps_with_violation': steps_with_violation,
            'violation_rate':      float(steps_with_violation / self.step_count),
            'ph_violation_count':  ph_count,
            'vfa_violation_count': vfa_count,
            'nh3_violation_count': nh3_count,
            # Exceedance magnitudes
            'ph_max_magnitude':  float(np.max(ph_mag))  if ph_mag  else 0.0,
            'vfa_max_magnitude': float(np.max(vfa_mag)) if vfa_mag else 0.0,
            'nh3_max_magnitude': float(np.max(nh3_mag)) if nh3_mag else 0.0,
            # Max consecutive violation run per channel (in steps)
            'ph_max_duration':  self._compute_violation_duration(self.ph_violations),
            'vfa_max_duration': self._compute_violation_duration(self.vfa_violations),
            'nh3_max_duration': self._compute_violation_duration(self.nh3_violations),
            # State statistics
            'ph_mean':  float(np.mean(ph_array)),
            'ph_std':   float(np.std(ph_array)),
            'ph_min':   float(np.min(ph_array)),
            'ph_max':   float(np.max(ph_array)),
            'vfa_mean': float(np.mean(vfa_array)),
            'vfa_max':  float(np.max(vfa_array)),
            'nh3_mean': float(np.mean(nh3_array)),
            'nh3_max':  float(np.max(nh3_array)),
        }

    def _compute_violation_duration(
        self, violations: List[Tuple[int, float]]
    ) -> int:
        """Return the length of the longest consecutive run of violated steps."""
        if not violations:
            return 0
        steps = [s for s, _ in violations]
        max_run = current_run = 1
        for i in range(1, len(steps)):
            if steps[i] == steps[i - 1] + 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        return max_run

    def _compute_stability_metrics(self) -> Dict[str, float]:
        ch4 = np.array(self.ch4_flows)
        ph  = np.array(self.ph_values)
        vfa = np.array(self.vfa_values)
        ch4_mean = float(np.mean(ch4))
        return {
            'ch4_volatility':               float(np.mean(np.abs(np.diff(ch4)))),
            'ph_volatility':                float(np.mean(np.abs(np.diff(ph)))),
            'vfa_volatility':               float(np.mean(np.abs(np.diff(vfa)))),
            'ch4_coefficient_of_variation': float(np.std(ch4) / ch4_mean)
                                            if ch4_mean > 0 else 0.0,
        }

    def _compute_control_metrics(self) -> Dict[str, float]:
        q_ad = np.array(self.q_ad_values)
        fm   = np.array(self.feed_mult_values)
        return {
            'q_ad_mean':        float(np.mean(q_ad)),
            'q_ad_std':         float(np.std(q_ad)),
            'q_ad_min':         float(np.min(q_ad)),
            'q_ad_max':         float(np.max(q_ad)),
            'q_ad_effort':      float(np.mean(np.abs(np.diff(q_ad)))),
            'feed_mult_mean':   float(np.mean(fm)),
            'feed_mult_std':    float(np.std(fm)),
            'feed_mult_effort': float(np.mean(np.abs(np.diff(fm)))),
        }

    def _compute_economic_metrics(self) -> Dict[str, float]:
        ch4  = np.array(self.ch4_flows)
        q_ad = np.array(self.q_ad_values)

        total_ch4       = float(np.sum(ch4) * self.step_size_days)
        ch4_energy_kwh  = total_ch4 * 10.0   # ~10 kWh/m³ CH4

        # Simplified pump model: power scales quadratically with flow rate,
        # normalised to 1 kW at the nominal operating point (178 m³/d).
        pump_kw         = (q_ad / self._nominal_flow) ** 2
        pump_energy_kwh = float(np.sum(pump_kw) * self.step_size_days * 24.0)

        net_energy_kwh  = ch4_energy_kwh - pump_energy_kwh
        efficiency      = (ch4_energy_kwh / pump_energy_kwh
                           if pump_energy_kwh > 0 else 0.0)

        return {
            'ch4_energy_kwh':        ch4_energy_kwh,
            'pump_energy_kwh':       pump_energy_kwh,
            'net_energy_kwh':        net_energy_kwh,
            'energy_efficiency_ratio': float(efficiency),
        }

    def _compute_summary_score(self, metrics: Dict) -> Dict[str, float]:
        """
        Composite score used for internal controller ranking.

        Components (weights: production 40%, safety 40%, stability 20%):
            production_score = avg_ch4_flow / 2000  (normalised to ~[0,1]
                               for typical mesophilic digester output)
            safety_score     = 1 - violation_rate   (in [0, 1])
            stability_score  = 1 - min(CV, 1)       (in [0, 1])

        A penalty of -1 is applied if the episode terminated early.

        Note: this score is used only for ranking within this evaluation
        script.  The paper's primary metrics are avg_ch4_flow and
        violation_rate reported separately.
        """
        # 2000 m³/d ≈ upper bound of typical mesophilic BSM2 CH4 output,
        # chosen to map avg_ch4_flow into roughly [0, 1].
        production_score = metrics['production']['avg_ch4_flow'] / 2000.0
        safety_score     = 1.0 - min(1.0, metrics['safety']['violation_rate'])
        stability_score  = 1.0 - min(1.0, metrics['stability']['ch4_coefficient_of_variation'])
        termination_penalty = -1.0 if self.terminated_early else 0.0

        overall_score = (
            0.4 * production_score
            + 0.4 * safety_score
            + 0.2 * stability_score
            + termination_penalty
        )

        return {
            'production_score': float(production_score),
            'safety_score':     float(safety_score),
            'stability_score':  float(stability_score),
            'overall_score':    float(overall_score),
        }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("MetricsCalculator — unit test")
    print("-" * 40)

    rng  = np.random.default_rng(seed=0)
    calc = MetricsCalculator()

    for step in range(100):
        # observation is not used by add_step; pass a dummy array
        obs    = np.zeros(13)
        action = np.array([180.0, 1.0, 0.0])
        reward = 0.8
        info = {
            'q_ch4':          1700 + rng.standard_normal() * 50,
            'pH':             7.2  + rng.standard_normal() * 0.1,
            'total_vfa':      0.15 + rng.standard_normal() * 0.02,
            'S_nh3':          0.001 + abs(rng.standard_normal()) * 0.0005,
            'q_ad':           180.0,
            'feed_multiplier': 1.0,
        }
        if 50 < step < 60:
            info['total_vfa'] = 0.22   # inject VFA violation

        calc.add_step(obs, action, reward, info)

    metrics = calc.compute_metrics()
    calc.print_summary(metrics)

    vr = metrics['safety']['violation_rate']
    assert 0.0 <= vr <= 1.0, f"violation_rate out of [0,1]: {vr}"
    print(f"\nAll assertions passed.  violation_rate={vr:.4f}")
