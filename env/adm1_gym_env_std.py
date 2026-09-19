#!/usr/bin/env python3
"""
ADM1 Gymnasium Environment — Standard (No Thermal Extension)
=============================================================

Gymnasium-compatible environment for RL-based control of anaerobic digestion
using the *standard* ADM1 model (no dynamic temperature states, no Q_HEX).
Temperature is fixed at 35 °C throughout.

Used for the cross-model comparison experiments: all controllers are evaluated
on the four load-variation scenarios (nominal, high_load, low_load, shock_load),
with thermal scenarios excluded.

Key differences vs ADM1Env_v2 (adm1_gym_env.py):
  - Uses ADM1SolverStd (38-state ODE, no T_L/T_a)
  - Action space is 2-dim: [q_ad, feed_mult]   (no Q_HEX)
  - Full obs is 12-dim: drops T_L_norm (index 12 of ADM1Env_v2)
  - Simple obs is 4-dim: [pH, q_ch4, q_ad_current, feed_mult_current]
  - Only four scenarios supported: nominal, high_load, low_load, shock_load
  - No thermal parameter initialisation in reset()
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    from .adm1_solver_std import ADM1SolverStd
    from .scenario_manager import ScenarioManager
except ImportError:
    from adm1_solver_std import ADM1SolverStd
    from scenario_manager import ScenarioManager


# Scenarios supported by this environment (temperature scenarios excluded)
STD_SCENARIOS = ['low_load', 'nominal', 'plant_load', 'high_load', 'peak_load',
                 'fog_surge', 'elevated_start', 'acidified_recovery']


class ADM1Env_Std(gym.Env):
    """
    Gymnasium environment wrapping the standard (non-thermal) ADM1 solver.

    Observation space — full (12-dim, obs_mode='full', default):
        idx  variable            range                description
        ---  --------            -----                -----------
         0   total_vfa           [0, 1.2]  kg COD/m³   S_ac+S_pro+S_bu+S_va
         1   alkalinity          [0, 0.3]  kmol/m³      S_hco3_ion + S_NH3 (DAE-solved)
         2   vfa_alk_ratio       [0, 2.0]  —             total_vfa / alkalinity
         3   S_h2                [0, 1e-4] kg COD/m³   dissolved hydrogen
         4   pH                  [5.5, 8.5] —
         5   S_nh3               [0, 0.01] kmol N/m³   free ammonia
         6   S_IN                [0, 0.2]  kmol N/m³   inorganic nitrogen
         7   X_ac                [0, 3.0]  kg COD/m³   acetoclastic biomass
         8   X_h2                [0, 3.0]  kg COD/m³   hydrogenotrophic biomass
         9   q_ch4               [0, 4000] m³/day       methane flow rate
        10   q_ad_current        [50, 300] m³/day       current feed flow
        11   feed_mult_current   [0.7, 1.3] —           current feed multiplier
        (no T_L_norm — temperature is fixed at 35 °C in the standard model)

    SCADA observation (5-dim, obs_mode='scada'):
        [vfa_alk_ratio, pH, q_ch4, q_ad_current, feed_mult_current]
        = full_obs indices [2, 4, 9, 10, 11]
        All five signals are measurable from SCADA (real-time) or daily LAB data.
        Grounded in Muscatine WRRF dataset (Hunter & Schroer 2023).

    Action space (2-dim continuous):
        [q_ad (m³/day), feed_mult]
        bounds: [50, 300] × [0.7, 1.3]

    Safety violation thresholds (same as ADM1Env_v2):
        pH < 6.8 or pH > 7.8,  VFA > 0.159 kg COD/m³,  NH3 > 0.004 kmol/m³

    Episode termination (catastrophic failure only):
        pH < 5.8 or > 8.8,  NH3 > 0.01,  VFA > 0.535

    Supported scenarios: nominal, high_load, low_load, shock_load
    """

    # Reactor geometry and feed range, taken from the reference facility rather
    # than from the BSM2 benchmark digester the environment previously used.
    # Muscatine WRRF operates two 485,000 gal (1836 m3) mesophilic CSTRs
    # (Hunter & Schroer 2024).  The feed range is that facility's own per-
    # digester loading over 2020-2023: the 5th and 95th percentiles of
    # (TWAS + PS + HSW + FOG) split evenly between the two digesters, which
    # correspond to hydraulic retention times of 45 d and 12 d against a
    # median of 19.5 d.  The BSM2 volume with the old 50-300 m3/d range gave a
    # nominal HRT of 15.5 d; keeping that range at the corrected volume would
    # give 8.3 d, well short of anything this facility runs.
    V_LIQ_M3       = 1836.0
    V_GAS_M3       = 162.0            # BSM2 headspace fraction (300/3400)
    Q_AD_MIN_M3D   = 41.0             # plant p05, HRT 44.8 d
    Q_AD_MAX_M3D   = 159.0            # plant p95, HRT 11.5 d


    metadata = {'render_modes': ['human']}

    # Indices in the full 12-dim observation used by 'scada' mode
    # All 5 signals are measurable from SCADA (real-time) or daily LAB:
    #   vfa_alk_ratio → Dig1-FOS-TAC (daily LAB, key safety signal)
    #   pH            → Dig1-pH (daily LAB)
    #   q_ch4         → Biogas × 0.65 (SCADA real-time)
    #   q_ad_current  → Q-TWAS_GPM (SCADA real-time)
    #   feed_mult     → controlled input
    # ── Safety thresholds ─────────────────────────────────────────────────
    # VFA, as kg COD/m3.  Acetic acid: 60 g/mol, 64 g COD/mol, so
    # 1 mg/L acetic = 1.067e-3 kg COD/m3.
    # VFA limits are taken from the operating ranges reported for true
    # (chromatographically determined) VFA in mesophilic sludge digesters,
    # not from a plant's titrimetric record.  The two are not commensurable:
    # a two-point titration reads the pH 5.0-4.4 leg, where carbonate
    # contributes roughly eight times the acetate signal at the bicarbonate
    # levels typical of municipal digestion, so titrimetric VFA over-reads
    # true VFA by an order of magnitude.  Chemostat theory bounds the true
    # value independently: with growth balancing decay, steady-state acetate
    # is K_S*k_dec/(mu_max*I - k_dec), i.e. single to low double digits mg/L,
    # irrespective of biomass.  This environment holds 61-221 mg/L across its
    # scenarios at nominal feed, inside the reported normal range.
    #
    #   soft  300 mg/L as HAc  - upper end of normal operation
    #   hard  1500 mg/L as HAc - onset of methanogenic inhibition
    #
    # Converted with 64 g COD per 60 g acetic acid.
    VFA_SOFT_KGCOD_M3 = 0.320   # 300 mg/L as acetic acid
    VFA_HARD_KGCOD_M3 = 1.600   # 1500 mg/L as acetic acid
    # Free ammonia, kmol N/m3, expressed as multiples of ADM1's 50 %
    # inhibition constant for aceticlastic methanogens (K_I_nh3 = 0.0018,
    # Batstone et al. 2002): soft = 2.2x, hard = 5.6x.
    NH3_SOFT_KMOL_M3 = 0.004
    NH3_HARD_KMOL_M3 = 0.010
    # pH.  6.8-7.8 is the range for stable mesophilic methanogenesis; below
    # 5.8 aceticlastic methanogens are outside their viable range (ADM1 sets
    # complete pH inhibition of acetate uptake at pH_LL_ac = 6.0).
    PH_SOFT_LO, PH_SOFT_HI = 6.8, 7.8
    PH_HARD_LO, PH_HARD_HI = 5.8, 8.8

    # The reference facility records all six of these daily: VFA and
    # alkalinity as separate analyses, pH, biogas flow, and the feed
    # volumes that set flow rate and strength.  The constraint is on VFA,
    # so VFA is observed directly rather than only through its ratio to
    # alkalinity, which varies fourfold across these scenarios.
    SCADA_OBS_INDICES = [0, 1, 4, 9, 10, 11]
    # obs_mode='trend' appends three quantities a plant can compute from the
    # same daily record: the day-on-day change in VFA and in methane flow, and
    # the mean feed rate over the past week.  VFA accumulation is a trend, and
    # the instantaneous set above cannot distinguish a level that is steady
    # from the same level still rising.
    TREND_HISTORY_DAYS = 7
    SIMPLE_OBS_INDICES = SCADA_OBS_INDICES  # backward compat alias
    # [vfa_alk_ratio, pH, q_ch4, q_ad_current, feed_mult_current]

    def __init__(
        self,
        scenario_name: str = 'nominal',
        step_size: float = 0.01041667,   # 15 minutes in days
        influent_mode: str = 'direct',   # 'direct' (BSM2, cited) | 'composite' | 'mws_gtw'
        V_liq: float = V_LIQ_M3,
        V_gas: float = V_GAS_M3,
        reward_config: Optional[Dict] = None,
        enable_disturbances: bool = True,
        random_seed: Optional[int] = None,
        obs_mode: str = 'full',          # 'full' (12-dim) | 'scada' (5-dim)
        substrate: Optional[Dict[str, float]] = None,
        vfa_soft: Optional[float] = None,
        vfa_hard: Optional[float] = None,
    ):
        """
        Initialise the standard ADM1 environment.

        Args:
            scenario_name:       One of STD_SCENARIOS.
            vfa_soft, vfa_hard:  Override the derived VFA limits.  Used only for
                                 the threshold-robustness study; leaving them at
                                 None keeps the values derived in
                                 the literature operating ranges.
            step_size:           Simulation timestep in days (default: 15 min).
            V_liq:               Liquid volume (m³).
            V_gas:               Gas headspace volume (m³).
            reward_config:       Custom reward configuration dict (optional).
            enable_disturbances: Inject scenario disturbances (e.g. shock_load spike).
            random_seed:         RNG seed for reproducibility.
            obs_mode:            'full' (12-dim) or 'scada' (5-dim, SCADA-grounded).
        """
        super().__init__()

        if scenario_name not in STD_SCENARIOS:
            raise ValueError(
                f"Scenario '{scenario_name}' not supported by ADM1Env_Std. "
                f"Supported: {STD_SCENARIOS}"
            )

        self.scenario_name       = scenario_name
        self.step_size           = step_size
        self.influent_mode       = influent_mode
        self.substrate           = substrate
        self.V_liq               = V_liq
        self.V_gas               = V_gas
        self.enable_disturbances = enable_disturbances
        self._seed               = random_seed
        self.obs_mode            = obs_mode
        # Instance-level limits; class defaults unless the robustness
        # study overrides them.
        if vfa_soft is not None:
            self.VFA_SOFT_KGCOD_M3 = float(vfa_soft)
        if vfa_hard is not None:
            self.VFA_HARD_KGCOD_M3 = float(vfa_hard)

        # Scenario configuration
        self.scenario_manager  = ScenarioManager()
        self.scenario_config   = self.scenario_manager.load_scenario(scenario_name)
        self.scenario_duration = self.scenario_config['duration_days']
        self.max_steps         = int(self.scenario_duration / step_size)

        # Reward configuration
        self.reward_config = reward_config or {
            'production_scale':     2000.0,
            'ph_penalty_scale':     2.0,
            'vfa_penalty_scale':    3.0,
            'nh3_penalty_scale':    50.0,
            'energy_penalty_max':   0.2,
            'stability_penalty_max': 0.1,
        }

        # ── Action space: [q_ad, feed_mult] — 2-dim, no Q_HEX ────────────────
        self.action_space = spaces.Box(
            low=np.array([self.Q_AD_MIN_M3D, 0.7], dtype=np.float32),
            high=np.array([self.Q_AD_MAX_M3D, 1.3], dtype=np.float32),
            dtype=np.float32,
        )

        # ── Observation space ─────────────────────────────────────────────────
        _full_low = np.array([
            0.0,   # 0:  total_vfa
            0.0,   # 1:  alkalinity
            0.0,   # 2:  vfa_alk_ratio
            0.0,   # 3:  S_h2
            6.5,   # 4:  pH               (observed min 7.03)   (hard threshold 5.8)
            0.0,   # 5:  S_nh3
            0.0,   # 6:  S_IN
            0.0,   # 7:  X_ac
            0.0,   # 8:  X_h2
            0.0,   # 9:  q_ch4
            self.Q_AD_MIN_M3D,  # 10: q_ad_current
            0.7,   # 11: feed_mult_current
        ], dtype=np.float32)
        # Upper bounds are set from the ranges the environment actually reaches
        # across the loading scenarios, and must bracket the safety thresholds
        # so the danger region stays resolvable after normalisation.  The two
        # echoed action channels use the action-space bounds exactly.
        _full_high = np.array([
            2.5,    # 0:  total_vfa        (hard threshold 1.60, observed 2.02)
            0.25,   # 1:  alkalinity       (observed 0.204)
            16.0,   # 2:  vfa_alk_ratio    (observed 14.8)
            1.2e-6, # 3:  S_h2             (observed 9.5e-7)
            8.0,    # 4:  pH               (observed 7.61; hard bound 8.8)
            0.011,  # 5:  S_nh3            (hard threshold 0.010, observed 0.0087)
            0.25,   # 6:  S_IN             (observed 0.210)
            2.2,    # 7:  X_ac             (observed 1.75)
            1.0,    # 8:  X_h2             (observed 0.77)
            4500.0, # 9:  q_ch4            (observed 3889)
            self.Q_AD_MAX_M3D,  # 10: q_ad_current     (= action upper bound)
            1.3,    # 11: feed_mult_current(= action upper bound)
        ], dtype=np.float32)

        if obs_mode in ('scada', 'simple'):
            obs_low  = _full_low[self.SCADA_OBS_INDICES]
            obs_high = _full_high[self.SCADA_OBS_INDICES]
        elif obs_mode in ('trend', 'dev'):
            # Bounds follow the same convention as the base observation: just
            # above the range the environment actually reaches, measured over
            # the scenario grid (dVFA -0.287 to +0.405, dCH4 -1324 to +2939).
            # The former +-1.0 on dVFA left that channel using a third of its
            # range, and the former +-2000 on dCH4 clipped its upper tail.
            obs_low  = np.concatenate([_full_low[self.SCADA_OBS_INDICES],
                                       np.array([-0.45, -3000.0, self.Q_AD_MIN_M3D],
                                                dtype=np.float32)])
            obs_high = np.concatenate([_full_high[self.SCADA_OBS_INDICES],
                                       np.array([0.45, 3000.0, self.Q_AD_MAX_M3D],
                                                dtype=np.float32)])
        else:
            obs_low, obs_high = _full_low, _full_high

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # ── Solver ────────────────────────────────────────────────────────────
        self.solver = ADM1SolverStd(V_liq=V_liq, V_gas=V_gas)

        # ── Internal state ────────────────────────────────────────────────────
        self.current_step        = 0
        self.current_time_days   = 0.0
        self.current_state       = None
        self.q_ch4               = 0.0
        self.prev_q_ch4          = 0.0
        self._prev_vfa           = None
        from collections import deque
        self._q_history          = deque(maxlen=self.TREND_HISTORY_DAYS)
        self._vfa_history        = deque(maxlen=self.TREND_HISTORY_DAYS)
        self._ch4_history        = deque(maxlen=self.TREND_HISTORY_DAYS)
        self.total_ch4_produced  = 0.0
        self.episode_reward      = 0.0
        self.q_ad_current        = 178.4674
        self.feed_mult_current   = 1.0

        self.ph_history  = []
        self.vfa_history = []
        self.ch4_history = []
        self.violation_count = {
            'ph_low': 0, 'ph_high': 0, 'vfa_high': 0, 'nh3_high': 0
        }

        self._load_data()

    # ──────────────────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_data(self):
        data_path = Path(__file__).parent / 'data' / 'digester_influent.csv'
        if not data_path.exists():
            raise FileNotFoundError(
                f"Influent data not found at {data_path}.\n"
                f"Expected: env/data/digester_influent.csv"
            )
        self.influent_df = pd.read_csv(data_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_influent_dict(self, step: int = 0) -> Dict[str, float]:
        sim_time   = step * self.step_size
        csv_index  = int(sim_time / 0.01041667)
        csv_index  = min(csv_index, len(self.influent_df) - 1)

        # The influent CSV is at 15-min resolution and varies substantially
        # within a day.  Point-sampling one row per control step would alias
        # the feed whenever the control interval exceeds 15 min (a daily step
        # would keep 1 sample in 96), so average across the rows the step
        # actually spans.
        n_rows = max(1, int(round(self.step_size / 0.01041667)))
        if n_rows > 1:
            end = min(csv_index + n_rows, len(self.influent_df))
            row = self.influent_df.iloc[csv_index:end].mean(numeric_only=True)
        else:
            row = self.influent_df.iloc[csv_index]

        required = [
            'S_su', 'S_aa', 'S_fa', 'S_va', 'S_bu', 'S_pro', 'S_ac',
            'S_h2', 'S_ch4', 'S_IC', 'S_IN', 'S_I',
            'X_xc', 'X_ch', 'X_pr', 'X_li', 'X_su', 'X_aa', 'X_fa',
            'X_c4', 'X_pro', 'X_ac', 'X_h2', 'X_I',
            'S_cation', 'S_anion',
        ]
        inf = {v: float(row[v]) if v in row else 0.0 for v in required}

        if self.substrate is not None:
            inf = self._reallocate_particulate_cod(inf, self.substrate)
        if self.influent_mode == 'composite':
            inf = self._to_composite(inf)
        elif self.influent_mode == 'mws_gtw':
            inf = self._to_mws_gtw(inf)
        return inf

    def _reallocate_particulate_cod(self, inf: Dict[str, float],
                                    fractions: Dict[str, float]) -> Dict[str, float]:
        """
        Redistribute the biodegradable particulate COD across carbohydrate,
        protein and lipid, holding their total constant.  Inert COD is left
        alone: the scenario multiplier scales only the biodegradable species,
        so reallocating across the inert as well would change the delivered
        organic load along with the composition.

        Nitrogen enters this environment's feed only through protein, since the
        influent supplies X_ch, X_pr and X_li directly rather than through the
        composite/disintegration path.  Changing the protein share therefore
        changes the ammonia released on degradation consistently, with no
        separate nitrogen bookkeeping - which is what makes composition the
        right axis for asking whether a digester can be acidified at all.
        """
        keys = ('X_ch', 'X_pr', 'X_li')
        total = sum(inf.get(k, 0.0) for k in keys)
        f_sum = sum(fractions.get(k, 0.0) for k in keys)
        if total <= 0 or f_sum <= 0:
            return inf
        out = dict(inf)
        for k in keys:
            out[k] = total * fractions.get(k, 0.0) / f_sum
        return out

    # ── Influent characterisation: municipal sludge + grease trap waste ──
    # COD fractions and feed VFA/alkalinity from Razaviarani & Buchanan,
    # Chem. Eng. J. 266 (2015) 91-99 (Table 1 "MWS1", Table 2 "MWS1 + GTW").
    # The BSM2 influent shipped with this environment is protein-dominant and
    # carries no VFA or feed alkalinity, whereas a digester co-digesting grease
    # trap waste is lipid-dominant and receives partly acidified sludge.
    _MWS_GTW_COD_FRACTIONS = {          # fractions of total influent COD
        'X_ch': 0.155, 'X_pr': 0.040, 'X_li': 0.510, 'X_I': 0.011, 'S_I': 0.284,
    }
    # Feed VFA, Table 1 column MWS1 (mg/L), with theoretical oxygen demands
    # (g COD per g acid) used to convert to the ADM1 COD basis.
    _MWS_FEED_VFA_MGL = {'ac': 853.0, 'pro': 875.0, 'bu': 108.0 + 611.0,
                         'va': 208.0 + 328.0}
    _VFA_COD_PER_G    = {'ac': 64.0/60.0, 'pro': 112.0/74.0,
                         'bu': 160.0/88.0, 'va': 208.0/102.0}
    _MWS_FEED_ALK_MGL_CACO3 = 1850.0    # Table 1, MWS1 total alkalinity
    _MWS_FEED_TAN_MGN_L     = 346.0     # Table 1, MWS1 total ammonia nitrogen

    def _to_mws_gtw(self, inf: Dict[str, float]) -> Dict[str, float]:
        """
        Re-characterise the influent as municipal sludge co-digested with
        grease trap waste, preserving the total influent COD of the shipped
        BSM2 feed so the organic loading rate is unchanged.
        """
        total_cod = (inf['X_ch'] + inf['X_pr'] + inf['X_li'] + inf['X_I']
                     + inf['S_I'] + inf['X_xc'])
        if total_cod <= 0.0:
            return inf
        out = dict(inf)
        out['X_xc'] = 0.0
        for k, f in self._MWS_GTW_COD_FRACTIONS.items():
            out[k] = total_cod * f
        # Feed VFA (kg COD/m3 = g COD/L)
        for k, mgl in self._MWS_FEED_VFA_MGL.items():
            out['S_' + k] = mgl * 1e-3 * self._VFA_COD_PER_G[k]
        # Feed alkalinity as an equivalent cation charge (kmol/m3)
        out['S_cation'] = self._MWS_FEED_ALK_MGL_CACO3 / 50000.0
        # Feed ammonia.  Sludge arrives with a substantial ammonia load, which
        # is the digester's main buffer source; the shipped BSM2 feed carries
        # an order of magnitude less.  N: 14 g/mol.
        out['S_IN'] = self._MWS_FEED_TAN_MGN_L / 14.0 / 1000.0
        return out

    # Fraction of X_xc that disintegrates into degradable ch/pr/li.
    # Must match the solver's f_ch_xc + f_pr_xc + f_li_xc (0.2 + 0.2 + 0.3).
    _F_DEG_XC = 0.7

    def _to_composite(self, inf: Dict[str, float]) -> Dict[str, float]:
        """
        Route particulate influent through the composite (X_xc) disintegration
        step instead of feeding X_ch/X_pr/X_li directly.

        The CSV characterises the feed as already-disintegrated particulates,
        which bypasses disintegration (k_dis, ~2 d) and leaves hydrolysis
        (k_hyd = 10 /d, ~2.4 h) as the only lag.  The digester then tracks
        daily feed swings almost instantaneously, which is not how a
        municipal digester behaves.  Routing through X_xc restores the
        rate-limiting step so VFA accumulates over days.

        Total particulate COD is preserved: the inert COD that disintegration
        releases from X_xc is deducted from the directly-fed X_I.
        """
        deg = inf['X_ch'] + inf['X_pr'] + inf['X_li']
        if deg <= 0.0:
            return inf
        part_total = deg + inf['X_I']
        x_xc = deg / self._F_DEG_XC

        # Set the disintegration yields so that X_xc releases carbohydrate,
        # protein and lipid in the same proportions the influent originally
        # specified.  Leaving them at the ADM1 defaults (0.2/0.2/0.3) would
        # silently re-characterise a protein-dominant feed as lipid-rich,
        # which shifts the steady state and depresses pH below the soft limit.
        self.solver.f_ch_xc = self._F_DEG_XC * inf['X_ch'] / deg
        self.solver.f_pr_xc = self._F_DEG_XC * inf['X_pr'] / deg
        self.solver.f_li_xc = self._F_DEG_XC * inf['X_li'] / deg

        # ADM1's nitrogen balance closes only when the composite's nitrogen
        # content matches what disintegration releases:
        #     N_xc = f_xI_xc*N_I + f_sI_xc*N_I + f_pr_xc*N_aa
        # The published parameter set satisfies this at f_pr_xc = 0.2.  Having
        # just changed f_pr_xc, N_xc must be recomputed or disintegration
        # becomes a net sink of inorganic nitrogen, collapsing ammonia,
        # alkalinity and pH.
        so = self.solver
        so.N_xc = (so.f_xI_xc * so.N_I + so.f_sI_xc * so.N_I
                   + so.f_pr_xc * so.N_aa)

        out = dict(inf)
        out['X_ch'] = out['X_pr'] = out['X_li'] = 0.0
        out['X_xc'] = x_xc
        out['X_I']  = max(0.0, part_total - x_xc)
        return out

    def _calculate_total_vfa(self, state: Dict[str, float]) -> float:
        return (state.get('S_ac', 0.0) + state.get('S_pro', 0.0)
                + state.get('S_bu', 0.0) + state.get('S_va', 0.0))

    def _calculate_alkalinity(self, state: Dict[str, float]) -> float:
        """
        Titration alkalinity (TAC), in kmol charge equivalents per m3.

        A plant's TAC is measured by titrating to pH ~4.3, which neutralises
        bicarbonate *and* the conjugate bases of the VFAs.  Counting only
        bicarbonate makes the simulated denominator fall as VFA rises, whereas
        the measured one is partly buffered by the VFA salts themselves - the
        two move in opposite directions, so the simulated ratio is not the
        quantity the plant reports.  The VFA ion states are in kg COD/m3 and
        are converted with their COD equivalents (64, 112, 160, 208 g COD/mol),
        the same divisors the charge balance uses.
        """
        hco3 = state.get('S_hco3_ion', None)
        if hco3 is None:
            hco3 = state.get('S_IC', 0.0) * 0.876
        vfa_ions = (state.get('S_ac_ion',  0.0) / 64.0
                    + state.get('S_pro_ion', 0.0) / 112.0
                    + state.get('S_bu_ion',  0.0) / 160.0
                    + state.get('S_va_ion',  0.0) / 208.0)
        return hco3 + vfa_ions + state.get('S_nh3', 0.0)


    # Nordmann two-point titration: 0.1 N H2SO4 on a 20 mL sample, so one mL
    # of titrant adds 0.005 kmol H+ per m3 of sample.
    _KMOL_PER_ML   = 0.005
    _TAC_PER_ML    = 250.0            # mg CaCO3/L per mL to pH 5.0
    _FOS_SLOPE     = 1.66             # Nordmann empirical coefficients for the
    _FOS_OFFSET    = 0.15             # pH 5.0 -> 4.4 leg
    _FOS_PER_ML    = 500.0

    def _titrant_to_pH(self, state: Dict[str, float], target_pH: float) -> float:
        """
        Strong acid (kmol/m3) needed to bring the sample to target_pH.

        The solver's charge balance is linear in added strong acid, so the
        requirement is just the charge imbalance evaluated at the target pH.
        """
        sv  = self.solver
        S_H = 10.0 ** (-target_pH)
        S_va, S_bu   = state.get('S_va', 0.0),  state.get('S_bu', 0.0)
        S_pro, S_ac  = state.get('S_pro', 0.0), state.get('S_ac', 0.0)
        S_IC, S_IN   = state.get('S_IC', 0.0),  state.get('S_IN', 0.0)
        va   = sv.K_a_va  * S_va  / (sv.K_a_va  + S_H)
        bu   = sv.K_a_bu  * S_bu  / (sv.K_a_bu  + S_H)
        pro  = sv.K_a_pro * S_pro / (sv.K_a_pro + S_H)
        ac   = sv.K_a_ac  * S_ac  / (sv.K_a_ac  + S_H)
        hco3 = sv.K_a_co2 * S_IC  / (sv.K_a_co2 + S_H)
        nh3  = sv.K_a_IN  * S_IN  / (sv.K_a_IN  + S_H)
        return (state.get('S_cation', 0.0) + (S_IN - nh3) + S_H
                - hco3 - ac / 64.0 - pro / 112.0 - bu / 160.0 - va / 208.0
                - sv.K_w / S_H - state.get('S_anion', 0.0))

    def _simulated_nordmann(self, state: Dict[str, float]):
        """
        FOS and TAC (mg/L) as a Nordmann titration would read them on this
        state, rather than the model's true VFA concentration.

        The two differ substantially: carbonate interferes with the acetate
        leg of the titration, so titrimetric FOS over-reads true VFA, markedly
        so at the bicarbonate levels typical of municipal digesters.  Comparing
        a simulated true VFA against a plant's titrimetric record therefore
        compares incommensurable quantities - on this environment the two
        differ by a factor of about 13, of which the measurement definition
        accounts for roughly seven.
        """
        a50 = self._titrant_to_pH(state, 5.0)
        a44 = self._titrant_to_pH(state, 4.4)
        mL_tac = max(a50, 0.0) / self._KMOL_PER_ML
        mL_fos = max(a44 - a50, 0.0) / self._KMOL_PER_ML
        tac = mL_tac * self._TAC_PER_ML
        fos = max((mL_fos * self._FOS_SLOPE - self._FOS_OFFSET) * self._FOS_PER_ML, 0.0)
        return fos, tac

    def _calculate_fos_tac(self, state: Dict[str, float]) -> float:
        """FOS/TAC as the plant's instrument would report it (dimensionless)."""
        fos, tac = self._simulated_nordmann(state)
        return fos / max(tac, 1e-9)

    # ──────────────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)

        # Reset counters
        self.current_step       = 0
        self.current_time_days  = 0.0
        self.total_ch4_produced = 0.0
        self.prev_q_ch4         = 0.0
        self.q_ch4              = 0.0
        self.episode_reward     = 0.0
        self.q_ad_current       = 178.4674
        self.feed_mult_current  = 1.0

        self.ph_history  = []
        self.vfa_history = []
        self.ch4_history = []
        self.violation_count = {
            'ph_low': 0, 'ph_high': 0, 'vfa_high': 0, 'nh3_high': 0
        }

        # Support runtime scenario switching
        if options and 'scenario' in options:
            new_scenario = options['scenario']
            if new_scenario not in STD_SCENARIOS:
                raise ValueError(
                    f"Scenario '{new_scenario}' not supported. Use one of {STD_SCENARIOS}"
                )
            self.scenario_name    = new_scenario
            self.scenario_config  = self.scenario_manager.load_scenario(new_scenario)
            self.scenario_duration = self.scenario_config['duration_days']
            self.max_steps        = int(self.scenario_duration / self.step_size)

        # Load initial state
        initial_state = self.scenario_manager.get_initial_state(self.scenario_name)
        self.solver.set_state(initial_state)

        # Set initial influent
        base_influent     = self._build_influent_dict(0)
        modified_influent = self.scenario_manager.apply_influent_multiplier(
            base_influent, self.scenario_name
        )
        self.solver.set_influent(modified_influent)
        self.solver.set_flow_rate(self.q_ad_current)

        self.current_state = self.solver.state.copy()
        observation = self._get_observation()

        info = {
            'scenario':      self.scenario_name,
            'max_steps':     self.max_steps,
            'duration_days': self.scenario_duration,
        }
        return observation, info

    # ──────────────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.

        Args:
            action: 2-dim array [q_ad (m³/day), feed_mult]

        Returns:
            observation, reward, terminated, truncated, info
        """
        q_ad            = np.clip(float(action[0]), self.Q_AD_MIN_M3D, self.Q_AD_MAX_M3D)
        feed_multiplier = np.clip(float(action[1]),  0.7,   1.3)

        self.q_ad_current      = q_ad
        self.feed_mult_current = feed_multiplier
        self.solver.set_flow_rate(q_ad)

        # Build influent
        base_influent     = self._build_influent_dict(self.current_step)
        modified_influent = self.scenario_manager.apply_influent_multiplier(
            base_influent, self.scenario_name
        )

        # Apply feed multiplier to organic substrates
        for key in ['X_ch', 'X_pr', 'X_li', 'X_xc']:
            if key in modified_influent:
                modified_influent[key] *= feed_multiplier

        # Inject disturbances (load-based only; temperature disturbances ignored)
        if self.enable_disturbances:
            disturbance = self.scenario_manager.check_disturbances(self.current_time_days)
            if disturbance and disturbance['type'] != 'temperature_ramp':
                modified_influent = self.scenario_manager.apply_disturbance(
                    modified_influent, disturbance
                )

        self.solver.set_influent(modified_influent)

        # Integrate one step (no Q_HEX)
        try:
            new_state, q_ch4 = self.solver.step(dt=self.step_size)
            self.current_state = new_state
            self.q_ch4         = max(0.0, q_ch4)

            self.current_step       += 1
            self.current_time_days  += self.step_size
            self.total_ch4_produced += self.q_ch4 * self.step_size

            self.ph_history.append(new_state.get('pH', 7.0))
            self.vfa_history.append(self._calculate_total_vfa(new_state))
            self.ch4_history.append(self.q_ch4)

        except Exception as e:
            print(f"Solver error at step {self.current_step}: {e}")
            self.q_ch4 = 0.0

        observation = self._get_observation()
        reward      = self._calculate_reward()
        self.episode_reward += reward

        terminated = self._is_catastrophic_failure()
        truncated  = self.current_step >= self.max_steps

        info = self._build_info_dict(terminated or truncated)
        self._prev_vfa  = self._calculate_total_vfa(self.current_state)
        self._q_history.append(self.q_ad_current)
        self._vfa_history.append(self._calculate_total_vfa(self.current_state))
        self._ch4_history.append(self.q_ch4)
        self.prev_q_ch4 = self.q_ch4

        return observation, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """Build 11-dim (full) or 4-dim (simple) observation vector."""
        s = self.current_state

        total_vfa     = self._calculate_total_vfa(s)
        alkalinity    = self._calculate_alkalinity(s)
        vfa_alk_ratio = total_vfa / max(alkalinity, 1e-6)

        full_obs = np.array([
            total_vfa,                   # 0
            alkalinity,                  # 1
            vfa_alk_ratio,               # 2
            s.get('S_h2',    0.0),       # 3
            s.get('pH',      7.0),       # 4
            s.get('S_nh3',   0.0),       # 5
            s.get('S_IN',    0.0),       # 6
            s.get('X_ac',    0.0),       # 7
            s.get('X_h2',    0.0),       # 8
            self.q_ch4,                  # 9
            self.q_ad_current,           # 10
            self.feed_mult_current,      # 11
        ], dtype=np.float32)

        if self.obs_mode in ('scada', 'simple'):
            return full_obs[self.SCADA_OBS_INDICES]
        if self.obs_mode == 'trend':
            d_vfa = total_vfa - self._prev_vfa if self._prev_vfa is not None else 0.0
            d_ch4 = self.q_ch4 - self.prev_q_ch4
            q_hist = (float(np.mean(self._q_history)) if self._q_history
                      else self.q_ad_current)
            return np.concatenate([
                full_obs[self.SCADA_OBS_INDICES],
                np.array([d_vfa, d_ch4, q_hist], dtype=np.float32)])
        if self.obs_mode == 'dev':
            # Departure from the trailing week rather than from yesterday.  A
            # day-on-day difference is near zero except during a transition and
            # so carries information only rarely; the departure from a weekly
            # mean is informative at every step, and is how an operator reads a
            # trend against recent history.
            v_ref = (float(np.mean(self._vfa_history)) if self._vfa_history
                     else total_vfa)
            c_ref = (float(np.mean(self._ch4_history)) if self._ch4_history
                     else self.q_ch4)
            q_hist = (float(np.mean(self._q_history)) if self._q_history
                      else self.q_ad_current)
            return np.concatenate([
                full_obs[self.SCADA_OBS_INDICES],
                np.array([total_vfa - v_ref, self.q_ch4 - c_ref, q_hist],
                         dtype=np.float32)])
        return full_obs

    # ──────────────────────────────────────────────────────────────────────────
    # Reward
    # ──────────────────────────────────────────────────────────────────────────

    def _calculate_reward(self) -> float:
        rc = self.reward_config

        production_reward = self.q_ch4 / rc['production_scale']
        safety_penalty    = self._calculate_safety_penalty()
        energy_penalty    = -(self.q_ad_current / self.Q_AD_MAX_M3D) ** 2 * rc['energy_penalty_max']
        volatility        = abs(self.q_ch4 - self.prev_q_ch4)
        stability_penalty = -(volatility / 100.0) * rc['stability_penalty_max']

        return production_reward + safety_penalty + energy_penalty + stability_penalty

    def _calculate_safety_penalty(self) -> float:
        s = self.current_state
        rc = self.reward_config
        total_penalty = 0.0
        penalty_type  = rc.get('penalty_type', 'quadratic')

        pH = s.get('pH', 7.0)
        if pH < self.PH_SOFT_LO:
            dev = self.PH_SOFT_LO - pH
            if penalty_type == 'linear+constant':
                total_penalty -= rc.get('ph_constant_penalty', 0.5)
                total_penalty -= dev * rc['ph_penalty_scale']
            else:
                total_penalty -= (dev ** 2) * rc['ph_penalty_scale']
            self.violation_count['ph_low'] += 1
        elif pH > self.PH_SOFT_HI:
            dev = pH - self.PH_SOFT_HI
            if penalty_type == 'linear+constant':
                total_penalty -= rc.get('ph_constant_penalty', 0.5)
                total_penalty -= dev * rc['ph_penalty_scale']
            else:
                total_penalty -= (dev ** 2) * rc['ph_penalty_scale']
            self.violation_count['ph_high'] += 1

        total_vfa = self._calculate_total_vfa(s)
        if total_vfa > self.VFA_SOFT_KGCOD_M3:
            excess = total_vfa - self.VFA_SOFT_KGCOD_M3
            if penalty_type == 'linear+constant':
                total_penalty -= rc.get('vfa_constant_penalty', 1.0)
                total_penalty -= excess * rc['vfa_penalty_scale']
            else:
                total_penalty -= (excess ** 2) * rc['vfa_penalty_scale']
            self.violation_count['vfa_high'] += 1

        S_nh3 = s.get('S_nh3', 0.0)
        if S_nh3 > self.NH3_SOFT_KMOL_M3:
            excess = S_nh3 - self.NH3_SOFT_KMOL_M3
            if penalty_type == 'linear+constant':
                total_penalty -= rc.get('nh3_constant_penalty', 1.5)
                total_penalty -= excess * rc['nh3_penalty_scale']
            else:
                total_penalty -= (excess ** 2) * rc['nh3_penalty_scale']
            self.violation_count['nh3_high'] += 1

        return total_penalty

    def _is_catastrophic_failure(self) -> bool:
        s   = self.current_state
        pH  = s.get('pH', 7.0)
        if pH < self.PH_HARD_LO or pH > self.PH_HARD_HI:
            return True
        if s.get('S_nh3', 0.0) > self.NH3_HARD_KMOL_M3:
            return True
        if self._calculate_total_vfa(s) > self.VFA_HARD_KGCOD_M3:
            return True
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # Info
    # ──────────────────────────────────────────────────────────────────────────

    def _build_info_dict(self, is_done: bool) -> Dict[str, Any]:
        s = self.current_state
        info = {
            'step':            self.current_step,
            'time_days':       self.current_time_days,
            'q_ch4':           self.q_ch4,
            'q_co2':           self.solver.q_co2,
            'pH':              s.get('pH', 7.0),
            'total_vfa':       self._calculate_total_vfa(s),
            'alkalinity':      self._calculate_alkalinity(s),
            'S_nh3':           s.get('S_nh3', 0.0),
            'q_ad':            self.q_ad_current,
            'feed_multiplier': self.feed_mult_current,
            'total_ch4':       self.total_ch4_produced,
        }
        if is_done:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.current_step,
            }
            info['ch4_produced'] = self.total_ch4_produced
            info['ch4_avg_flow'] = (
                self.total_ch4_produced
                / max(1, self.current_step * self.step_size)
            )
            info['avg_ph']           = np.mean(self.ph_history)  if self.ph_history  else 7.0
            info['avg_vfa']          = np.mean(self.vfa_history) if self.vfa_history else 0.0
            info['ch4_std']          = np.std(self.ch4_history)  if self.ch4_history else 0.0
            info['violation_count']  = self.violation_count.copy()
        return info

    # ──────────────────────────────────────────────────────────────────────────
    # Render / close
    # ──────────────────────────────────────────────────────────────────────────

    def render(self, mode: str = 'human'):
        if mode == 'human':
            s         = self.current_state
            total_vfa = self._calculate_total_vfa(s)
            alk       = self._calculate_alkalinity(s)
            print(f"\n=== Step {self.current_step} (Day {self.current_time_days:.2f}) ===")
            print(f"Scenario : {self.scenario_name}")
            print(f"pH       : {s.get('pH', 0):.2f}")
            print(f"CH4 flow : {self.q_ch4:.2f} m³/d")
            print(f"Total VFA: {total_vfa:.4f} kg COD/m³")
            print(f"Alk      : {alk:.4f} kmol/m³")
            print(f"q_ad     : {self.q_ad_current:.1f} m³/d")
            print(f"feed_mult: {self.feed_mult_current:.2f}")

    def close(self):
        pass


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("ADM1Env_Std — smoke test")
    print("=" * 70)

    for sc in STD_SCENARIOS:
        print(f"\n[{sc}]")
        try:
            env = ADM1Env_Std(scenario_name=sc)
            obs, info = env.reset(seed=42)
            print(f"  obs shape  : {obs.shape}")
            print(f"  action shape: {env.action_space.shape}")

            for _ in range(3):
                act = env.action_space.sample()
                obs, rew, term, trunc, info = env.step(act)
            print(f"  step ok — pH={info['pH']:.2f}, q_ch4={info['q_ch4']:.1f}")
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Done.")
