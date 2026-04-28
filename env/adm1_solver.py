#!/usr/bin/env python3
"""
ADM1 Solver — Temperature-Extended ODE Implementation
======================================================

Self-contained implementation of the Anaerobic Digestion Model No. 1 (ADM1,
Batstone et al. 2002) with a dynamic thermal extension enabling liquid-phase
temperature as an ODE state variable.

Thermal extensions (relative to standard BSM2 ADM1):
    - Dynamic energy balance for liquid temperature T_L (K)
    - Microbial adaptation temperature T_a (K) with time constant tau_a = 30 d
    - Arrhenius correction on hydrolysis (Ea = 64 kJ/mol)
    - Cardinal Temperature Model on mu_max for acidogens/acetogens
    - Kovalovszki F_K inhibition factor on methanogens
    - Advective and conductive heat losses; configurable heat exchanger Q_HEX

State vector (40-dimensional):
    [0:38]  standard ADM1 biological/chemical states (BSM2 ordering)
    [38]    T_L  -- liquid temperature (K)
    [39]    T_a  -- microbial adaptation temperature (K)
"""

import numpy as np
import scipy.integrate
from typing import Dict, Tuple


class ADM1Solver:
    """
    Temperature-extended ADM1 ODE solver with single-step execution.

    Integrates the 40-dimensional ODE (38 ADM1 states + T_L + T_a) using
    DOP853 for each environment step, then resolves the DAE (pH, ions, H2)
    via Newton-Raphson (BSM2 standard procedure).
    """

    def __init__(self, V_liq=3400.0, V_gas=300.0):
        """
        Initialize ADM1 solver.

        Args:
            V_liq: Liquid volume (m³), default 3400
            V_gas: Gas headspace volume (m³), default 300

        Note:
            V_liq and V_gas remain CONSTANT during simulation (CSTR assumption).
            Different values can be used for different reactor scales.
        """

        # Physical constants
        self.R = 0.083145  # bar.M^-1.K^-1
        self.T_base = 298.15  # K
        self.p_atm = 1.013  # bar
        self.T_op = 308.15  # K
        self.T_ad = 308.15  # K

        # Stoichiometric parameters
        self._init_stoichiometric_params()

        # Biochemical parameters
        self._init_biochemical_params()

        # Physico-chemical parameters
        self._init_physicochemical_params()

        # Physical parameters (parameterized, but CONSTANT during simulation)
        self.V_liq = float(V_liq)  # m^3
        self.V_gas = float(V_gas)  # m^3
        self.V_ad = self.V_liq + self.V_gas
        self.m_liq = self.V_liq * 1000.0  # Liquid mass (kg), ρ≈1000 kg/m³

        # pH inhibition parameters
        self._init_pH_params()

        # Current state
        self.state = None
        self.q_ad = 178.4674  # m^3.d^-1

        # Gas flow variables
        self.q_gas = 0
        self.q_ch4 = 0
        self.q_co2 = 0
        self.p_gas = 0

        # ── Thermal model parameters ──────────────────────────────────────────
        # Dynamic liquid temperature and microbial adaptation temperature
        self.T_L = self.T_op    # Liquid phase temperature (K) – ODE state
        self.T_a = self.T_op    # Microbial adaptation temperature (K) – ODE state
        self.Q_HEX = 0.0        # Heat exchanger power (W), set by controller

        # Arrhenius parameters (hydrolysis reactions, Ea=64 kJ/mol, Biores. Technol. 2024)
        self.Ea_hyd = 64000.0   # Activation energy (J/mol)
        self.R_gas = 8.314      # Universal gas constant (J/mol/K)
        self.T_ref = 308.15     # Reference temperature (K) = 35°C

        # Cardinal Temperature Model (Angelidaki 1999, thermophilic optimum)
        self.T_opt = 328.15     # Optimal temperature (K) = 55°C
        self.T_max = 338.15     # Maximum temperature (K) = 65°C
        self.T_min = 281.15     # Minimum temperature (K) = 8°C
        # Precompute normalization: cardinal_factor(T_ref) should equal 1.0
        # At T_ref=308.15 < T_opt=328.15: raw = (T_ref-T_min)/(T_opt-T_min)
        _card_ref_raw = (self.T_ref - self.T_min) / (self.T_opt - self.T_min)
        self._card_norm = 1.0 / _card_ref_raw  # normalisation factor

        # Kovalovszki F_K inhibition parameters (methanogens only)
        self.s_hg = 5.0         # Temperature inhibition std deviation (K)
        self.tau_a = 30.0       # Adaptation time constant (d)

        # Energy balance parameters
        self.Cp = 4200.0        # Specific heat capacity of liquid (J/kg/K)
        self.UA = 50.0          # Overall heat loss coefficient (W/K)
        self.T_env = 298.15     # Ambient temperature (K) = 25°C
        self.T_feed = 298.15    # Feed temperature (K) = 25°C

        # Reaction enthalpies (kJ/mol CH4 produced, negative = exothermic)
        # 64 g COD ≡ 1 mol CH4 for both acetoclastic and hydrogenotrophic
        self.dH_ac_to_ch4 = -135.6   # Acetate → CH4 (kJ/mol CH4)
        self.dH_h2_to_ch4 = -130.7   # H2/CO2 → CH4 (kJ/mol CH4)

        # Monitoring outputs – updated after each step() call
        self.Q_reac_last = 0.0  # Metabolic heat generation (W)
        # ─────────────────────────────────────────────────────────────────────

    def _init_stoichiometric_params(self):
        """Initialize stoichiometric parameters"""
        self.f_sI_xc = 0.1
        self.f_xI_xc = 0.2
        self.f_ch_xc = 0.2
        self.f_pr_xc = 0.2
        self.f_li_xc = 0.3
        self.N_xc = 0.0376 / 14
        self.N_I = 0.06 / 14
        self.N_aa = 0.007
        self.C_xc = 0.02786
        self.C_sI = 0.03
        self.C_ch = 0.0313
        self.C_pr = 0.03
        self.C_li = 0.022
        self.C_xI = 0.03
        self.C_su = 0.0313
        self.C_aa = 0.03
        self.f_fa_li = 0.95
        self.C_fa = 0.0217
        self.f_h2_su = 0.19
        self.f_bu_su = 0.13
        self.f_pro_su = 0.27
        self.f_ac_su = 0.41
        self.N_bac = 0.08 / 14
        self.C_bu = 0.025
        self.C_pro = 0.0268
        self.C_ac = 0.0313
        self.C_bac = 0.0313
        self.Y_su = 0.1
        self.f_h2_aa = 0.06
        self.f_va_aa = 0.23
        self.f_bu_aa = 0.26
        self.f_pro_aa = 0.05
        self.f_ac_aa = 0.40
        self.C_va = 0.024
        self.Y_aa = 0.08
        self.Y_fa = 0.06
        self.Y_c4 = 0.06
        self.Y_pro = 0.04
        self.C_ch4 = 0.0156
        self.Y_ac = 0.05
        self.Y_h2 = 0.06

    def _init_biochemical_params(self):
        """Initialize biochemical parameters"""
        self.k_dis = 0.5
        self.k_hyd_ch = 10
        self.k_hyd_pr = 10
        self.k_hyd_li = 10
        self.K_S_IN = 10 ** -4
        self.k_m_su = 30
        self.K_S_su = 0.5
        self.pH_UL_aa = 5.5
        self.pH_LL_aa = 4
        self.k_m_aa = 50
        self.K_S_aa = 0.3
        self.k_m_fa = 6
        self.K_S_fa = 0.4
        self.K_I_h2_fa = 5 * 10 ** -6
        self.k_m_c4 = 20
        self.K_S_c4 = 0.2
        self.K_I_h2_c4 = 10 ** -5
        self.k_m_pro = 13
        self.K_S_pro = 0.1
        self.K_I_h2_pro = 3.5 * 10 ** -6
        self.k_m_ac = 8
        self.K_S_ac = 0.15
        self.K_I_nh3 = 0.0018
        self.pH_UL_ac = 7
        self.pH_LL_ac = 6
        self.k_m_h2 = 35
        self.K_S_h2 = 7 * 10 ** -6
        self.pH_UL_h2 = 6
        self.pH_LL_h2 = 5
        self.k_dec_X_su = 0.02
        self.k_dec_X_aa = 0.02
        self.k_dec_X_fa = 0.02
        self.k_dec_X_c4 = 0.02
        self.k_dec_X_pro = 0.02
        self.k_dec_X_ac = 0.02
        self.k_dec_X_h2 = 0.02

    def _init_physicochemical_params(self):
        """Initialize physico-chemical parameters (T-independent ones set here, T-dependent via helper)"""
        # Fixed acid dissociation constants (T-independent, from BSM2)
        self.K_a_va = 10 ** -4.86
        self.K_a_bu = 10 ** -4.82
        self.K_a_pro = 10 ** -4.88
        self.K_a_ac = 10 ** -4.76

        # Acid-base kinetic rate constants
        self.k_A_B_va = 10 ** 10
        self.k_A_B_bu = 10 ** 10
        self.k_A_B_pro = 10 ** 10
        self.k_A_B_ac = 10 ** 10
        self.k_A_B_co2 = 10 ** 10
        self.k_A_B_IN = 10 ** 10

        # Gas pressure control
        self.k_p = 5 * 10 ** 4

        # Gas-liquid mass transfer coefficient (k_L_a)
        self.k_L_a_base = 200.0
        self.k_L_a = self.k_L_a_base
        self.agitator_speed_ref = 100.0  # Reference agitator speed (RPM)
        self.agitator_speed = 100.0      # Current agitator speed (RPM)

        # Initialize temperature-dependent physico-chemical constants at T_ad
        self._update_physicochemical_params(self.T_ad)

    def _update_physicochemical_params(self, T: float):
        """
        Update all temperature-dependent physico-chemical constants.

        Called at init, after set_temperature(), and after each ODE substep
        (before DAESolve) to keep DAE consistent with the current T_L.

        Args:
            T: Temperature (K) to evaluate constants at
        """
        R100 = 100.0 * self.R  # = 8.3145 J/(mol·K), used for unit consistency
        self.K_w = 10 ** -14.0 * np.exp((55900 / R100) * (1 / self.T_base - 1 / T))
        self.K_a_co2 = 10 ** -6.35 * np.exp((7646 / R100) * (1 / self.T_base - 1 / T))
        self.K_a_IN = 10 ** -9.25 * np.exp((51965 / R100) * (1 / self.T_base - 1 / T))
        self.p_gas_h2o = 0.0313 * np.exp(5290 * (1 / self.T_base - 1 / T))
        self.K_H_co2 = 0.035 * np.exp((-19410 / R100) * (1 / self.T_base - 1 / T))
        self.K_H_ch4 = 0.0014 * np.exp((-14240 / R100) * (1 / self.T_base - 1 / T))
        self.K_H_h2 = 7.8e-4 * np.exp((-4180 / R100) * (1 / self.T_base - 1 / T))
        # Keep T_op in sync so DAESolve gas-pressure formula uses correct T
        self.T_op = T
        self.T_ad = T

    def _init_pH_params(self):
        """Initialize pH inhibition parameters"""
        self.K_pH_aa = 10 ** (-1 * (self.pH_LL_aa + self.pH_UL_aa) / 2.0)
        self.nn_aa = 3.0 / (self.pH_UL_aa - self.pH_LL_aa)
        self.K_pH_ac = 10 ** (-1 * (self.pH_LL_ac + self.pH_UL_ac) / 2.0)
        self.n_ac = 3.0 / (self.pH_UL_ac - self.pH_LL_ac)
        self.K_pH_h2 = 10 ** (-1 * (self.pH_LL_h2 + self.pH_UL_h2) / 2.0)
        self.n_h2 = 3.0 / (self.pH_UL_h2 - self.pH_LL_h2)

    # ──────────────────────────────────────────────────────────────────────────
    # Temperature correction helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _arrhenius_factor(self, T: float) -> float:
        """
        Arrhenius correction factor for hydrolysis rate constants.
        Normalized to 1.0 at T_ref (35°C, 308.15 K).
        Ea = 64 kJ/mol (Bioresource Technology 413, 2024).
        """
        return np.exp(-self.Ea_hyd / self.R_gas * (1.0 / T - 1.0 / self.T_ref))

    def _cardinal_factor(self, T: float) -> float:
        """
        Cardinal Temperature Model for μ_max (Angelidaki 1999).
        Piecewise linear, normalized to 1.0 at T_ref.
        T_opt=55°C, T_max=65°C, T_min=8°C.
        """
        if T <= self.T_min or T >= self.T_max:
            return 0.0
        elif T <= self.T_opt:
            raw = (T - self.T_min) / (self.T_opt - self.T_min)
        else:
            raw = (self.T_max - T) / (self.T_max - self.T_opt)
        return raw * self._card_norm

    def _fk_factor(self, T_L: float, T_a: float) -> float:
        """
        Kovalovszki F_K temperature inhibition factor for methanogens.
        F_K = exp(-(T_L - T_a)^2 / (2*s_hg^2)).
        Equals 1.0 at steady state (T_L ≈ T_a).
        """
        return np.exp(-((T_L - T_a) ** 2) / (2.0 * self.s_hg ** 2))

    def _compute_Q_reac(self) -> float:
        """
        Compute metabolic heat generation (W) from the current solver state.
        Uses the same formula as in ADM1_ODE but evaluated at the post-step state.
        Called once per step() for monitoring/visualization purposes.
        """
        if self.state is None:
            return 0.0
        state = self.state
        card = self._cardinal_factor(self.T_L)
        fk   = self._fk_factor(self.T_L, self.T_a)
        S_H_ion = max(state.get('S_H_ion', 1e-7), 1e-14)
        S_IN    = max(state.get('S_IN',    1e-6),  1e-10)
        S_nh3   = state.get('S_nh3', 0.0)
        S_ac    = state.get('S_ac',  0.0)
        X_ac    = state.get('X_ac',  0.0)
        S_h2    = state.get('S_h2',  0.0)
        X_h2    = state.get('X_h2',  0.0)

        I_pH_ac  = (self.K_pH_ac ** self.n_ac) / (S_H_ion ** self.n_ac + self.K_pH_ac ** self.n_ac)
        I_pH_h2  = (self.K_pH_h2 ** self.n_h2) / (S_H_ion ** self.n_h2 + self.K_pH_h2 ** self.n_h2)
        I_IN_lim = 1.0 / (1.0 + self.K_S_IN / S_IN)
        I_nh3    = 1.0 / (1.0 + S_nh3 / self.K_I_nh3)

        k_m_ac_T = self.k_m_ac * card * fk
        k_m_h2_T = self.k_m_h2 * card * fk

        Rho_11 = k_m_ac_T * (S_ac / (self.K_S_ac + S_ac + 1e-10)) * X_ac * I_pH_ac * I_IN_lim * I_nh3
        Rho_12 = k_m_h2_T * (S_h2 / (self.K_S_h2 + S_h2 + 1e-10)) * X_h2 * I_pH_h2 * I_IN_lim

        mol_ac = (1.0 - self.Y_ac) * Rho_11 * self.V_liq / 0.064  # kmol CH4/d
        mol_h2 = (1.0 - self.Y_h2) * Rho_12 * self.V_liq / 0.064

        Q_reac = (-self.dH_ac_to_ch4 * mol_ac - self.dH_h2_to_ch4 * mol_h2) * 1e3 / 86400.0
        return float(Q_reac)

    # ──────────────────────────────────────────────────────────────────────────

    def set_state(self, state_dict: Dict):
        """
        Set the current state of the digester

        Args:
            state_dict: Dictionary containing all 38 state variables
        """
        self.state = state_dict.copy()

    def get_state(self) -> Dict:
        """
        Get the current state of the digester

        Returns:
            Dictionary containing all 38 state variables
        """
        return self.state.copy() if self.state is not None else {}

    def set_influent(self, influent_dict: Dict):
        """
        Set influent conditions

        Args:
            influent_dict: Dictionary containing influent concentrations
        """
        self.influent = influent_dict.copy()

    def set_flow_rate(self, q_ad: float):
        """
        Set flow rate

        Args:
            q_ad: Flow rate in m^3/d
        """
        self.q_ad = np.clip(q_ad, 50, 300)

    def set_agitator_speed(self, agitator_speed: float):
        """
        Set agitator speed and update k_L_a accordingly

        The gas-liquid mass transfer coefficient (k_L_a) is affected by agitation.
        Higher agitation increases gas bubble formation and release.

        Relationship: k_L_a = k_L_a_base * (agitator_speed / agitator_speed_ref)^0.7

        The exponent 0.7 is typical for stirred tank reactors (literature range: 0.5-1.0)

        Args:
            agitator_speed: Agitator speed in RPM (0-200)

        Effects:
            - Higher speed -> Higher k_L_a -> Faster gas transfer -> More gas release
            - Lower speed -> Lower k_L_a -> Slower gas transfer -> Gas accumulation in liquid
        """
        # Clip to valid range
        self.agitator_speed = np.clip(agitator_speed, 10.0, 200.0)

        # Calculate k_L_a based on agitator speed
        # Using power law relationship: k_L_a ∝ N^0.7 (typical for stirred tanks)
        speed_ratio = self.agitator_speed / self.agitator_speed_ref
        self.k_L_a = self.k_L_a_base * (speed_ratio ** 0.7)

        # Ensure k_L_a stays within reasonable bounds
        self.k_L_a = np.clip(self.k_L_a, 50.0, 500.0)

    def get_k_L_a(self) -> float:
        """Get current gas-liquid mass transfer coefficient"""
        return self.k_L_a

    def set_temperature(self, temperature_celsius: float):
        """
        Set operating temperature (digital-twin use: overrides T_L directly).

        For the digital twin, temperature is treated as a direct setpoint.
        Q_HEX is automatically computed to maintain this temperature against
        heat losses: Q_HEX = UA × (T_L - T_env).

        For RL training, use the Q_HEX parameter in step() instead.

        Args:
            temperature_celsius: Temperature in Celsius (20–60°C)
        """
        temperature_celsius = np.clip(temperature_celsius, 20.0, 60.0)
        T_K = temperature_celsius + 273.15

        # Override liquid and adaptation temperature directly
        self.T_L = T_K
        self.T_a = T_K

        # Update all T-dependent physico-chemical constants (fixes original bug)
        self._update_physicochemical_params(T_K)

        # Set Q_HEX to steady-state value that compensates heat losses
        # (so energy balance stays near this setpoint during the step)
        self.Q_HEX = self.UA * (T_K - self.T_env)


    def get_temperature(self) -> float:
        """Get current liquid temperature in Celsius"""
        return self.T_L - 273.15

    def ADM1_ODE(self, t, state_vector):
        """
        ADM1 differential equations with dynamic temperature.

        Args:
            t: Time
            state_vector: 40-element state vector
                [0:38]  ADM1 biological/chemical states
                [38]    T_L – liquid phase temperature (K)
                [39]    T_a – microbial adaptation temperature (K)

        Returns:
            derivatives: 40-element time derivative array (units: per day)
        """
        # ── Extract ADM1 states (indices 0–37) ───────────────────────────────
        (S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_I,
         X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I,
         S_cation, S_anion, S_H_ion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion,
         S_hco3_ion, S_co2, S_nh3, S_nh4_ion, S_gas_h2, S_gas_ch4, S_gas_co2) = state_vector[:38]

        # ── Extract thermal states (indices 38–39) ────────────────────────────
        T_L = float(state_vector[38])   # liquid temperature (K)
        T_a = float(state_vector[39])   # adaptation temperature (K)

        # Guard against unphysical temperatures in the integrator
        T_L = max(273.15, min(373.15, T_L))
        T_a = max(273.15, min(373.15, T_a))

        # ── Temperature-dependent physico-chemical constants (local, no self mutation) ──
        R100 = 100.0 * self.R
        K_w_loc     = 10 ** -14.0 * np.exp((55900  / R100) * (1 / self.T_base - 1 / T_L))
        K_a_co2_loc = 10 ** -6.35 * np.exp((7646   / R100) * (1 / self.T_base - 1 / T_L))
        K_a_IN_loc  = 10 ** -9.25 * np.exp((51965  / R100) * (1 / self.T_base - 1 / T_L))
        p_gas_h2o_loc = 0.0313 * np.exp(5290 * (1 / self.T_base - 1 / T_L))
        K_H_co2_loc = 0.035  * np.exp((-19410 / R100) * (1 / self.T_base - 1 / T_L))
        K_H_ch4_loc = 0.0014 * np.exp((-14240 / R100) * (1 / self.T_base - 1 / T_L))
        K_H_h2_loc  = 7.8e-4 * np.exp((-4180  / R100) * (1 / self.T_base - 1 / T_L))

        # ── Temperature-corrected kinetic parameters ──────────────────────────
        arr  = self._arrhenius_factor(T_L)
        card = self._cardinal_factor(T_L)
        fk   = self._fk_factor(T_L, T_a)

        k_hyd_ch_T = self.k_hyd_ch * arr
        k_hyd_pr_T = self.k_hyd_pr * arr
        k_hyd_li_T = self.k_hyd_li * arr

        k_m_su_T  = self.k_m_su  * card
        k_m_aa_T  = self.k_m_aa  * card
        k_m_fa_T  = self.k_m_fa  * card
        k_m_c4_T  = self.k_m_c4  * card
        k_m_pro_T = self.k_m_pro * card
        k_m_ac_T  = self.k_m_ac  * card * fk   # F_K only on methanogens
        k_m_h2_T  = self.k_m_h2  * card * fk

        # Get influent values
        inf = self.influent

        # Calculate derived variables
        S_nh4_ion = S_IN - S_nh3
        S_co2 = S_IC - S_hco3_ion

        # Inhibition functions
        I_pH_aa = (self.K_pH_aa ** self.nn_aa) / (S_H_ion ** self.nn_aa + self.K_pH_aa ** self.nn_aa)
        I_pH_ac = (self.K_pH_ac ** self.n_ac) / (S_H_ion ** self.n_ac + self.K_pH_ac ** self.n_ac)
        I_pH_h2 = (self.K_pH_h2 ** self.n_h2) / (S_H_ion ** self.n_h2 + self.K_pH_h2 ** self.n_h2)
        I_IN_lim = 1 / (1 + (self.K_S_IN / S_IN))
        I_h2_fa = 1 / (1 + (S_h2 / self.K_I_h2_fa))
        I_h2_c4 = 1 / (1 + (S_h2 / self.K_I_h2_c4))
        I_h2_pro = 1 / (1 + (S_h2 / self.K_I_h2_pro))
        I_nh3 = 1 / (1 + (S_nh3 / self.K_I_nh3))

        I_5 = I_pH_aa * I_IN_lim
        I_6 = I_5
        I_7 = I_pH_aa * I_IN_lim * I_h2_fa
        I_8 = I_pH_aa * I_IN_lim * I_h2_c4
        I_9 = I_8
        I_10 = I_pH_aa * I_IN_lim * I_h2_pro
        I_11 = I_pH_ac * I_IN_lim * I_nh3
        I_12 = I_pH_h2 * I_IN_lim

        # Biochemical process rates (using T-corrected kinetics)
        Rho_1 = self.k_dis * X_xc
        Rho_2 = k_hyd_ch_T * X_ch
        Rho_3 = k_hyd_pr_T * X_pr
        Rho_4 = k_hyd_li_T * X_li
        Rho_5 = k_m_su_T  * S_su / (self.K_S_su + S_su) * X_su * I_5
        Rho_6 = k_m_aa_T  * (S_aa / (self.K_S_aa + S_aa)) * X_aa * I_6
        Rho_7 = k_m_fa_T  * (S_fa / (self.K_S_fa + S_fa)) * X_fa * I_7
        Rho_8 = k_m_c4_T  * (S_va / (self.K_S_c4 + S_va)) * X_c4 * (S_va / (S_bu + S_va + 1e-6)) * I_8
        Rho_9 = k_m_c4_T  * (S_bu / (self.K_S_c4 + S_bu)) * X_c4 * (S_bu / (S_bu + S_va + 1e-6)) * I_9
        Rho_10 = k_m_pro_T * (S_pro / (self.K_S_pro + S_pro)) * X_pro * I_10
        Rho_11 = k_m_ac_T  * (S_ac / (self.K_S_ac + S_ac)) * X_ac * I_11
        Rho_12 = k_m_h2_T  * (S_h2 / (self.K_S_h2 + S_h2)) * X_h2 * I_12
        Rho_13 = self.k_dec_X_su * X_su
        Rho_14 = self.k_dec_X_aa * X_aa
        Rho_15 = self.k_dec_X_fa * X_fa
        Rho_16 = self.k_dec_X_c4 * X_c4
        Rho_17 = self.k_dec_X_pro * X_pro
        Rho_18 = self.k_dec_X_ac * X_ac
        Rho_19 = self.k_dec_X_h2 * X_h2

        # Gas phase calculations (using T_L for ideal gas, local p_gas_h2o)
        p_gas_h2  = S_gas_h2  * self.R * T_L / 16
        p_gas_ch4 = S_gas_ch4 * self.R * T_L / 64
        p_gas_co2 = S_gas_co2 * self.R * T_L

        p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o_loc
        q_gas = self.k_p * (p_gas - self.p_atm)
        if q_gas < 0:
            q_gas = 0

        # Store for later access
        self.p_gas = p_gas
        self.q_gas = q_gas
        self.q_ch4 = q_gas * (p_gas_ch4 / p_gas) if p_gas > 0 else 0
        self.q_co2 = q_gas * (p_gas_co2 / p_gas) if p_gas > 0 else 0

        # Gas transfer rates (using local K_H values)
        Rho_T_8  = self.k_L_a * (S_h2  - 16  * K_H_h2_loc  * p_gas_h2)
        Rho_T_9  = self.k_L_a * (S_ch4 - 64  * K_H_ch4_loc * p_gas_ch4)
        Rho_T_10 = self.k_L_a * (S_co2 - K_H_co2_loc * p_gas_co2)

        # Differential equations for soluble matter
        diff_S_su = self.q_ad / self.V_liq * (inf['S_su'] - S_su) + Rho_2 + (1 - self.f_fa_li) * Rho_4 - Rho_5
        diff_S_aa = self.q_ad / self.V_liq * (inf['S_aa'] - S_aa) + Rho_3 - Rho_6
        diff_S_fa = self.q_ad / self.V_liq * (inf['S_fa'] - S_fa) + (self.f_fa_li * Rho_4) - Rho_7
        diff_S_va = self.q_ad / self.V_liq * (inf['S_va'] - S_va) + (1 - self.Y_aa) * self.f_va_aa * Rho_6 - Rho_8
        diff_S_bu = self.q_ad / self.V_liq * (inf['S_bu'] - S_bu) + (1 - self.Y_su) * self.f_bu_su * Rho_5 + (1 - self.Y_aa) * self.f_bu_aa * Rho_6 - Rho_9
        diff_S_pro = self.q_ad / self.V_liq * (inf['S_pro'] - S_pro) + (1 - self.Y_su) * self.f_pro_su * Rho_5 + (1 - self.Y_aa) * self.f_pro_aa * Rho_6 + (1 - self.Y_c4) * 0.54 * Rho_8 - Rho_10
        diff_S_ac = self.q_ad / self.V_liq * (inf['S_ac'] - S_ac) + (1 - self.Y_su) * self.f_ac_su * Rho_5 + (1 - self.Y_aa) * self.f_ac_aa * Rho_6 + (1 - self.Y_fa) * 0.7 * Rho_7 + (1 - self.Y_c4) * 0.31 * Rho_8 + (1 - self.Y_c4) * 0.8 * Rho_9 + (1 - self.Y_pro) * 0.57 * Rho_10 - Rho_11
        # S_h2 is solved algebraically in DAESolve, not as ODE (BSM2 standard)
        diff_S_h2 = 0
        diff_S_ch4 = self.q_ad / self.V_liq * (inf['S_ch4'] - S_ch4) + (1 - self.Y_ac) * Rho_11 + (1 - self.Y_h2) * Rho_12 - Rho_T_9

        # S_IC differential
        s_1 = -1 * self.C_xc + self.f_sI_xc * self.C_sI + self.f_ch_xc * self.C_ch + self.f_pr_xc * self.C_pr + self.f_li_xc * self.C_li + self.f_xI_xc * self.C_xI
        s_2 = -1 * self.C_ch + self.C_su
        s_3 = -1 * self.C_pr + self.C_aa
        s_4 = -1 * self.C_li + (1 - self.f_fa_li) * self.C_su + self.f_fa_li * self.C_fa
        s_5 = -1 * self.C_su + (1 - self.Y_su) * (self.f_bu_su * self.C_bu + self.f_pro_su * self.C_pro + self.f_ac_su * self.C_ac) + self.Y_su * self.C_bac
        s_6 = -1 * self.C_aa + (1 - self.Y_aa) * (self.f_va_aa * self.C_va + self.f_bu_aa * self.C_bu + self.f_pro_aa * self.C_pro + self.f_ac_aa * self.C_ac) + self.Y_aa * self.C_bac
        s_7 = -1 * self.C_fa + (1 - self.Y_fa) * 0.7 * self.C_ac + self.Y_fa * self.C_bac
        s_8 = -1 * self.C_va + (1 - self.Y_c4) * 0.54 * self.C_pro + (1 - self.Y_c4) * 0.31 * self.C_ac + self.Y_c4 * self.C_bac
        s_9 = -1 * self.C_bu + (1 - self.Y_c4) * 0.8 * self.C_ac + self.Y_c4 * self.C_bac
        s_10 = -1 * self.C_pro + (1 - self.Y_pro) * 0.57 * self.C_ac + self.Y_pro * self.C_bac
        s_11 = -1 * self.C_ac + (1 - self.Y_ac) * self.C_ch4 + self.Y_ac * self.C_bac
        s_12 = (1 - self.Y_h2) * self.C_ch4 + self.Y_h2 * self.C_bac
        s_13 = -1 * self.C_bac + self.C_xc

        Sigma = (s_1 * Rho_1 + s_2 * Rho_2 + s_3 * Rho_3 + s_4 * Rho_4 + s_5 * Rho_5 + s_6 * Rho_6 +
                 s_7 * Rho_7 + s_8 * Rho_8 + s_9 * Rho_9 + s_10 * Rho_10 + s_11 * Rho_11 + s_12 * Rho_12 +
                 s_13 * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19))

        diff_S_IC = self.q_ad / self.V_liq * (inf['S_IC'] - S_IC) - Sigma - Rho_T_10

        diff_S_IN = self.q_ad / self.V_liq * (inf['S_IN'] - S_IN) + (self.N_xc - self.f_xI_xc * self.N_I - self.f_sI_xc * self.N_I - self.f_pr_xc * self.N_aa) * Rho_1 - self.Y_su * self.N_bac * Rho_5 + (self.N_aa - self.Y_aa * self.N_bac) * Rho_6 - self.Y_fa * self.N_bac * Rho_7 - self.Y_c4 * self.N_bac * Rho_8 - self.Y_c4 * self.N_bac * Rho_9 - self.Y_pro * self.N_bac * Rho_10 - self.Y_ac * self.N_bac * Rho_11 - self.Y_h2 * self.N_bac * Rho_12 + (self.N_bac - self.N_xc) * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)

        diff_S_I = self.q_ad / self.V_liq * (inf['S_I'] - S_I) + self.f_sI_xc * Rho_1

        # Differential equations for particulate matter
        diff_X_xc = self.q_ad / self.V_liq * (inf['X_xc'] - X_xc) - Rho_1 + Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19
        diff_X_ch = self.q_ad / self.V_liq * (inf['X_ch'] - X_ch) + self.f_ch_xc * Rho_1 - Rho_2
        diff_X_pr = self.q_ad / self.V_liq * (inf['X_pr'] - X_pr) + self.f_pr_xc * Rho_1 - Rho_3
        diff_X_li = self.q_ad / self.V_liq * (inf['X_li'] - X_li) + self.f_li_xc * Rho_1 - Rho_4
        diff_X_su = self.q_ad / self.V_liq * (inf['X_su'] - X_su) + self.Y_su * Rho_5 - Rho_13
        diff_X_aa = self.q_ad / self.V_liq * (inf['X_aa'] - X_aa) + self.Y_aa * Rho_6 - Rho_14
        diff_X_fa = self.q_ad / self.V_liq * (inf['X_fa'] - X_fa) + self.Y_fa * Rho_7 - Rho_15
        diff_X_c4 = self.q_ad / self.V_liq * (inf['X_c4'] - X_c4) + self.Y_c4 * Rho_8 + self.Y_c4 * Rho_9 - Rho_16
        diff_X_pro = self.q_ad / self.V_liq * (inf['X_pro'] - X_pro) + self.Y_pro * Rho_10 - Rho_17
        diff_X_ac = self.q_ad / self.V_liq * (inf['X_ac'] - X_ac) + self.Y_ac * Rho_11 - Rho_18
        diff_X_h2 = self.q_ad / self.V_liq * (inf['X_h2'] - X_h2) + self.Y_h2 * Rho_12 - Rho_19
        diff_X_I = self.q_ad / self.V_liq * (inf['X_I'] - X_I) + self.f_xI_xc * Rho_1

        # Cations and anions
        diff_S_cation = self.q_ad / self.V_liq * (inf['S_cation'] - S_cation)
        diff_S_anion = self.q_ad / self.V_liq * (inf['S_anion'] - S_anion)

        # Ion states (set to 0 for ODE implementation, solved by DAE)
        diff_S_H_ion = 0
        diff_S_va_ion = 0
        diff_S_bu_ion = 0
        diff_S_pro_ion = 0
        diff_S_ac_ion = 0
        diff_S_hco3_ion = 0
        diff_S_co2 = 0
        diff_S_nh3 = 0
        diff_S_nh4_ion = 0

        # Gas phase equations
        diff_S_gas_h2 = (q_gas / self.V_gas * -1 * S_gas_h2) + (Rho_T_8 * self.V_liq / self.V_gas)
        diff_S_gas_ch4 = (q_gas / self.V_gas * -1 * S_gas_ch4) + (Rho_T_9 * self.V_liq / self.V_gas)
        diff_S_gas_co2 = (q_gas / self.V_gas * -1 * S_gas_co2) + (Rho_T_10 * self.V_liq / self.V_gas)

        # ── Thermal ODEs ──────────────────────────────────────────────────────
        # Reaction heat (Case 1: kinetics-based, Bioresource Technology 413, 2024)
        # 64 g COD = 1 mol CH4 for both acetoclastic (Rho_11) and hydrogenotrophic (Rho_12)
        # Rho in kg COD/(m³·d); /0.064 → kmol CH4/(m³·d); ×V_liq → kmol/d
        mol_ch4_ac_per_d = (1.0 - self.Y_ac) * Rho_11 * self.V_liq / 0.064  # kmol CH4/d
        mol_ch4_h2_per_d = (1.0 - self.Y_h2) * Rho_12 * self.V_liq / 0.064  # kmol CH4/d
        # Q_reac in W: kJ/mol × kmol/d × 1e3 J/kJ / 86400 s/d; -dH because dH<0 (exothermic)
        Q_reac = (-self.dH_ac_to_ch4 * mol_ch4_ac_per_d
                  - self.dH_h2_to_ch4 * mol_ch4_h2_per_d) * 1e3 / 86400.0

        # Heat losses to environment (W)
        Q_loss = self.UA * (T_L - self.T_env)

        # Advective heat from incoming feed (W)
        # q_ad [m³/d] × 1000 kg/m³ / 86400 s/d × Cp [J/kg/K] × ΔT [K]
        Q_adv = (self.q_ad * 1000.0 / 86400.0) * self.Cp * (self.T_feed - T_L)

        # Energy balance ODE for T_L (K/d)
        dT_L_dt = (self.Q_HEX + Q_reac - Q_loss + Q_adv) / (self.m_liq * self.Cp) * 86400.0

        # Microbial adaptation ODE for T_a (K/d)
        dT_a_dt = (T_L - T_a) / self.tau_a
        # ─────────────────────────────────────────────────────────────────────

        return np.array([
            diff_S_su, diff_S_aa, diff_S_fa, diff_S_va, diff_S_bu, diff_S_pro, diff_S_ac, diff_S_h2,
            diff_S_ch4, diff_S_IC, diff_S_IN, diff_S_I, diff_X_xc, diff_X_ch, diff_X_pr, diff_X_li,
            diff_X_su, diff_X_aa, diff_X_fa, diff_X_c4, diff_X_pro, diff_X_ac, diff_X_h2, diff_X_I,
            diff_S_cation, diff_S_anion, diff_S_H_ion, diff_S_va_ion, diff_S_bu_ion, diff_S_pro_ion,
            diff_S_ac_ion, diff_S_hco3_ion, diff_S_co2, diff_S_nh3, diff_S_nh4_ion,
            diff_S_gas_h2, diff_S_gas_ch4, diff_S_gas_co2,
            dT_L_dt,   # index 38
            dT_a_dt    # index 39
        ])

    def DAESolve(self, debug=False):
        """
        Solve DAE equations for pH and H2
        Uses Newton-Raphson method
        """
        # Get current state
        S_va = self.state['S_va']
        S_bu = self.state['S_bu']
        S_pro = self.state['S_pro']
        S_ac = self.state['S_ac']
        S_IC = self.state['S_IC']
        S_IN = self.state['S_IN']
        S_cation = self.state['S_cation']
        S_anion = self.state['S_anion']
        S_H_ion = self.state['S_H_ion']
        S_h2 = self.state['S_h2']

        if debug:
            print(f"DAESolve input state:")
            print(f"  S_cation: {S_cation:.6e}, S_anion: {S_anion:.6e}")
            print(f"  S_IC: {S_IC:.6f}, S_IN: {S_IN:.6f}")
            print(f"  S_ac: {S_ac:.6f}, S_pro: {S_pro:.6f}")
            print(f"  S_H_ion (initial): {S_H_ion:.6e}")

        # Solver parameters
        tol = 10 ** (-12)
        maxIter = 1000

        # Solve for S_H_ion (pH)
        shdelta = 1.0
        i = 0

        while ((shdelta > tol or shdelta < -tol) and (i <= maxIter)):
            S_va_ion = self.K_a_va * S_va / (self.K_a_va + S_H_ion)
            S_bu_ion = self.K_a_bu * S_bu / (self.K_a_bu + S_H_ion)
            S_pro_ion = self.K_a_pro * S_pro / (self.K_a_pro + S_H_ion)
            S_ac_ion = self.K_a_ac * S_ac / (self.K_a_ac + S_H_ion)
            S_hco3_ion = self.K_a_co2 * S_IC / (self.K_a_co2 + S_H_ion)
            S_nh3 = self.K_a_IN * S_IN / (self.K_a_IN + S_H_ion)

            shdelta = (S_cation + (S_IN - S_nh3) + S_H_ion - S_hco3_ion -
                      S_ac_ion / 64.0 - S_pro_ion / 112.0 - S_bu_ion / 160.0 -
                      S_va_ion / 208.0 - self.K_w / S_H_ion - S_anion)

            shgradeq = (1 + self.K_a_IN * S_IN / ((self.K_a_IN + S_H_ion) * (self.K_a_IN + S_H_ion)) +
                       self.K_a_co2 * S_IC / ((self.K_a_co2 + S_H_ion) * (self.K_a_co2 + S_H_ion)) +
                       1 / 64.0 * self.K_a_ac * S_ac / ((self.K_a_ac + S_H_ion) * (self.K_a_ac + S_H_ion)) +
                       1 / 112.0 * self.K_a_pro * S_pro / ((self.K_a_pro + S_H_ion) * (self.K_a_pro + S_H_ion)) +
                       1 / 160.0 * self.K_a_bu * S_bu / ((self.K_a_bu + S_H_ion) * (self.K_a_bu + S_H_ion)) +
                       1 / 208.0 * self.K_a_va * S_va / ((self.K_a_va + S_H_ion) * (self.K_a_va + S_H_ion)) +
                       self.K_w / (S_H_ion * S_H_ion))

            S_H_ion = S_H_ion - shdelta / shgradeq
            if S_H_ion <= 0:
                S_H_ion = tol
            i += 1

        # Calculate pH
        pH = -np.log10(S_H_ion)

        if debug:
            print(f"  S_H_ion (final): {S_H_ion:.6e}, pH: {pH:.4f}, iterations: {i}")

        # Solve for S_h2 using Newton-Raphson (BSM2 standard)
        # Get additional state needed for S_h2 calculation
        S_su = self.state['S_su']
        S_aa = self.state['S_aa']
        S_fa = self.state['S_fa']
        S_pro = self.state['S_pro']
        X_su = self.state['X_su']
        X_aa = self.state['X_aa']
        X_fa = self.state['X_fa']
        X_c4 = self.state['X_c4']
        X_pro = self.state['X_pro']
        X_h2 = self.state['X_h2']
        S_gas_h2 = self.state['S_gas_h2']

        # Inhibition factors (use final S_H_ion from pH solve)
        I_pH_aa = (self.K_pH_aa ** self.nn_aa) / (S_H_ion ** self.nn_aa + self.K_pH_aa ** self.nn_aa)
        I_pH_h2 = (self.K_pH_h2 ** self.n_h2) / (S_H_ion ** self.n_h2 + self.K_pH_h2 ** self.n_h2)
        I_IN_lim = 1 / (1 + (self.K_S_IN / S_IN))

        # Newton-Raphson iteration for S_h2
        S_h2delta = 1.0
        j = 0
        eps = 1e-6

        while ((S_h2delta > tol or S_h2delta < -tol) and (j <= maxIter)):
            # Recalculate H2 inhibition factors with current S_h2
            I_h2_fa = 1 / (1 + (S_h2 / self.K_I_h2_fa))
            I_h2_c4 = 1 / (1 + (S_h2 / self.K_I_h2_c4))
            I_h2_pro = 1 / (1 + (S_h2 / self.K_I_h2_pro))

            I_5 = I_pH_aa * I_IN_lim
            I_6 = I_5
            I_7 = I_pH_aa * I_IN_lim * I_h2_fa
            I_8 = I_pH_aa * I_IN_lim * I_h2_c4
            I_9 = I_8
            I_10 = I_pH_aa * I_IN_lim * I_h2_pro
            I_12 = I_pH_h2 * I_IN_lim

            # Process rates (only those affecting S_h2)
            Rho_5 = self.k_m_su * (S_su / (self.K_S_su + S_su)) * X_su * I_5
            Rho_6 = self.k_m_aa * (S_aa / (self.K_S_aa + S_aa)) * X_aa * I_6
            Rho_7 = self.k_m_fa * (S_fa / (self.K_S_fa + S_fa)) * X_fa * I_7
            Rho_8 = self.k_m_c4 * (S_va / (self.K_S_c4 + S_va)) * X_c4 * (S_va / (S_bu + S_va + eps)) * I_8
            Rho_9 = self.k_m_c4 * (S_bu / (self.K_S_c4 + S_bu)) * X_c4 * (S_bu / (S_bu + S_va + eps)) * I_9
            Rho_10 = self.k_m_pro * (S_pro / (self.K_S_pro + S_pro)) * X_pro * I_10
            Rho_12 = self.k_m_h2 * (S_h2 / (self.K_S_h2 + S_h2)) * X_h2 * I_12

            # Gas transfer (use self.T_L updated after each substep)
            p_gas_h2 = S_gas_h2 * self.R * self.T_L / 16
            Rho_T_8 = self.k_L_a * (S_h2 - 16 * self.K_H_h2 * p_gas_h2)

            # S_h2 balance equation (BSM2 standard)
            S_h2delta = (self.q_ad / self.V_liq * (self.influent['S_h2'] - S_h2) +
                        (1 - self.Y_su) * self.f_h2_su * Rho_5 +
                        (1 - self.Y_aa) * self.f_h2_aa * Rho_6 +
                        (1 - self.Y_fa) * 0.3 * Rho_7 +
                        (1 - self.Y_c4) * 0.15 * Rho_8 +
                        (1 - self.Y_c4) * 0.2 * Rho_9 +
                        (1 - self.Y_pro) * 0.43 * Rho_10 -
                        Rho_12 - Rho_T_8)

            # Gradient for Newton-Raphson
            S_h2gradeq = (- 1.0 / self.V_liq * self.q_ad -
                         3.0 / 10.0 * (1 - self.Y_fa) * self.k_m_fa * S_fa / (self.K_S_fa + S_fa) * X_fa * I_pH_aa / (1 + self.K_S_IN / S_IN) / ((1 + S_h2 / self.K_I_h2_fa) * (1 + S_h2 / self.K_I_h2_fa)) / self.K_I_h2_fa -
                         3.0 / 20.0 * (1 - self.Y_c4) * self.k_m_c4 * S_va * S_va / (self.K_S_c4 + S_va) * X_c4 / (S_bu + S_va + eps) * I_pH_aa / (1 + self.K_S_IN / S_IN) / ((1 + S_h2 / self.K_I_h2_c4) * (1 + S_h2 / self.K_I_h2_c4)) / self.K_I_h2_c4 -
                         1.0 / 5.0 * (1 - self.Y_c4) * self.k_m_c4 * S_bu * S_bu / (self.K_S_c4 + S_bu) * X_c4 / (S_bu + S_va + eps) * I_pH_aa / (1 + self.K_S_IN / S_IN) / ((1 + S_h2 / self.K_I_h2_c4) * (1 + S_h2 / self.K_I_h2_c4)) / self.K_I_h2_c4 -
                         43.0 / 100.0 * (1 - self.Y_pro) * self.k_m_pro * S_pro / (self.K_S_pro + S_pro) * X_pro * I_pH_aa / (1 + self.K_S_IN / S_IN) / ((1 + S_h2 / self.K_I_h2_pro) * (1 + S_h2 / self.K_I_h2_pro)) / self.K_I_h2_pro -
                         self.k_m_h2 / (self.K_S_h2 + S_h2) * X_h2 * I_pH_h2 / (1 + self.K_S_IN / S_IN) +
                         self.k_m_h2 * S_h2 / ((self.K_S_h2 + S_h2) * (self.K_S_h2 + S_h2)) * X_h2 * I_pH_h2 / (1 + self.K_S_IN / S_IN) -
                         self.k_L_a)

            # Newton-Raphson update
            S_h2 = S_h2 - S_h2delta / S_h2gradeq
            if S_h2 <= 0:
                S_h2 = tol
            j += 1

        if debug:
            print(f"  S_h2 (final): {S_h2:.6e}, iterations: {j}")

        # Update state with DAE solution
        self.state['S_H_ion'] = S_H_ion
        self.state['S_va_ion'] = S_va_ion
        self.state['S_bu_ion'] = S_bu_ion
        self.state['S_pro_ion'] = S_pro_ion
        self.state['S_ac_ion'] = S_ac_ion
        self.state['S_hco3_ion'] = S_hco3_ion
        self.state['S_nh3'] = S_nh3
        self.state['S_nh4_ion'] = S_IN - S_nh3
        self.state['S_co2'] = S_IC - S_hco3_ion
        self.state['pH'] = pH
        self.state['S_h2'] = S_h2  # Update S_h2 from DAE solve

        return pH

    def step(self, dt: float = 1.0, Q_HEX: float = None) -> Tuple[Dict, float]:
        """
        Execute one simulation step with optional heat exchanger power.

        Args:
            dt:    Time step in days (internally subdivided for stability)
            Q_HEX: Heat exchanger power (W).
                   Positive → heating, Negative → cooling.
                   None → keep previous self.Q_HEX value (e.g. set by set_temperature()).

        Returns:
            new_state: Updated state dictionary (38 ADM1 keys + 'T_L', 'T_a')
            q_ch4:     Methane flow rate (m³/d)
        """
        if Q_HEX is not None:
            self.Q_HEX = float(Q_HEX)

        # ADM1 state variable names (38)
        state_names = ['S_su', 'S_aa', 'S_fa', 'S_va', 'S_bu', 'S_pro', 'S_ac', 'S_h2',
                       'S_ch4', 'S_IC', 'S_IN', 'S_I', 'X_xc', 'X_ch', 'X_pr', 'X_li',
                       'X_su', 'X_aa', 'X_fa', 'X_c4', 'X_pro', 'X_ac', 'X_h2', 'X_I',
                       'S_cation', 'S_anion', 'S_H_ion', 'S_va_ion', 'S_bu_ion', 'S_pro_ion',
                       'S_ac_ion', 'S_hco3_ion', 'S_co2', 'S_nh3', 'S_nh4_ion', 'S_gas_h2',
                       'S_gas_ch4', 'S_gas_co2']

        # Internal time step matching BSM2 (~15 minutes = 0.01041667 days)
        internal_dt = 0.01041667
        num_substeps = max(1, int(np.ceil(dt / internal_dt)))
        actual_internal_dt = dt / num_substeps

        for substep_idx in range(num_substeps):
            # Build 40-dim state vector: [38 ADM1 states | T_L | T_a]
            state_vector = np.array([
                self.state['S_su'], self.state['S_aa'], self.state['S_fa'], self.state['S_va'],
                self.state['S_bu'], self.state['S_pro'], self.state['S_ac'], self.state['S_h2'],
                self.state['S_ch4'], self.state['S_IC'], self.state['S_IN'], self.state['S_I'],
                self.state['X_xc'], self.state['X_ch'], self.state['X_pr'], self.state['X_li'],
                self.state['X_su'], self.state['X_aa'], self.state['X_fa'], self.state['X_c4'],
                self.state['X_pro'], self.state['X_ac'], self.state['X_h2'], self.state['X_I'],
                self.state['S_cation'], self.state['S_anion'], self.state['S_H_ion'],
                self.state['S_va_ion'], self.state['S_bu_ion'], self.state['S_pro_ion'],
                self.state['S_ac_ion'], self.state['S_hco3_ion'], self.state['S_co2'],
                self.state['S_nh3'], self.state['S_nh4_ion'], self.state['S_gas_h2'],
                self.state['S_gas_ch4'], self.state['S_gas_co2'],
                self.T_L,   # index 38
                self.T_a    # index 39
            ])

            # Integrate ODE using DOP853 (BSM2 standard)
            sol = scipy.integrate.solve_ivp(
                self.ADM1_ODE,
                [0, actual_internal_dt],
                state_vector,
                method='DOP853',
                rtol=1e-6,
                atol=1e-8
            )

            # Extract final state (40-dim)
            final_state = sol.y[:, -1]

            # Update ADM1 state dictionary (first 38 components)
            for i, name in enumerate(state_names):
                self.state[name] = final_state[i]

            # Update thermal state variables
            self.T_L = float(final_state[38])
            self.T_a = float(final_state[39])

            # Update temperature-dependent physico-chemical constants for DAESolve
            self._update_physicochemical_params(self.T_L)

            # Solve DAE for pH and ions
            pH = self.DAESolve()

            # Recalculate derived variables after DAE solve
            self.state['S_nh4_ion'] = self.state['S_IN'] - self.state['S_nh3']
            self.state['S_co2'] = self.state['S_IC'] - self.state['S_hco3_ion']

        # Expose thermal state in the state dict for observation
        self.state['T_L'] = self.T_L
        self.state['T_a'] = self.T_a

        # Update monitoring output (used by digital twin visualization)
        self.Q_reac_last = self._compute_Q_reac()

        return self.state.copy(), self.q_ch4


# Test the solver
if __name__ == "__main__":
    print("Testing ADM1 Solver...")

    solver = ADM1Solver()

    # Initialize with example state
    initial_state = {
        'S_su': 0.012394, 'S_aa': 0.0055432, 'S_fa': 0.10741, 'S_va': 0.012333,
        'S_bu': 0.014003, 'S_pro': 0.017584, 'S_ac': 0.089315, 'S_h2': 2.51e-07,
        'S_ch4': 0.05549, 'S_IC': 0.095149, 'S_IN': 0.094468, 'S_I': 0.13087,
        'X_xc': 0.10792, 'X_ch': 0.020517, 'X_pr': 0.08422, 'X_li': 0.043629,
        'X_su': 0.31222, 'X_aa': 0.93167, 'X_fa': 0.33839, 'X_c4': 0.33577,
        'X_pro': 0.10112, 'X_ac': 0.67724, 'X_h2': 0.28484, 'X_I': 17.2162,
        'S_cation': 1.08e-47, 'S_anion': 0.0052101, 'S_H_ion': 5.46e-08,
        'S_va_ion': 0.012284, 'S_bu_ion': 0.013953, 'S_pro_ion': 0.017511,
        'S_ac_ion': 0.089035, 'S_hco3_ion': 0.08568, 'S_co2': 0.0094689,
        'S_nh3': 0.001884, 'S_nh4_ion': 0.092584, 'S_gas_h2': 1.10e-05,
        'S_gas_ch4': 1.6535, 'S_gas_co2': 0.01354, 'pH': 7.26
    }

    # Example influent
    influent = {
        'S_su': 0, 'S_aa': 0.043879861735077, 'S_fa': 0, 'S_va': 0,
        'S_bu': 0, 'S_pro': 0, 'S_ac': 0, 'S_h2': 0, 'S_ch4': 0,
        'S_IC': 0.007932576498324, 'S_IN': 0.001972075171719, 'S_I': 0.028066531383966,
        'X_xc': 0, 'X_ch': 3.72360557732604, 'X_pr': 15.9235782340124,
        'X_li': 8.04700469138446, 'X_su': 0, 'X_aa': 0, 'X_fa': 0, 'X_c4': 0,
        'X_pro': 0, 'X_ac': 0, 'X_h2': 0, 'X_I': 17.010652669241,
        'S_cation': 0, 'S_anion': 0.005210083664145
    }

    solver.set_state(initial_state)
    solver.set_influent(influent)
    solver.set_flow_rate(178.4674)

    # Run one step
    print("\nRunning single time step (1 day)...")
    new_state, q_ch4 = solver.step(dt=1.0)

    print(f"\nResults:")
    print(f"  pH: {new_state['pH']:.4f}")
    print(f"  CH4 flow: {q_ch4:.4f} m³/d")
    print(f"  S_ac: {new_state['S_ac']:.6f} kg COD/m³")
    print(f"  X_ac: {new_state['X_ac']:.6f} kg COD/m³")
    print(f"  S_gas_ch4: {new_state['S_gas_ch4']:.6f} kg COD/m³")

    print("\nADM1 Solver test passed.")
