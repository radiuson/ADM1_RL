#!/usr/bin/env python3
"""
ADM1 Solver — Standard (No Thermal Extension)
==============================================

Standard ADM1 ODE implementation following Batstone et al. (2002) and
the BSM2 parameterisation by Rosen et al. (2006).  No dynamic temperature
states: the liquid temperature is fixed at T_op = 308.15 K (35 °C) and
all physico-chemical constants are pre-computed at that temperature.

This is the non-thermal baseline used in the cross-model comparison
experiments (Section XX of the paper).

State vector (38-dimensional):
    indices 0–37: standard ADM1 biological/chemical states (BSM2 ordering)
    (compare: ADM1Solver adds T_L at [38] and T_a at [39])

Key differences from ADM1Solver (adm1_solver.py):
    - No T_L / T_a ODE states
    - No Arrhenius hydrolysis correction
    - No Cardinal Temperature Model on mu_max
    - No Kovalovszki F_K methanogen inhibition
    - No Q_HEX heat-exchanger control input
    - All T-dependent constants computed once at 35 °C
"""

import numpy as np
import scipy.integrate
from typing import Dict, Tuple


class ADM1SolverStd:
    """
    Standard (non-thermal) ADM1 ODE solver.

    Integrates the 38-dimensional ODE using DOP853, then resolves the
    DAE (pH, ions, H2) via Newton-Raphson — identical procedure to
    ADM1Solver but without the two thermal ODE states.
    """

    def __init__(self, V_liq: float = 3400.0, V_gas: float = 300.0):
        # Physical constants
        self.R       = 0.083145   # bar·M⁻¹·K⁻¹
        self.T_base  = 298.15     # K
        self.p_atm   = 1.013      # bar
        self.T_op    = 308.15     # K  (35 °C, fixed)
        self.T_ad    = 308.15     # K

        # Reactor volumes
        self.V_liq = float(V_liq)
        self.V_gas = float(V_gas)

        # Stoichiometric parameters (identical to ADM1Solver)
        self._init_stoichiometric_params()
        self._init_biochemical_params()
        self._init_physicochemical_params()
        self._init_pH_params()

        # State / influent
        self.state    = None
        self.influent = None
        self.q_ad     = 178.4674  # m³/d default

        # Gas flow outputs (updated each step)
        self.q_gas = 0.0
        self.q_ch4 = 0.0
        self.q_co2 = 0.0
        self.p_gas = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Parameter initialisation (identical constants to ADM1Solver)
    # ──────────────────────────────────────────────────────────────────────────

    def _init_stoichiometric_params(self):
        self.f_sI_xc = 0.1;  self.f_xI_xc = 0.2;  self.f_ch_xc = 0.2
        self.f_pr_xc = 0.2;  self.f_li_xc = 0.3
        self.N_xc    = 0.0376 / 14;  self.N_I = 0.06 / 14;  self.N_aa = 0.007
        self.C_xc  = 0.02786;  self.C_sI  = 0.03;   self.C_ch  = 0.0313
        self.C_pr  = 0.03;     self.C_li  = 0.022;  self.C_xI  = 0.03
        self.C_su  = 0.0313;   self.C_aa  = 0.03;   self.f_fa_li = 0.95
        self.C_fa  = 0.0217;   self.f_h2_su = 0.19; self.f_bu_su = 0.13
        self.f_pro_su = 0.27;  self.f_ac_su = 0.41
        self.N_bac = 0.08 / 14
        self.C_bu  = 0.025;  self.C_pro = 0.0268;  self.C_ac  = 0.0313
        self.C_bac = 0.0313; self.Y_su  = 0.1;     self.f_h2_aa = 0.06
        self.f_va_aa = 0.23; self.f_bu_aa = 0.26;  self.f_pro_aa = 0.05
        self.f_ac_aa = 0.40; self.C_va  = 0.024;   self.Y_aa  = 0.08
        self.Y_fa  = 0.06;   self.Y_c4  = 0.06;    self.Y_pro = 0.04
        self.C_ch4 = 0.0156; self.Y_ac  = 0.05;    self.Y_h2  = 0.06

    def _init_biochemical_params(self):
        self.k_dis    = 0.5;   self.k_hyd_ch = 10;   self.k_hyd_pr = 10
        self.k_hyd_li = 10;    self.K_S_IN   = 1e-4
        self.k_m_su   = 30;    self.K_S_su   = 0.5
        self.pH_UL_aa = 5.5;   self.pH_LL_aa = 4
        self.k_m_aa   = 50;    self.K_S_aa   = 0.3
        self.k_m_fa   = 6;     self.K_S_fa   = 0.4;   self.K_I_h2_fa  = 5e-6
        self.k_m_c4   = 20;    self.K_S_c4   = 0.2;   self.K_I_h2_c4  = 1e-5
        self.k_m_pro  = 13;    self.K_S_pro  = 0.1;   self.K_I_h2_pro = 3.5e-6
        self.k_m_ac   = 8;     self.K_S_ac   = 0.15;  self.K_I_nh3    = 0.0018
        self.pH_UL_ac = 7;     self.pH_LL_ac = 6
        self.k_m_h2   = 35;    self.K_S_h2   = 7e-6
        self.pH_UL_h2 = 6;     self.pH_LL_h2 = 5
        self.k_dec_X_su  = 0.02;  self.k_dec_X_aa  = 0.02
        self.k_dec_X_fa  = 0.02;  self.k_dec_X_c4  = 0.02
        self.k_dec_X_pro = 0.02;  self.k_dec_X_ac  = 0.02
        self.k_dec_X_h2  = 0.02

    def _init_physicochemical_params(self):
        """Compute all T-dependent constants once at T_op = 308.15 K."""
        T   = self.T_op
        R100 = 100.0 * self.R

        # Fixed acid dissociation constants (T-independent, BSM2)
        self.K_a_va  = 10 ** -4.86
        self.K_a_bu  = 10 ** -4.82
        self.K_a_pro = 10 ** -4.88
        self.K_a_ac  = 10 ** -4.76

        # T-dependent constants (evaluated once at 35 °C)
        self.K_w      = 10**-14.0 * np.exp((55900  / R100) * (1/self.T_base - 1/T))
        self.K_a_co2  = 10**-6.35 * np.exp((7646   / R100) * (1/self.T_base - 1/T))
        self.K_a_IN   = 10**-9.25 * np.exp((51965  / R100) * (1/self.T_base - 1/T))
        self.p_gas_h2o = 0.0313  * np.exp(5290     *          (1/self.T_base - 1/T))
        self.K_H_co2  = 0.035   * np.exp((-19410  / R100) * (1/self.T_base - 1/T))
        self.K_H_ch4  = 0.0014  * np.exp((-14240  / R100) * (1/self.T_base - 1/T))
        self.K_H_h2   = 7.8e-4  * np.exp((-4180   / R100) * (1/self.T_base - 1/T))

        # Acid-base kinetic constants
        self.k_A_B_va  = 1e10;  self.k_A_B_bu  = 1e10
        self.k_A_B_pro = 1e10;  self.k_A_B_ac  = 1e10
        self.k_A_B_co2 = 1e10;  self.k_A_B_IN  = 1e10

        # Gas transfer and pressure
        self.k_p   = 5e4    # m³·d⁻¹·bar⁻¹
        self.k_L_a = 200.0  # d⁻¹

    def _init_pH_params(self):
        self.K_pH_aa = 10 ** (-1 * (self.pH_LL_aa + self.pH_UL_aa) / 2.0)
        self.nn_aa   = 3.0 / (self.pH_UL_aa - self.pH_LL_aa)
        self.K_pH_ac = 10 ** (-1 * (self.pH_LL_ac + self.pH_UL_ac) / 2.0)
        self.n_ac    = 3.0 / (self.pH_UL_ac - self.pH_LL_ac)
        self.K_pH_h2 = 10 ** (-1 * (self.pH_LL_h2 + self.pH_UL_h2) / 2.0)
        self.n_h2    = 3.0 / (self.pH_UL_h2 - self.pH_LL_h2)

    # ──────────────────────────────────────────────────────────────────────────
    # State / influent interface
    # ──────────────────────────────────────────────────────────────────────────

    def set_state(self, state_dict: Dict):
        self.state = state_dict.copy()

    def get_state(self) -> Dict:
        return self.state.copy() if self.state is not None else {}

    def set_influent(self, influent_dict: Dict):
        self.influent = influent_dict.copy()

    def set_flow_rate(self, q_ad: float):
        self.q_ad = float(np.clip(q_ad, 50.0, 300.0))

    # ──────────────────────────────────────────────────────────────────────────
    # ODE right-hand side  (38-dim, no thermal states)
    # ──────────────────────────────────────────────────────────────────────────

    def ADM1_ODE(self, t, sv):
        """Standard ADM1 differential equations (BSM2, 38 states)."""
        (S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4,
         S_IC, S_IN, S_I,
         X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I,
         S_cation, S_anion,
         S_H_ion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion,
         S_hco3_ion, S_co2, S_nh3, S_nh4_ion,
         S_gas_h2, S_gas_ch4, S_gas_co2) = sv

        inf = self.influent

        # Derived ion quantities
        S_nh4_ion = S_IN - S_nh3
        S_co2     = S_IC - S_hco3_ion

        # ── Inhibition functions ──────────────────────────────────────────────
        I_pH_aa  = (self.K_pH_aa ** self.nn_aa) / (S_H_ion ** self.nn_aa + self.K_pH_aa ** self.nn_aa)
        I_pH_ac  = (self.K_pH_ac ** self.n_ac)  / (S_H_ion ** self.n_ac  + self.K_pH_ac ** self.n_ac)
        I_pH_h2  = (self.K_pH_h2 ** self.n_h2)  / (S_H_ion ** self.n_h2  + self.K_pH_h2 ** self.n_h2)
        I_IN_lim = 1.0 / (1.0 + self.K_S_IN / S_IN)
        I_h2_fa  = 1.0 / (1.0 + S_h2 / self.K_I_h2_fa)
        I_h2_c4  = 1.0 / (1.0 + S_h2 / self.K_I_h2_c4)
        I_h2_pro = 1.0 / (1.0 + S_h2 / self.K_I_h2_pro)
        I_nh3    = 1.0 / (1.0 + S_nh3 / self.K_I_nh3)

        I_5  = I_pH_aa * I_IN_lim
        I_6  = I_5
        I_7  = I_pH_aa * I_IN_lim * I_h2_fa
        I_8  = I_pH_aa * I_IN_lim * I_h2_c4
        I_9  = I_8
        I_10 = I_pH_aa * I_IN_lim * I_h2_pro
        I_11 = I_pH_ac * I_IN_lim * I_nh3
        I_12 = I_pH_h2 * I_IN_lim

        # ── Biochemical process rates ─────────────────────────────────────────
        Rho_1  = self.k_dis    * X_xc
        Rho_2  = self.k_hyd_ch * X_ch
        Rho_3  = self.k_hyd_pr * X_pr
        Rho_4  = self.k_hyd_li * X_li
        Rho_5  = self.k_m_su  * S_su  / (self.K_S_su  + S_su)  * X_su  * I_5
        Rho_6  = self.k_m_aa  * S_aa  / (self.K_S_aa  + S_aa)  * X_aa  * I_6
        Rho_7  = self.k_m_fa  * S_fa  / (self.K_S_fa  + S_fa)  * X_fa  * I_7
        Rho_8  = self.k_m_c4  * S_va  / (self.K_S_c4  + S_va)  * X_c4  * (S_va / (S_bu + S_va + 1e-6)) * I_8
        Rho_9  = self.k_m_c4  * S_bu  / (self.K_S_c4  + S_bu)  * X_c4  * (S_bu / (S_bu + S_va + 1e-6)) * I_9
        Rho_10 = self.k_m_pro * S_pro / (self.K_S_pro + S_pro) * X_pro * I_10
        Rho_11 = self.k_m_ac  * S_ac  / (self.K_S_ac  + S_ac)  * X_ac  * I_11
        Rho_12 = self.k_m_h2  * S_h2  / (self.K_S_h2  + S_h2)  * X_h2  * I_12
        Rho_13 = self.k_dec_X_su  * X_su
        Rho_14 = self.k_dec_X_aa  * X_aa
        Rho_15 = self.k_dec_X_fa  * X_fa
        Rho_16 = self.k_dec_X_c4  * X_c4
        Rho_17 = self.k_dec_X_pro * X_pro
        Rho_18 = self.k_dec_X_ac  * X_ac
        Rho_19 = self.k_dec_X_h2  * X_h2

        # ── Gas phase ─────────────────────────────────────────────────────────
        T = self.T_op
        p_gas_h2  = S_gas_h2  * self.R * T / 16
        p_gas_ch4 = S_gas_ch4 * self.R * T / 64
        p_gas_co2 = S_gas_co2 * self.R * T

        p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + self.p_gas_h2o
        q_gas = self.k_p * (p_gas - self.p_atm)
        if q_gas < 0:
            q_gas = 0.0

        self.p_gas = p_gas
        self.q_gas = q_gas
        self.q_ch4 = q_gas * (p_gas_ch4 / p_gas) if p_gas > 0 else 0.0
        self.q_co2 = q_gas * (p_gas_co2 / p_gas) if p_gas > 0 else 0.0

        Rho_T_8  = self.k_L_a * (S_h2  - 16  * self.K_H_h2  * p_gas_h2)
        Rho_T_9  = self.k_L_a * (S_ch4 - 64  * self.K_H_ch4 * p_gas_ch4)
        Rho_T_10 = self.k_L_a * (S_co2 - self.K_H_co2 * p_gas_co2)

        # ── Differential equations ────────────────────────────────────────────
        q = self.q_ad / self.V_liq

        diff_S_su  = q*(inf['S_su'] -S_su)  + Rho_2 + (1-self.f_fa_li)*Rho_4 - Rho_5
        diff_S_aa  = q*(inf['S_aa'] -S_aa)  + Rho_3 - Rho_6
        diff_S_fa  = q*(inf['S_fa'] -S_fa)  + self.f_fa_li*Rho_4 - Rho_7
        diff_S_va  = q*(inf['S_va'] -S_va)  + (1-self.Y_aa)*self.f_va_aa*Rho_6 - Rho_8
        diff_S_bu  = q*(inf['S_bu'] -S_bu)  + (1-self.Y_su)*self.f_bu_su*Rho_5 + (1-self.Y_aa)*self.f_bu_aa*Rho_6 - Rho_9
        diff_S_pro = q*(inf['S_pro']-S_pro) + (1-self.Y_su)*self.f_pro_su*Rho_5 + (1-self.Y_aa)*self.f_pro_aa*Rho_6 + (1-self.Y_c4)*0.54*Rho_8 - Rho_10
        diff_S_ac  = q*(inf['S_ac'] -S_ac)  + (1-self.Y_su)*self.f_ac_su*Rho_5 + (1-self.Y_aa)*self.f_ac_aa*Rho_6 + (1-self.Y_fa)*0.7*Rho_7 + (1-self.Y_c4)*0.31*Rho_8 + (1-self.Y_c4)*0.8*Rho_9 + (1-self.Y_pro)*0.57*Rho_10 - Rho_11
        diff_S_h2  = 0.0  # solved by DAE
        diff_S_ch4 = q*(inf['S_ch4']-S_ch4) + (1-self.Y_ac)*Rho_11 + (1-self.Y_h2)*Rho_12 - Rho_T_9

        # S_IC
        s1  = -self.C_xc + self.f_sI_xc*self.C_sI + self.f_ch_xc*self.C_ch + self.f_pr_xc*self.C_pr + self.f_li_xc*self.C_li + self.f_xI_xc*self.C_xI
        s2  = -self.C_ch  + self.C_su
        s3  = -self.C_pr  + self.C_aa
        s4  = -self.C_li  + (1-self.f_fa_li)*self.C_su + self.f_fa_li*self.C_fa
        s5  = -self.C_su  + (1-self.Y_su)*(self.f_bu_su*self.C_bu + self.f_pro_su*self.C_pro + self.f_ac_su*self.C_ac) + self.Y_su*self.C_bac
        s6  = -self.C_aa  + (1-self.Y_aa)*(self.f_va_aa*self.C_va + self.f_bu_aa*self.C_bu + self.f_pro_aa*self.C_pro + self.f_ac_aa*self.C_ac) + self.Y_aa*self.C_bac
        s7  = -self.C_fa  + (1-self.Y_fa)*0.7*self.C_ac + self.Y_fa*self.C_bac
        s8  = -self.C_va  + (1-self.Y_c4)*0.54*self.C_pro + (1-self.Y_c4)*0.31*self.C_ac + self.Y_c4*self.C_bac
        s9  = -self.C_bu  + (1-self.Y_c4)*0.8*self.C_ac + self.Y_c4*self.C_bac
        s10 = -self.C_pro + (1-self.Y_pro)*0.57*self.C_ac + self.Y_pro*self.C_bac
        s11 = -self.C_ac  + (1-self.Y_ac)*self.C_ch4 + self.Y_ac*self.C_bac
        s12 =  (1-self.Y_h2)*self.C_ch4 + self.Y_h2*self.C_bac
        s13 = -self.C_bac + self.C_xc
        dec_sum = Rho_13+Rho_14+Rho_15+Rho_16+Rho_17+Rho_18+Rho_19
        Sigma = s1*Rho_1 + s2*Rho_2 + s3*Rho_3 + s4*Rho_4 + s5*Rho_5 + s6*Rho_6 + s7*Rho_7 + s8*Rho_8 + s9*Rho_9 + s10*Rho_10 + s11*Rho_11 + s12*Rho_12 + s13*dec_sum

        diff_S_IC = q*(inf['S_IC']-S_IC) - Sigma - Rho_T_10
        diff_S_IN = q*(inf['S_IN']-S_IN) + (self.N_xc-self.f_xI_xc*self.N_I-self.f_sI_xc*self.N_I-self.f_pr_xc*self.N_aa)*Rho_1 - self.Y_su*self.N_bac*Rho_5 + (self.N_aa-self.Y_aa*self.N_bac)*Rho_6 - self.Y_fa*self.N_bac*Rho_7 - self.Y_c4*self.N_bac*Rho_8 - self.Y_c4*self.N_bac*Rho_9 - self.Y_pro*self.N_bac*Rho_10 - self.Y_ac*self.N_bac*Rho_11 - self.Y_h2*self.N_bac*Rho_12 + (self.N_bac-self.N_xc)*dec_sum
        diff_S_I  = q*(inf['S_I'] -S_I)  + self.f_sI_xc*Rho_1

        diff_X_xc  = q*(inf['X_xc']-X_xc) - Rho_1 + dec_sum
        diff_X_ch  = q*(inf['X_ch']-X_ch) + self.f_ch_xc*Rho_1 - Rho_2
        diff_X_pr  = q*(inf['X_pr']-X_pr) + self.f_pr_xc*Rho_1 - Rho_3
        diff_X_li  = q*(inf['X_li']-X_li) + self.f_li_xc*Rho_1 - Rho_4
        diff_X_su  = q*(inf['X_su']-X_su) + self.Y_su*Rho_5  - Rho_13
        diff_X_aa  = q*(inf['X_aa']-X_aa) + self.Y_aa*Rho_6  - Rho_14
        diff_X_fa  = q*(inf['X_fa']-X_fa) + self.Y_fa*Rho_7  - Rho_15
        diff_X_c4  = q*(inf['X_c4']-X_c4) + self.Y_c4*Rho_8  + self.Y_c4*Rho_9  - Rho_16
        diff_X_pro = q*(inf['X_pro']-X_pro)+ self.Y_pro*Rho_10 - Rho_17
        diff_X_ac  = q*(inf['X_ac']-X_ac) + self.Y_ac*Rho_11 - Rho_18
        diff_X_h2  = q*(inf['X_h2']-X_h2) + self.Y_h2*Rho_12 - Rho_19
        diff_X_I   = q*(inf['X_I'] -X_I)  + self.f_xI_xc*Rho_1

        diff_S_cation = q*(inf['S_cation']-S_cation)
        diff_S_anion  = q*(inf['S_anion'] -S_anion)

        # Ion states solved by DAE
        diff_S_H_ion   = 0.0;  diff_S_va_ion  = 0.0;  diff_S_bu_ion  = 0.0
        diff_S_pro_ion = 0.0;  diff_S_ac_ion  = 0.0;  diff_S_hco3_ion = 0.0
        diff_S_co2     = 0.0;  diff_S_nh3     = 0.0;  diff_S_nh4_ion  = 0.0

        diff_S_gas_h2  = (-q_gas/self.V_gas)*S_gas_h2  + Rho_T_8 *self.V_liq/self.V_gas
        diff_S_gas_ch4 = (-q_gas/self.V_gas)*S_gas_ch4 + Rho_T_9 *self.V_liq/self.V_gas
        diff_S_gas_co2 = (-q_gas/self.V_gas)*S_gas_co2 + Rho_T_10*self.V_liq/self.V_gas

        return np.array([
            diff_S_su, diff_S_aa, diff_S_fa, diff_S_va, diff_S_bu, diff_S_pro, diff_S_ac, diff_S_h2,
            diff_S_ch4, diff_S_IC, diff_S_IN, diff_S_I,
            diff_X_xc, diff_X_ch, diff_X_pr, diff_X_li, diff_X_su, diff_X_aa, diff_X_fa, diff_X_c4,
            diff_X_pro, diff_X_ac, diff_X_h2, diff_X_I,
            diff_S_cation, diff_S_anion,
            diff_S_H_ion, diff_S_va_ion, diff_S_bu_ion, diff_S_pro_ion, diff_S_ac_ion,
            diff_S_hco3_ion, diff_S_co2, diff_S_nh3, diff_S_nh4_ion,
            diff_S_gas_h2, diff_S_gas_ch4, diff_S_gas_co2
        ])

    # ──────────────────────────────────────────────────────────────────────────
    # DAE solver  (identical to ADM1Solver — Newton-Raphson for pH and S_h2)
    # ──────────────────────────────────────────────────────────────────────────

    def DAESolve(self):
        S_va = self.state['S_va'];  S_bu  = self.state['S_bu']
        S_pro = self.state['S_pro']; S_ac = self.state['S_ac']
        S_IC  = self.state['S_IC'];  S_IN  = self.state['S_IN']
        S_cation = self.state['S_cation']; S_anion = self.state['S_anion']
        S_H_ion  = self.state['S_H_ion'];  S_h2    = self.state['S_h2']

        tol = 1e-12;  maxIter = 1000

        # ── Solve S_H_ion (pH) ────────────────────────────────────────────────
        shdelta = 1.0;  i = 0
        while (abs(shdelta) > tol) and (i <= maxIter):
            S_va_ion   = self.K_a_va  * S_va  / (self.K_a_va  + S_H_ion)
            S_bu_ion   = self.K_a_bu  * S_bu  / (self.K_a_bu  + S_H_ion)
            S_pro_ion  = self.K_a_pro * S_pro / (self.K_a_pro + S_H_ion)
            S_ac_ion   = self.K_a_ac  * S_ac  / (self.K_a_ac  + S_H_ion)
            S_hco3_ion = self.K_a_co2 * S_IC  / (self.K_a_co2 + S_H_ion)
            S_nh3      = self.K_a_IN  * S_IN  / (self.K_a_IN  + S_H_ion)

            shdelta = (S_cation + (S_IN - S_nh3) + S_H_ion
                       - S_hco3_ion - S_ac_ion/64 - S_pro_ion/112
                       - S_bu_ion/160 - S_va_ion/208 - self.K_w/S_H_ion - S_anion)
            shgradeq = (1
                        + self.K_a_IN  * S_IN  / (self.K_a_IN  + S_H_ion)**2
                        + self.K_a_co2 * S_IC  / (self.K_a_co2 + S_H_ion)**2
                        + self.K_a_ac  * S_ac  / (self.K_a_ac  + S_H_ion)**2 / 64
                        + self.K_a_pro * S_pro / (self.K_a_pro + S_H_ion)**2 / 112
                        + self.K_a_bu  * S_bu  / (self.K_a_bu  + S_H_ion)**2 / 160
                        + self.K_a_va  * S_va  / (self.K_a_va  + S_H_ion)**2 / 208
                        + self.K_w / S_H_ion**2)
            S_H_ion -= shdelta / shgradeq
            if S_H_ion <= 0:
                S_H_ion = tol
            i += 1

        pH = -np.log10(S_H_ion)

        # ── Solve S_h2 ────────────────────────────────────────────────────────
        S_su  = self.state['S_su'];  S_aa = self.state['S_aa']
        S_fa  = self.state['S_fa'];  S_pro = self.state['S_pro']
        X_su  = self.state['X_su'];  X_aa  = self.state['X_aa']
        X_fa  = self.state['X_fa'];  X_c4  = self.state['X_c4']
        X_pro = self.state['X_pro']; X_h2  = self.state['X_h2']
        S_gas_h2 = self.state['S_gas_h2']

        I_pH_aa  = (self.K_pH_aa ** self.nn_aa) / (S_H_ion ** self.nn_aa + self.K_pH_aa ** self.nn_aa)
        I_pH_h2  = (self.K_pH_h2 ** self.n_h2)  / (S_H_ion ** self.n_h2  + self.K_pH_h2 ** self.n_h2)
        I_IN_lim = 1.0 / (1.0 + self.K_S_IN / S_IN)
        eps = 1e-6

        S_h2delta = 1.0;  j = 0
        while (abs(S_h2delta) > tol) and (j <= maxIter):
            I_h2_fa  = 1.0 / (1.0 + S_h2 / self.K_I_h2_fa)
            I_h2_c4  = 1.0 / (1.0 + S_h2 / self.K_I_h2_c4)
            I_h2_pro = 1.0 / (1.0 + S_h2 / self.K_I_h2_pro)

            I_5  = I_pH_aa * I_IN_lim;  I_6 = I_5
            I_7  = I_pH_aa * I_IN_lim * I_h2_fa
            I_8  = I_pH_aa * I_IN_lim * I_h2_c4;  I_9 = I_8
            I_10 = I_pH_aa * I_IN_lim * I_h2_pro
            I_12 = I_pH_h2 * I_IN_lim

            Rho_5  = self.k_m_su  * S_su  / (self.K_S_su  + S_su)  * X_su  * I_5
            Rho_6  = self.k_m_aa  * S_aa  / (self.K_S_aa  + S_aa)  * X_aa  * I_6
            Rho_7  = self.k_m_fa  * S_fa  / (self.K_S_fa  + S_fa)  * X_fa  * I_7
            Rho_8  = self.k_m_c4  * S_va  / (self.K_S_c4  + S_va)  * X_c4  * (S_va / (S_bu + S_va + eps)) * I_8
            Rho_9  = self.k_m_c4  * S_bu  / (self.K_S_c4  + S_bu)  * X_c4  * (S_bu / (S_bu + S_va + eps)) * I_9
            Rho_10 = self.k_m_pro * S_pro / (self.K_S_pro + S_pro) * X_pro * I_10
            Rho_12 = self.k_m_h2  * S_h2  / (self.K_S_h2  + S_h2)  * X_h2  * I_12

            p_gas_h2 = S_gas_h2 * self.R * self.T_op / 16
            Rho_T_8  = self.k_L_a * (S_h2 - 16 * self.K_H_h2 * p_gas_h2)

            S_h2delta = (self.q_ad/self.V_liq*(self.influent['S_h2'] - S_h2)
                         + (1-self.Y_su)*self.f_h2_su*Rho_5
                         + (1-self.Y_aa)*self.f_h2_aa*Rho_6
                         + (1-self.Y_fa)*0.3*Rho_7
                         + (1-self.Y_c4)*0.15*Rho_8
                         + (1-self.Y_c4)*0.2*Rho_9
                         + (1-self.Y_pro)*0.43*Rho_10
                         - Rho_12 - Rho_T_8)

            S_h2gradeq = (
                -self.q_ad/self.V_liq
                - 0.3*(1-self.Y_fa)*self.k_m_fa*S_fa/(self.K_S_fa+S_fa)*X_fa*I_pH_aa/(1+self.K_S_IN/S_IN)/((1+S_h2/self.K_I_h2_fa)**2)/self.K_I_h2_fa
                - 0.15*(1-self.Y_c4)*self.k_m_c4*S_va**2/(self.K_S_c4+S_va)*X_c4/(S_bu+S_va+eps)*I_pH_aa/(1+self.K_S_IN/S_IN)/((1+S_h2/self.K_I_h2_c4)**2)/self.K_I_h2_c4
                - 0.2*(1-self.Y_c4)*self.k_m_c4*S_bu**2/(self.K_S_c4+S_bu)*X_c4/(S_bu+S_va+eps)*I_pH_aa/(1+self.K_S_IN/S_IN)/((1+S_h2/self.K_I_h2_c4)**2)/self.K_I_h2_c4
                - 0.43*(1-self.Y_pro)*self.k_m_pro*S_pro/(self.K_S_pro+S_pro)*X_pro*I_pH_aa/(1+self.K_S_IN/S_IN)/((1+S_h2/self.K_I_h2_pro)**2)/self.K_I_h2_pro
                - self.k_m_h2/(self.K_S_h2+S_h2)*X_h2*I_pH_h2/(1+self.K_S_IN/S_IN)
                + self.k_m_h2*S_h2/(self.K_S_h2+S_h2)**2*X_h2*I_pH_h2/(1+self.K_S_IN/S_IN)
                - self.k_L_a)

            S_h2 -= S_h2delta / S_h2gradeq
            if S_h2 <= 0:
                S_h2 = tol
            j += 1

        # Write back
        self.state['S_H_ion']    = S_H_ion
        self.state['S_va_ion']   = S_va_ion
        self.state['S_bu_ion']   = S_bu_ion
        self.state['S_pro_ion']  = S_pro_ion
        self.state['S_ac_ion']   = S_ac_ion
        self.state['S_hco3_ion'] = S_hco3_ion
        self.state['S_nh3']      = S_nh3
        self.state['S_nh4_ion']  = S_IN - S_nh3
        self.state['S_co2']      = S_IC - S_hco3_ion
        self.state['pH']         = pH
        self.state['S_h2']       = S_h2
        return pH

    # ──────────────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────────────

    STATE_NAMES = [
        'S_su','S_aa','S_fa','S_va','S_bu','S_pro','S_ac','S_h2',
        'S_ch4','S_IC','S_IN','S_I',
        'X_xc','X_ch','X_pr','X_li','X_su','X_aa','X_fa','X_c4',
        'X_pro','X_ac','X_h2','X_I',
        'S_cation','S_anion',
        'S_H_ion','S_va_ion','S_bu_ion','S_pro_ion','S_ac_ion',
        'S_hco3_ion','S_co2','S_nh3','S_nh4_ion',
        'S_gas_h2','S_gas_ch4','S_gas_co2',
    ]

    def step(self, dt: float = 1.0) -> Tuple[Dict, float]:
        """
        Execute one simulation step.

        Args:
            dt: Time step in days (internally subdivided at 15-min resolution)

        Returns:
            new_state: Updated state dictionary (38 ADM1 keys)
            q_ch4:     Methane flow rate (m³/d)
        """
        internal_dt   = 0.01041667  # 15 min
        num_substeps  = max(1, int(np.ceil(dt / internal_dt)))
        actual_dt_sub = dt / num_substeps

        for _ in range(num_substeps):
            sv = np.array([self.state[k] for k in self.STATE_NAMES])

            sol = scipy.integrate.solve_ivp(
                self.ADM1_ODE,
                [0, actual_dt_sub],
                sv,
                method='DOP853',
                rtol=1e-6,
                atol=1e-8,
            )
            final = sol.y[:, -1]
            for i, name in enumerate(self.STATE_NAMES):
                self.state[name] = final[i]

            self.DAESolve()

            self.state['S_nh4_ion'] = self.state['S_IN'] - self.state['S_nh3']
            self.state['S_co2']     = self.state['S_IC'] - self.state['S_hco3_ion']

        return self.state.copy(), self.q_ch4
