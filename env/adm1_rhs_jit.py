"""Auto-generated numba kernel for ADM1_ODE.  DO NOT EDIT BY HAND.

Regenerate with scratchpad/gen_jit.py after any change to
ADM1SolverStd.ADM1_ODE.  The body below is a mechanical transcription of
that method: ``self.<param>`` becomes ``p[i]`` and ``inf['<key>']``
becomes ``f[i]``, with the index order fixed by PARAM_NAMES/INFLUENT_KEYS.
fastmath is off so the arithmetic matches the interpreted version bit for
bit.
"""
import numpy as np
from numba import njit

PARAM_NAMES = ['C_aa', 'C_ac', 'C_bac', 'C_bu', 'C_ch', 'C_ch4', 'C_fa', 'C_li', 'C_pr', 'C_pro', 'C_sI', 'C_su', 'C_va', 'C_xI', 'C_xc', 'K_H_ch4', 'K_H_co2', 'K_H_h2', 'K_I_h2_c4', 'K_I_h2_fa', 'K_I_h2_pro', 'K_I_nh3', 'K_S_IN', 'K_S_aa', 'K_S_ac', 'K_S_c4', 'K_S_fa', 'K_S_h2', 'K_S_pro', 'K_S_su', 'K_pH_aa', 'K_pH_ac', 'K_pH_h2', 'N_I', 'N_aa', 'N_bac', 'N_xc', 'R', 'T_op', 'V_gas', 'V_liq', 'Y_aa', 'Y_ac', 'Y_c4', 'Y_fa', 'Y_h2', 'Y_pro', 'Y_su', 'f_ac_aa', 'f_ac_su', 'f_bu_aa', 'f_bu_su', 'f_ch_xc', 'f_fa_li', 'f_li_xc', 'f_pr_xc', 'f_pro_aa', 'f_pro_su', 'f_sI_xc', 'f_va_aa', 'f_xI_xc', 'k_L_a', 'k_dec_X_aa', 'k_dec_X_ac', 'k_dec_X_c4', 'k_dec_X_fa', 'k_dec_X_h2', 'k_dec_X_pro', 'k_dec_X_su', 'k_dis', 'k_hyd_ch', 'k_hyd_li', 'k_hyd_pr', 'k_m_aa', 'k_m_ac', 'k_m_c4', 'k_m_fa', 'k_m_h2', 'k_m_pro', 'k_m_su', 'k_p', 'n_ac', 'n_h2', 'nn_aa', 'p_atm', 'p_gas_h2o', 'q_ad']
INFLUENT_KEYS = ['S_I', 'S_IC', 'S_IN', 'S_aa', 'S_ac', 'S_anion', 'S_bu', 'S_cation', 'S_ch4', 'S_fa', 'S_pro', 'S_su', 'S_va', 'X_I', 'X_aa', 'X_ac', 'X_c4', 'X_ch', 'X_fa', 'X_h2', 'X_li', 'X_pr', 'X_pro', 'X_su', 'X_xc']
GAS_OUTPUTS = ['p_gas', 'q_gas', 'q_ch4', 'q_co2']


def pack(solver):
    """Build the (params, influent, gas-out) float64 arrays the kernel expects.

    ``gas`` is an output buffer: the kernel writes GAS_OUTPUTS into it on
    every call, so after an integration it holds the values from the last
    right-hand-side evaluation -- exactly what the interpreted method left
    on ``self``.
    """
    p = np.array([float(getattr(solver, n)) for n in PARAM_NAMES])
    f = np.array([float(solver.influent[k]) for k in INFLUENT_KEYS])
    return p, f, np.zeros(len(GAS_OUTPUTS))


@njit(cache=True, fastmath=False)
def adm1_rhs(t, sv, p, f, gas):
    p_C_aa = p[0]
    p_C_ac = p[1]
    p_C_bac = p[2]
    p_C_bu = p[3]
    p_C_ch = p[4]
    p_C_ch4 = p[5]
    p_C_fa = p[6]
    p_C_li = p[7]
    p_C_pr = p[8]
    p_C_pro = p[9]
    p_C_sI = p[10]
    p_C_su = p[11]
    p_C_va = p[12]
    p_C_xI = p[13]
    p_C_xc = p[14]
    p_K_H_ch4 = p[15]
    p_K_H_co2 = p[16]
    p_K_H_h2 = p[17]
    p_K_I_h2_c4 = p[18]
    p_K_I_h2_fa = p[19]
    p_K_I_h2_pro = p[20]
    p_K_I_nh3 = p[21]
    p_K_S_IN = p[22]
    p_K_S_aa = p[23]
    p_K_S_ac = p[24]
    p_K_S_c4 = p[25]
    p_K_S_fa = p[26]
    p_K_S_h2 = p[27]
    p_K_S_pro = p[28]
    p_K_S_su = p[29]
    p_K_pH_aa = p[30]
    p_K_pH_ac = p[31]
    p_K_pH_h2 = p[32]
    p_N_I = p[33]
    p_N_aa = p[34]
    p_N_bac = p[35]
    p_N_xc = p[36]
    p_R = p[37]
    p_T_op = p[38]
    p_V_gas = p[39]
    p_V_liq = p[40]
    p_Y_aa = p[41]
    p_Y_ac = p[42]
    p_Y_c4 = p[43]
    p_Y_fa = p[44]
    p_Y_h2 = p[45]
    p_Y_pro = p[46]
    p_Y_su = p[47]
    p_f_ac_aa = p[48]
    p_f_ac_su = p[49]
    p_f_bu_aa = p[50]
    p_f_bu_su = p[51]
    p_f_ch_xc = p[52]
    p_f_fa_li = p[53]
    p_f_li_xc = p[54]
    p_f_pr_xc = p[55]
    p_f_pro_aa = p[56]
    p_f_pro_su = p[57]
    p_f_sI_xc = p[58]
    p_f_va_aa = p[59]
    p_f_xI_xc = p[60]
    p_k_L_a = p[61]
    p_k_dec_X_aa = p[62]
    p_k_dec_X_ac = p[63]
    p_k_dec_X_c4 = p[64]
    p_k_dec_X_fa = p[65]
    p_k_dec_X_h2 = p[66]
    p_k_dec_X_pro = p[67]
    p_k_dec_X_su = p[68]
    p_k_dis = p[69]
    p_k_hyd_ch = p[70]
    p_k_hyd_li = p[71]
    p_k_hyd_pr = p[72]
    p_k_m_aa = p[73]
    p_k_m_ac = p[74]
    p_k_m_c4 = p[75]
    p_k_m_fa = p[76]
    p_k_m_h2 = p[77]
    p_k_m_pro = p[78]
    p_k_m_su = p[79]
    p_k_p = p[80]
    p_n_ac = p[81]
    p_n_h2 = p[82]
    p_nn_aa = p[83]
    p_p_atm = p[84]
    p_p_gas_h2o = p[85]
    p_q_ad = p[86]
    i_S_I = f[0]
    i_S_IC = f[1]
    i_S_IN = f[2]
    i_S_aa = f[3]
    i_S_ac = f[4]
    i_S_anion = f[5]
    i_S_bu = f[6]
    i_S_cation = f[7]
    i_S_ch4 = f[8]
    i_S_fa = f[9]
    i_S_pro = f[10]
    i_S_su = f[11]
    i_S_va = f[12]
    i_X_I = f[13]
    i_X_aa = f[14]
    i_X_ac = f[15]
    i_X_c4 = f[16]
    i_X_ch = f[17]
    i_X_fa = f[18]
    i_X_h2 = f[19]
    i_X_li = f[20]
    i_X_pr = f[21]
    i_X_pro = f[22]
    i_X_su = f[23]
    i_X_xc = f[24]
    (S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4,
     S_IC, S_IN, S_I,
     X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I,
     S_cation, S_anion,
     S_H_ion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion,
     S_hco3_ion, S_co2, S_nh3, S_nh4_ion,
     S_gas_h2, S_gas_ch4, S_gas_co2) = sv

    # Derived ion quantities
    S_nh4_ion = S_IN - S_nh3
    S_co2     = S_IC - S_hco3_ion

    # ── Inhibition functions ──────────────────────────────────────────────
    I_pH_aa  = (p_K_pH_aa ** p_nn_aa) / (S_H_ion ** p_nn_aa + p_K_pH_aa ** p_nn_aa)
    I_pH_ac  = (p_K_pH_ac ** p_n_ac)  / (S_H_ion ** p_n_ac  + p_K_pH_ac ** p_n_ac)
    I_pH_h2  = (p_K_pH_h2 ** p_n_h2)  / (S_H_ion ** p_n_h2  + p_K_pH_h2 ** p_n_h2)
    I_IN_lim = 1.0 / (1.0 + p_K_S_IN / S_IN)
    I_h2_fa  = 1.0 / (1.0 + S_h2 / p_K_I_h2_fa)
    I_h2_c4  = 1.0 / (1.0 + S_h2 / p_K_I_h2_c4)
    I_h2_pro = 1.0 / (1.0 + S_h2 / p_K_I_h2_pro)
    I_nh3    = 1.0 / (1.0 + S_nh3 / p_K_I_nh3)

    I_5  = I_pH_aa * I_IN_lim
    I_6  = I_5
    I_7  = I_pH_aa * I_IN_lim * I_h2_fa
    I_8  = I_pH_aa * I_IN_lim * I_h2_c4
    I_9  = I_8
    I_10 = I_pH_aa * I_IN_lim * I_h2_pro
    I_11 = I_pH_ac * I_IN_lim * I_nh3
    I_12 = I_pH_h2 * I_IN_lim

    # ── Biochemical process rates ─────────────────────────────────────────
    Rho_1  = p_k_dis    * X_xc
    Rho_2  = p_k_hyd_ch * X_ch
    Rho_3  = p_k_hyd_pr * X_pr
    Rho_4  = p_k_hyd_li * X_li
    Rho_5  = p_k_m_su  * S_su  / (p_K_S_su  + S_su)  * X_su  * I_5
    Rho_6  = p_k_m_aa  * S_aa  / (p_K_S_aa  + S_aa)  * X_aa  * I_6
    Rho_7  = p_k_m_fa  * S_fa  / (p_K_S_fa  + S_fa)  * X_fa  * I_7
    Rho_8  = p_k_m_c4  * S_va  / (p_K_S_c4  + S_va)  * X_c4  * (S_va / (S_bu + S_va + 1e-6)) * I_8
    Rho_9  = p_k_m_c4  * S_bu  / (p_K_S_c4  + S_bu)  * X_c4  * (S_bu / (S_bu + S_va + 1e-6)) * I_9
    Rho_10 = p_k_m_pro * S_pro / (p_K_S_pro + S_pro) * X_pro * I_10
    Rho_11 = p_k_m_ac  * S_ac  / (p_K_S_ac  + S_ac)  * X_ac  * I_11
    Rho_12 = p_k_m_h2  * S_h2  / (p_K_S_h2  + S_h2)  * X_h2  * I_12
    Rho_13 = p_k_dec_X_su  * X_su
    Rho_14 = p_k_dec_X_aa  * X_aa
    Rho_15 = p_k_dec_X_fa  * X_fa
    Rho_16 = p_k_dec_X_c4  * X_c4
    Rho_17 = p_k_dec_X_pro * X_pro
    Rho_18 = p_k_dec_X_ac  * X_ac
    Rho_19 = p_k_dec_X_h2  * X_h2

    # ── Gas phase ─────────────────────────────────────────────────────────
    T = p_T_op
    p_gas_h2  = S_gas_h2  * p_R * T / 16
    p_gas_ch4 = S_gas_ch4 * p_R * T / 64
    p_gas_co2 = S_gas_co2 * p_R * T

    p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_p_gas_h2o
    q_gas = p_k_p * (p_gas - p_p_atm)
    if q_gas < 0:
        q_gas = 0.0

    gas[0] = p_gas
    gas[1] = q_gas
    gas[2] = q_gas * (p_gas_ch4 / p_gas) if p_gas > 0 else 0.0
    gas[3] = q_gas * (p_gas_co2 / p_gas) if p_gas > 0 else 0.0

    Rho_T_8  = p_k_L_a * (S_h2  - 16  * p_K_H_h2  * p_gas_h2)
    Rho_T_9  = p_k_L_a * (S_ch4 - 64  * p_K_H_ch4 * p_gas_ch4)
    Rho_T_10 = p_k_L_a * (S_co2 - p_K_H_co2 * p_gas_co2)

    # ── Differential equations ────────────────────────────────────────────
    q = p_q_ad / p_V_liq

    diff_S_su  = q*(i_S_su -S_su)  + Rho_2 + (1-p_f_fa_li)*Rho_4 - Rho_5
    diff_S_aa  = q*(i_S_aa -S_aa)  + Rho_3 - Rho_6
    diff_S_fa  = q*(i_S_fa -S_fa)  + p_f_fa_li*Rho_4 - Rho_7
    diff_S_va  = q*(i_S_va -S_va)  + (1-p_Y_aa)*p_f_va_aa*Rho_6 - Rho_8
    diff_S_bu  = q*(i_S_bu -S_bu)  + (1-p_Y_su)*p_f_bu_su*Rho_5 + (1-p_Y_aa)*p_f_bu_aa*Rho_6 - Rho_9
    diff_S_pro = q*(i_S_pro-S_pro) + (1-p_Y_su)*p_f_pro_su*Rho_5 + (1-p_Y_aa)*p_f_pro_aa*Rho_6 + (1-p_Y_c4)*0.54*Rho_8 - Rho_10
    diff_S_ac  = q*(i_S_ac -S_ac)  + (1-p_Y_su)*p_f_ac_su*Rho_5 + (1-p_Y_aa)*p_f_ac_aa*Rho_6 + (1-p_Y_fa)*0.7*Rho_7 + (1-p_Y_c4)*0.31*Rho_8 + (1-p_Y_c4)*0.8*Rho_9 + (1-p_Y_pro)*0.57*Rho_10 - Rho_11
    diff_S_h2  = 0.0  # solved by DAE
    diff_S_ch4 = q*(i_S_ch4-S_ch4) + (1-p_Y_ac)*Rho_11 + (1-p_Y_h2)*Rho_12 - Rho_T_9

    # S_IC
    s1  = -p_C_xc + p_f_sI_xc*p_C_sI + p_f_ch_xc*p_C_ch + p_f_pr_xc*p_C_pr + p_f_li_xc*p_C_li + p_f_xI_xc*p_C_xI
    s2  = -p_C_ch  + p_C_su
    s3  = -p_C_pr  + p_C_aa
    s4  = -p_C_li  + (1-p_f_fa_li)*p_C_su + p_f_fa_li*p_C_fa
    s5  = -p_C_su  + (1-p_Y_su)*(p_f_bu_su*p_C_bu + p_f_pro_su*p_C_pro + p_f_ac_su*p_C_ac) + p_Y_su*p_C_bac
    s6  = -p_C_aa  + (1-p_Y_aa)*(p_f_va_aa*p_C_va + p_f_bu_aa*p_C_bu + p_f_pro_aa*p_C_pro + p_f_ac_aa*p_C_ac) + p_Y_aa*p_C_bac
    s7  = -p_C_fa  + (1-p_Y_fa)*0.7*p_C_ac + p_Y_fa*p_C_bac
    s8  = -p_C_va  + (1-p_Y_c4)*0.54*p_C_pro + (1-p_Y_c4)*0.31*p_C_ac + p_Y_c4*p_C_bac
    s9  = -p_C_bu  + (1-p_Y_c4)*0.8*p_C_ac + p_Y_c4*p_C_bac
    s10 = -p_C_pro + (1-p_Y_pro)*0.57*p_C_ac + p_Y_pro*p_C_bac
    s11 = -p_C_ac  + (1-p_Y_ac)*p_C_ch4 + p_Y_ac*p_C_bac
    s12 =  (1-p_Y_h2)*p_C_ch4 + p_Y_h2*p_C_bac
    s13 = -p_C_bac + p_C_xc
    dec_sum = Rho_13+Rho_14+Rho_15+Rho_16+Rho_17+Rho_18+Rho_19
    Sigma = s1*Rho_1 + s2*Rho_2 + s3*Rho_3 + s4*Rho_4 + s5*Rho_5 + s6*Rho_6 + s7*Rho_7 + s8*Rho_8 + s9*Rho_9 + s10*Rho_10 + s11*Rho_11 + s12*Rho_12 + s13*dec_sum

    diff_S_IC = q*(i_S_IC-S_IC) - Sigma - Rho_T_10
    diff_S_IN = q*(i_S_IN-S_IN) + (p_N_xc-p_f_xI_xc*p_N_I-p_f_sI_xc*p_N_I-p_f_pr_xc*p_N_aa)*Rho_1 - p_Y_su*p_N_bac*Rho_5 + (p_N_aa-p_Y_aa*p_N_bac)*Rho_6 - p_Y_fa*p_N_bac*Rho_7 - p_Y_c4*p_N_bac*Rho_8 - p_Y_c4*p_N_bac*Rho_9 - p_Y_pro*p_N_bac*Rho_10 - p_Y_ac*p_N_bac*Rho_11 - p_Y_h2*p_N_bac*Rho_12 + (p_N_bac-p_N_xc)*dec_sum
    diff_S_I  = q*(i_S_I -S_I)  + p_f_sI_xc*Rho_1

    diff_X_xc  = q*(i_X_xc-X_xc) - Rho_1 + dec_sum
    diff_X_ch  = q*(i_X_ch-X_ch) + p_f_ch_xc*Rho_1 - Rho_2
    diff_X_pr  = q*(i_X_pr-X_pr) + p_f_pr_xc*Rho_1 - Rho_3
    diff_X_li  = q*(i_X_li-X_li) + p_f_li_xc*Rho_1 - Rho_4
    diff_X_su  = q*(i_X_su-X_su) + p_Y_su*Rho_5  - Rho_13
    diff_X_aa  = q*(i_X_aa-X_aa) + p_Y_aa*Rho_6  - Rho_14
    diff_X_fa  = q*(i_X_fa-X_fa) + p_Y_fa*Rho_7  - Rho_15
    diff_X_c4  = q*(i_X_c4-X_c4) + p_Y_c4*Rho_8  + p_Y_c4*Rho_9  - Rho_16
    diff_X_pro = q*(i_X_pro-X_pro)+ p_Y_pro*Rho_10 - Rho_17
    diff_X_ac  = q*(i_X_ac-X_ac) + p_Y_ac*Rho_11 - Rho_18
    diff_X_h2  = q*(i_X_h2-X_h2) + p_Y_h2*Rho_12 - Rho_19
    diff_X_I   = q*(i_X_I -X_I)  + p_f_xI_xc*Rho_1

    diff_S_cation = q*(i_S_cation-S_cation)
    diff_S_anion  = q*(i_S_anion -S_anion)

    # Ion states solved by DAE
    diff_S_H_ion   = 0.0;  diff_S_va_ion  = 0.0;  diff_S_bu_ion  = 0.0
    diff_S_pro_ion = 0.0;  diff_S_ac_ion  = 0.0;  diff_S_hco3_ion = 0.0
    diff_S_co2     = 0.0;  diff_S_nh3     = 0.0;  diff_S_nh4_ion  = 0.0

    diff_S_gas_h2  = (-q_gas/p_V_gas)*S_gas_h2  + Rho_T_8 *p_V_liq/p_V_gas
    diff_S_gas_ch4 = (-q_gas/p_V_gas)*S_gas_ch4 + Rho_T_9 *p_V_liq/p_V_gas
    diff_S_gas_co2 = (-q_gas/p_V_gas)*S_gas_co2 + Rho_T_10*p_V_liq/p_V_gas

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
