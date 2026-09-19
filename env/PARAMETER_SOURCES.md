# Parameter sources

Every biochemical, stoichiometric and physico-chemical parameter in
`adm1_solver_std.py` traces to one of two published sets. Environment-level
values that are design choices rather than measured quantities are listed
separately and labelled as such.

## Kinetic and stoichiometric parameters

| Source | Parameters |
|---|---|
| Batstone et al. (2002), *Water Sci. Technol.* 45(10):65–73 — ADM1 | All stoichiometric coefficients (`f_*`, `Y_*`, `N_*`), `k_m_aa`, `K_S_aa`, `k_m_h2`, `K_S_h2`, `K_S_IN`, all `k_dec_X_*`, all hydrogen and ammonia inhibition constants, all pH inhibition parameters, `k_p`, `k_L_a` |
| Razaviarani & Buchanan (2015), *Chem. Eng. J.* 266:91–99, Table 4 — ADM1 calibrated for mesophilic co-digestion of municipal wastewater sludge with restaurant grease trap waste | `k_dis` 0.2, `k_hyd_ch` 0.75, `k_hyd_pr` 0.7, `k_hyd_li` 2.1, `k_m_su` 37.4, `K_S_su` 0.496, `k_m_c4` 14.1, `K_S_c4` 0.193, `k_m_fa` 5.9, `K_S_fa` 0.3815, `k_m_pro` 17.1, `K_S_pro` 0.0635, `k_m_ac` 10.9, `K_S_ac` 0.0961 |

The calibrated hydrolysis rates matter for this benchmark. ADM1's default
10 d⁻¹ corresponds to a 2.4 h time constant, so the digester tracks daily feed
changes almost instantaneously and there is no multi-day VFA accumulation for a
controller to anticipate. The calibrated values put the rate-limiting step back
on a daily scale.

Physico-chemical constants (`K_a_*`, `K_H_*`, `k_A_B_*`, van't Hoff
corrections) and the reactor geometry (`V_liq` 3400 m³, `V_gas` 300 m³) follow
the ADM1 / BSM2 implementation.

Influent: the BSM2 characterisation shipped in `data/digester_influent.csv`.

## Safety thresholds

| Threshold | Value | Basis |
|---|---|---|
| VFA soft | 0.85 kg COD/m³ (797 mg/L acetic) | Acetic acid above 800 mg/L signals impending digester failure (Hill et al., *Trans. ASAE* 30(2), 1987) |
| VFA hard | 1.60 kg COD/m³ (1500 mg/L acetic) | Lower bound of the 1500–3000 mg/L total-VFA band used in practical AD monitoring |
| NH₃ soft / hard | 0.004 / 0.010 kmol N/m³ | 2.2× / 5.6× ADM1's 50 % inhibition constant for aceticlastic methanogens (`K_I_nh3` = 0.0018) |
| pH soft | 6.8–7.8 | Range for stable mesophilic methanogenesis |
| pH hard | 5.8–8.8 | ADM1 places complete pH inhibition of acetate uptake at `pH_LL_ac` = 6.0 |

## Scenario loading

Organic loading rate at the nominal action, with influent COD 47.6 kg/m³,
q = 178.5 m³/d and V = 3400 m³. Typical municipal AD runs at 1.5–3.5 kg
COD/m³/d; high-rate operation extends to 5–6.

| Scenario | Influent multiplier | OLR (kg COD/m³/d) |
|---|---|---|
| `low_load` | 0.7 | 1.75 |
| `nominal`, `shock_load`, `pre_stressed` | 1.0 | 2.50 |
| `high_load` | 1.3 | 3.25 |
| `high_load_real` | 1.8 | 4.50 |
| `lipid_overload` | 2.2 | 5.50 |

`lipid_overload` is the only scenario in which over-feeding drives VFA past the
hard threshold, so it is what makes the safety objective binding.

## Design choices (not literature-derived)

These are benchmark design values and are labelled as such rather than
presented as physical constants.

- Action bounds: `q_ad` 50–300 m³/d (HRT 11.3–68 d), `feed_mult` 0.7–1.3.
- Reward weights: `production_scale` 2000, pH/VFA/NH₃ penalty scales,
  energy and stability penalty caps. These are algorithm design parameters.
- Episode length 60 d and the daily control interval.
- Observation bounds, set from the ranges the environment reaches across the
  loading scenarios and required to bracket the safety thresholds so the hazard
  region stays resolvable after normalisation.

## Known limitations

- Simulated pH spans roughly 7.19–7.53 across the loading range, so the pH
  channel carries little information and the pH thresholds are rarely active.
- Absolute VFA is well below what the Muscatine WRRF record shows for routine
  operation; the environment represents a standard mesophilic digester rather
  than that specific plant.
