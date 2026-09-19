# Superseded

Both scripts here derive quantities from the reference plant's VFA record.
That record is not commensurable with a mechanistic model's VFA state, so
neither derivation is used any more.

- `derive_thresholds.py` — transferred the plant's VFA limits to the simulator
  as multiples of routine operation.  The environment's limits now come from
  the operating ranges reported for chromatographically determined VFA
  (300 mg/L normal upper bound, 1500 mg/L inhibition onset), which are on the
  same measurement basis as the model state.
- `scada_calibration.py` — mapped plant FOS/TAC readings onto simulator
  observations through a two-anchor linear fit.  The observation space is now
  the six quantities the facility records daily, in their own units, so no
  mapping is needed.

Why the plant's VFA record is not usable for either purpose, established from
the record itself rather than from a stated analytical method:

1.  It shows no correlation with FOG loading at any lag from 0 to 28 d
    (+0.02 to -0.11) while correlating with alkalinity at +0.61.  Fat, oil and
    grease is the classic driver of VFA accumulation in co-digestion.
2.  Its day-to-day variability (11.3 % median, above 50 % on 5 % of days)
    exceeds what a 1836 m3 reactor at 19.5 d hydraulic retention can express;
    dilution alone bounds bulk concentration change at 5.1 %/d.  Alkalinity
    measured on the same samples changes by 3.0 % median.
3.  Chemostat theory bounds steady-state acetate at K_S*k_dec/(mu_max*I-k_dec),
    i.e. single to low double digits mg/L, irrespective of biomass; the record
    sits at 1178 mg/L median.

The facility's other records remain in use: reactor volume, feed volumes and
composition, and biogas production.
