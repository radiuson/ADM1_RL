# Figure captions

**Fig. 1 — Production against constraint exceedance for four controller families.**
(a) Each family is swept over its own tuning parameter, so every curve is that
family's own production-exceedance frontier rather than a single operating
point: feed rate for constant feeding, action level for the rule-based
supervisor, setpoint for the PI loop, and penalty weight for the learned policy.
Exceedance is the share of operating time above the 300 mg/L action level,
pooled over eight loading scenarios of 60 d each. (b) Production of the learned
policy relative to each conventional family at matched exceedance, obtained by
interpolating the learned frontier. The advantage widens as the exceedance
requirement tightens: at about 20 % of the time above the level a constant feed
reaches 96 % of the learned policy's production, whereas below 3 % the
conventional families give up 9-50 %.

**Fig. 2 — Where the difference comes from.**
(a) Distribution of VFA over all scenarios for the learned policy at w = 0.5 and
the PI loop at setpoint 250, two controllers that spend the same 18.3 % of
operating time above the action level. The shaded margin band is the 30 % of
range immediately below the level. The PI loop holds the digester well under the
band and enters it only in transients; the learned policy places its mode inside
it. (b) Time spent within the margin band for all eight controllers, with each
controller's exceedance share in parentheses. Learned policies occupy the band
28-35 % of the time; every conventional controller occupies it 1.7-12.1 %,
including those with higher exceedance.

Both figures pool eight loading scenarios (OLR 1.5-5.9 kg COD m^-3 d^-1) over
three evaluation seeds. Learned policies are Soft Actor-Critic at 150k steps.
