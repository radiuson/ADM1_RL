"""A two-stage predictor for model predictive control of this environment.

Predicting with the full 38-state ADM1 costs about a second per simulated day,
which is why an earlier attempt at NMPC here was abandoned.  The anaerobic
digestion control literature does not predict with the full model either: a
two-stage (AM2-type) reduction is used, with its parameters fitted to the plant
and its unmeasured states carried by an observer.  This module provides that
reduction and the fit.

    dX1/dt = (mu1 - alpha*D - kd) X1
    dX2/dt = (mu2 - alpha*D - kd) X2
    dS1/dt = D (S1in - S1) - k1 mu1 X1
    dS2/dt = D (S2in - S2) + k2 mu1 X1 - k3 mu2 X2

    mu1 = m1 S1 / (Ks1 + S1)                     acidogenesis, Monod
    mu2 = m2 S2 / (Ks2 + S2 + (S2/KI)^2)         methanogenesis, Haldane

S2 is the VFA pool the constraint is placed on, and methane production is taken
proportional to the methanogenic reaction rate.  `alpha` is the fraction of
biomass leaving with the effluent, which is below one when solids are retained.
"""
import numpy as np
from scipy.integrate import solve_ivp


PARAM_NAMES = ('m1', 'Ks1', 'm2', 'Ks2', 'KI', 'k1', 'k2', 'k3',
               'alpha', 'kd', 'S1in', 'S2in', 'kCH4')


class AM2:
    def __init__(self, p):
        self.p = dict(p)

    def deriv(self, t, x, D):
        p = self.p
        X1, X2, S1, S2 = np.maximum(x, 0.0)
        mu1 = p['m1'] * S1 / (p['Ks1'] + S1 + 1e-12)
        mu2 = p['m2'] * S2 / (p['Ks2'] + S2 + S2 * S2 / p['KI'] + 1e-12)
        aD = p['alpha'] * D
        return [(mu1 - aD - p['kd']) * X1,
                (mu2 - aD - p['kd']) * X2,
                D * (p['S1in'] - S1) - p['k1'] * mu1 * X1,
                D * (p['S2in'] - S2) + p['k2'] * mu1 * X1 - p['k3'] * mu2 * X2]

    def step(self, x, D, dt=1.0):
        s = solve_ivp(self.deriv, [0, dt], np.maximum(x, 0.0), args=(D,),
                      method='LSODA', rtol=1e-5, atol=1e-8)
        return np.maximum(s.y[:, -1], 0.0)

    def ch4(self, x, D):
        p = self.p
        X2, S2 = max(x[1], 0.0), max(x[3], 0.0)
        mu2 = p['m2'] * S2 / (p['Ks2'] + S2 + S2 * S2 / p['KI'] + 1e-12)
        return p['kCH4'] * mu2 * X2

    def rollout(self, x0, Ds, dt=1.0):
        """States and methane along a feed sequence."""
        x = np.asarray(x0, dtype=float)
        S2, CH4 = [], []
        for D in Ds:
            CH4.append(self.ch4(x, D))
            x = self.step(x, D, dt)
            S2.append(x[3])
        return np.array(S2), np.array(CH4), x
