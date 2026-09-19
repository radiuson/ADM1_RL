"""Conventional controllers, for the comparison a process audience will ask for.

Each family is swept over its own tuning parameter so that it traces a
production-constraint frontier rather than a single point, which is what makes
it comparable with the reward-parameter sweep.

  Constant     open-loop feed at a fixed rate - what most plants run
  Rule-based   the operator heuristic: cut feed when VFA is high, raise it when
               low, in fixed steps (cf. the sour-digester response in operator
               training material: reduce or stop feeding until recovery)
  PI           feedback on VFA to a setpoint held below the limit

All three act on the same six plant-measurable signals the learned policies see
and command the same two inputs, feed rate and feed strength, so the comparison
isolates the control law.  Holding the strength at its nominal value while the
learned policy may raise it would cap the conventional controllers at 77 % of
the organic load the learned policy can reach, which is a handicap rather than
a difference in control.
"""
import numpy as np


class ConstantFeed:
    def __init__(self, q, mult=1.0):
        self.q, self.mult = float(q), float(mult)
        self.name = f'constant_q{q:g}'

    def reset(self): pass

    def act(self, vfa_mgL, alk, pH, q_ch4, q_prev, mult_prev):
        return self.q, self.mult


class RuleBased:
    """Step the feed down when VFA is above the action level, up when below."""

    def __init__(self, act_level_mgL, q_lo, q_hi, step=0.12, mult=1.3):
        self.lvl = float(act_level_mgL)
        self.q_lo, self.q_hi = float(q_lo), float(q_hi)
        self.step = float(step)
        self.mult = float(mult)
        self.name = f'rule_{act_level_mgL:g}_m{mult:g}'
        self.q = None

    def reset(self):
        self.q = self.q_hi

    def act(self, vfa_mgL, alk, pH, q_ch4, q_prev, mult_prev):
        if self.q is None:
            self.q = self.q_hi
        if vfa_mgL > self.lvl:
            self.q *= (1.0 - self.step)
        elif vfa_mgL < 0.7 * self.lvl:
            self.q *= (1.0 + self.step)
        self.q = float(np.clip(self.q, self.q_lo, self.q_hi))
        return self.q, self.mult


class PIFeed:
    """PI control of VFA by feed rate.

    The process gain is about 3.7 mg/L per m3/d at the reference loading
    (98 mg/L at 94 m3/d rising to 336 mg/L at 159), so a proportional gain of
    order 0.1 m3/d per mg/L is a fraction of the inverse gain.  The integral
    time is set on the scale of the hydraulic retention time, since that is how
    long the VFA pool takes to respond.
    """

    def __init__(self, setpoint_mgL, q_lo, q_hi, Kc=0.10, tau_i=20.0, mult=1.3):
        self.sp = float(setpoint_mgL)
        self.q_lo, self.q_hi = float(q_lo), float(q_hi)
        self.Kc, self.tau_i = float(Kc), float(tau_i)
        self.mult = float(mult)
        self.name = f'pi_sp{setpoint_mgL:g}_m{mult:g}'
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def act(self, vfa_mgL, alk, pH, q_ch4, q_prev, mult_prev):
        err = self.sp - vfa_mgL                       # positive => room to feed
        q_un = self.q_hi + self.Kc * (err + self.integral / self.tau_i)
        q = float(np.clip(q_un, self.q_lo, self.q_hi))
        if q_un == q:                                  # anti-windup
            self.integral += err
        return q, self.mult
