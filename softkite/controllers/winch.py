"""Winch set-point generation (reel speed), in the spirit of WinchControllers.jl.

Modes
-----
``reel_out``  ``v = k_v * sqrt(F)`` (optimal for a quasi-steady crosswind kite when
              ``k_v`` is tuned for a reel-out factor of 1/3), with an upper force
              limiter at ``f_max`` and a lower one at ``f_min`` (both PI, anti-windup).
``reel_out_force``  constant-force reel-out at ``f_ro`` (PI on the force error), as
              flown in the 2012/2013 TU Delft test flights; lower limiter active.
``reel_in``   reel in at ``v_ri`` with an upper force limiter at ``f_ri`` (slows the
              reel-in down) and the lower limiter at ``f_min`` (speeds it up).
``speed``     follow a given speed (e.g. 0 during a transition) with the upper force
              limiter active.
``hold``      zero speed, no force limiter (constant tether length; only the motor
              force limit of the winch model still applies).

The set-point is acceleration-limited (``a_max``) for smooth mode switches.
"""

from __future__ import annotations

import math


class WinchController:
    def __init__(self, k_v, f_max=4000.0, f_min=300.0, f_ri=1200.0, v_ri=7.0, kp=0.002, ki=0.004,
                 v_ro_max=8.0, v_ri_max=8.0, a_max=4.0, dt=0.02, f_ro=3000.0):
        self.k_v = k_v
        self.f_max, self.f_min, self.f_ri = f_max, f_min, f_ri
        self.v_ri = v_ri
        self.f_ro = f_ro
        self.i_fc = 0.0  # integrator of the constant-force reel-out controller
        self.kp, self.ki = kp, ki
        self.v_ro_max, self.v_ri_max = v_ro_max, v_ri_max
        self.a_max = a_max
        self.dt = dt
        self.mode = "speed"
        self.v_target = 0.0
        self.v_set = 0.0
        self.i_up = 0.0  # integrator of the upper force limiter (>= 0)
        self.i_lo = 0.0  # integrator of the lower force limiter (<= 0)

    def set_mode(self, mode: str, v_target: float = 0.0):
        if mode != self.mode:
            self.i_lo = 0.0
            if mode == "reel_out_force":
                self.i_fc = self.v_set
            if mode == "reel_in":
                self.i_up = 0.0
        self.mode = mode
        self.v_target = v_target

    def _upper_limiter(self, force, f_lim):
        e = force - f_lim
        self.i_up = max(0.0, self.i_up + self.ki * e * self.dt)
        return max(0.0, self.kp * e + self.i_up)

    def _lower_limiter(self, force):
        e = force - self.f_min
        self.i_lo = min(0.0, self.i_lo + self.ki * e * self.dt)
        return min(0.0, self.kp * e + self.i_lo)

    def update(self, force: float) -> float:
        dt = self.dt
        if self.mode == "reel_out":
            v = self.k_v * math.sqrt(max(force, 0.0))
            v += self._upper_limiter(force, self.f_max) + self._lower_limiter(force)
        elif self.mode == "reel_out_force":
            e = force - self.f_ro
            self.i_fc = min(max(self.i_fc + self.ki * e * dt, -self.v_ri_max), self.v_ro_max)
            v = self.kp * e + self.i_fc + self._lower_limiter(force)
        elif self.mode == "reel_in":
            v = -self.v_ri + self._upper_limiter(force, self.f_ri) + self._lower_limiter(force)
        elif self.mode == "hold":
            v = 0.0
        else:  # speed
            v = self.v_target + self._upper_limiter(force, self.f_max)
        v = min(max(v, -self.v_ri_max), self.v_ro_max)
        dv = self.a_max * dt
        self.v_set += min(max(v - self.v_set, -dv), dv)
        return self.v_set


def tune_k_v(aero, u_p, rho=1.225, f=1 / 3, tether_factor=0.85):
    """Gain of the ``v = k_v sqrt(F)`` law for reel-out factor ``f``.

    Quasi-steady crosswind theory: ``F ~ 0.5 rho S C_R (1 + E^2) (v_w,r - v_ro)^2``
    with ``C_R = sqrt(CL^2 + CD^2)`` and the effective glide ratio ``E`` (kite glide
    ratio reduced by ``tether_factor`` for tether drag).  With ``v_ro = f v_w,r`` this
    gives ``k_v = f / ((1 - f) K)``, ``K = sqrt(0.5 rho S C_R (1 + E^2))``.
    """
    a = aero.alpha_trim(u_p)
    cl, cd = aero.cl(a), aero.cd(a)
    e = tether_factor * cl / cd
    K = math.sqrt(0.5 * rho * aero.area * math.hypot(cl, cd) * (1 + e * e))
    return f / ((1 - f) * K)
