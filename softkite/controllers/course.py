"""Heading / course controller (the inner steering loop).

The plant is the identified turn-rate law

    psi_dot = c1 * v_a * u_s + c2 / v_a * sin(psi) * cos(beta)

so the controller inverts it: a desired turn rate ``r_des`` (proportional to the angle
error plus an optional feed-forward from the path curvature) is turned into the
steering command

    u_s = (r_des - c2 / v_a * sin(psi) * cos(beta)) / (c1 * v_a).

This is the gain scheduling with 1/(c1 v_a) that RobustKiteControllers.jl uses, plus
an explicit cancellation of the gravity turn.  At low tangential speed the course is
undefined, so the error is taken from the heading; above ``v_course`` from the course;
in between the two errors are blended linearly (as in RobustKiteControllers.jl).
"""

from __future__ import annotations

import math


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class CourseController:
    def __init__(self, k_psi=2.5, k_psi_d=0.0, v_heading=5.0, v_course=10.0, max_steering=0.32):
        self.k_psi = k_psi
        self.k_psi_d = k_psi_d
        self.v_heading = v_heading
        self.v_course = v_course
        self.max_steering = max_steering
        self.last_err = None
        self.err = 0.0
        self.r_des = 0.0

    def blend(self, v_t: float) -> float:
        """Weight of the course feedback (0 = heading only, 1 = course only)."""
        if v_t <= self.v_heading:
            return 0.0
        if v_t >= self.v_course:
            return 1.0
        return (v_t - self.v_heading) / (self.v_course - self.v_heading)

    def steering(self, chi_set: float, obs: dict, dt: float, rate_ff: float = 0.0) -> float:
        """Steering command for a commanded course/heading ``chi_set`` [rad]."""
        w = self.blend(obs["v_tangential"])
        e_psi = wrap_pi(chi_set - obs["heading"])
        e_chi = wrap_pi(chi_set - obs["course"])
        # blend on the unit circle so that +-pi wrap-around does not matter
        err = math.atan2((1 - w) * math.sin(e_psi) + w * math.sin(e_chi),
                         (1 - w) * math.cos(e_psi) + w * math.cos(e_chi))
        d_err = 0.0 if self.last_err is None else wrap_pi(err - self.last_err) / dt
        self.last_err = err
        self.err = err
        r_des = self.k_psi * err + self.k_psi_d * d_err + rate_ff
        self.r_des = r_des
        va = max(obs["v_a"], 3.0)
        grav = obs["c2"] / va * math.sin(obs["heading"]) * math.cos(obs["beta"])
        u_s = (r_des - grav) / (obs["c1"] * va)
        return max(-self.max_steering, min(self.max_steering, u_s))
