"""Flight controllers built from the course loop, the figure-of-eight guidance and the
winch set-point generator.

* :class:`ParkingController` - hold the kite near zenith at constant tether length.
* :class:`Fig8Controller` - fly figures of eight at constant tether length.
* :class:`PumpingCycleController` - full pumping cycle::

      PARK -> REEL_OUT_ENTRY -> REEL_OUT -> RETRACT -> REEL_IN -> REEL_OUT_ENTRY -> ...

  REEL_OUT_ENTRY  power up (depower -> u_p_ro), dive into the eight, winch on the
                  reel-out law;  ends when the cross-track error is small.
  REEL_OUT        fly the eight while reeling out; ends at ``l_max`` as soon as the
                  kite flies upwards (at most ``ro_exit_wait`` s later).
  RETRACT         partly depower (-> u_p_tr), stop the winch and steer towards the
                  top of the wind window (azimuth 0, ``ri_elevation``); ends when the
                  elevation exceeds ``tr_elevation`` or after ``tr_time_max``.
  REEL_IN         depower (-> u_p_ri), park the kite (heading up, azimuth hold) and
                  reel in at ``v_ri`` with the tension limited to ``f_ri``; ends at
                  ``l_min``.  Above ``ri_beta_max - ri_beta_band`` the reel-in speed
                  is reduced linearly (to 20 % at ``ri_beta_max``), which keeps the
                  depowered kite from flying over the zenith.

All controllers take the observation dict ``KiteModel.out`` and return
``(u_s_set, u_p_set, v_set)``.
"""

from __future__ import annotations

import math

from .course import CourseController, wrap_pi
from .figure_eight import FigureEight
from .winch import WinchController, tune_k_v


PARK, REEL_OUT_ENTRY, REEL_OUT, RETRACT, REEL_IN = "park", "reel_out_entry", "reel_out", "retract", "reel_in"
PHASES = [PARK, REEL_OUT_ENTRY, REEL_OUT, RETRACT, REEL_IN]


def _course_ctrl(s):
    c = s.control
    return CourseController(k_psi=c.k_psi, k_psi_d=c.k_psi_d, v_heading=c.v_heading,
                            v_course=c.v_course, max_steering=s.kcu.max_steering)


def _fig8(s):
    c = s.control
    return FigureEight(azimuth_width=c.f8_azimuth, height=c.f8_height, elevation=c.f8_elevation,
                       lookahead=c.f8_lookahead, up_turns=c.f8_up_turns)


def _winch(s, model):
    c, w = s.control, s.winch
    k_v = tune_k_v(model.aero, c.u_p_ro, rho=s.environment.rho, f=c.f_ro_factor)
    return WinchController(k_v=k_v, f_max=c.f_max, f_min=c.f_min, f_ri=c.f_ri, v_ri=c.v_ri, kp=c.kp_force,
                           ki=c.ki_force, v_ro_max=w.v_ro_max, v_ri_max=w.v_ri_max, a_max=w.a_max,
                           dt=s.simulation.dt, f_ro=c.f_ro)


def park_heading(obs, k_azimuth, max_heading=math.radians(35)):
    """Heading set-point that drives the azimuth to zero (azimuth positive to the left,
    heading positive to the right)."""
    return max(-max_heading, min(max_heading, k_azimuth * obs["phi"]))


class ParkingController:
    """Hold the kite near zenith at constant tether length (heading up, azimuth hold)."""

    def __init__(self, settings, model, depower=None):
        self.s = settings
        self.course = _course_ctrl(settings)
        self.winch = _winch(settings, model)
        self.u_p = settings.control.u_p_ri if depower is None else depower
        self.phase = PARK
        self.cycle = 0
        self.chi_set = 0.0

    def update(self, obs):
        dt = self.s.simulation.dt
        self.chi_set = park_heading(obs, self.s.control.k_azimuth)
        u_s = self.course.steering(self.chi_set, obs, dt)
        self.winch.set_mode("speed", 0.0)
        return u_s, self.u_p, self.winch.update(obs["f_gnd"])


class Fig8Controller:
    """Figures of eight at constant tether length (no pumping).

    ``hold_length=True`` keeps the length fixed whatever the force (winch mode
    ``hold``); otherwise the upper force limiter may reel out above ``f_max``.
    """

    def __init__(self, settings, model, depower=None, hold_length=True):
        self.s = settings
        self.course = _course_ctrl(settings)
        self.fig8 = _fig8(settings)
        self.winch = _winch(settings, model)
        self.u_p = settings.control.u_p_f8 if depower is None else depower
        self.hold_length = hold_length
        self.phase = REEL_OUT_ENTRY
        self.cycle = 0
        self.chi_set = 0.0
        self.t_phase = 0.0

    def update(self, obs):
        dt = self.s.simulation.dt
        self.t_phase += dt
        if self.t_phase < 3.0:  # let the kite settle before diving into the pattern
            self.chi_set = park_heading(obs, self.s.control.k_azimuth)
            ff = 0.0
            u_p = self.s.control.u_p_ri
        else:
            self.chi_set, ff = self.fig8.guidance(obs)
            u_p = self.u_p
            if self.phase == REEL_OUT_ENTRY and self.fig8.cross_track < math.radians(3.0):
                self.phase = REEL_OUT
        u_s = self.course.steering(self.chi_set, obs, dt, ff)
        self.winch.set_mode("hold" if self.hold_length else "speed", 0.0)
        return u_s, u_p, self.winch.update(obs["f_gnd"])


class PumpingCycleController:
    """Full pumping-cycle controller (see module docstring)."""

    def __init__(self, settings, model, park_time=3.0):
        self.s = settings
        self.course = _course_ctrl(settings)
        self.fig8 = _fig8(settings)
        self.winch = _winch(settings, model)
        self.park_time = park_time
        self.phase = PARK
        self.cycle = 0
        self.t_phase = 0.0
        self.chi_set = 0.0
        self.u_p = settings.control.u_p_ri
        self.transitions = []  # (time, phase)

    def _enter(self, phase, obs):
        self.phase = phase
        self.t_phase = 0.0
        self.transitions.append((obs["time"], phase))
        if phase == REEL_OUT_ENTRY:
            self.cycle += 1
            self.fig8.s = None
            self.winch.set_mode("reel_out_force" if self.s.control.ro_mode == "force" else "reel_out")
        elif phase == RETRACT:
            self.winch.set_mode("speed", 0.0)
        elif phase == REEL_IN:
            self.winch.set_mode("reel_in")

    def update(self, obs):
        c = self.s.control
        dt = self.s.simulation.dt
        self.t_phase += dt
        ff = 0.0
        ph = self.phase

        if ph == PARK:
            self.u_p = c.u_p_ri
            self.chi_set = park_heading(obs, c.k_azimuth)
            self.winch.set_mode("speed", 0.0)
            if self.t_phase >= self.park_time:
                self._enter(REEL_OUT_ENTRY, obs)

        elif ph in (REEL_OUT_ENTRY, REEL_OUT):
            self.u_p = c.u_p_ro
            self.chi_set, ff = self.fig8.guidance(obs)
            if ph == REEL_OUT_ENTRY and self.fig8.cross_track < math.radians(3.0) and self.t_phase > 2.0:
                self._enter(REEL_OUT, obs)
            elif ph == REEL_OUT and obs["l_tether"] >= c.l_max:
                # leave the pattern while the kite climbs
                self.t_wait = getattr(self, "t_wait", 0.0) + dt
                if abs(wrap_pi(obs["course"])) < math.radians(90) or self.t_wait > c.ro_exit_wait:
                    self.t_wait = 0.0
                    self._enter(RETRACT, obs)

        elif ph == RETRACT:
            self.u_p = c.u_p_tr
            # steer towards the top of the wind window
            d_up = math.radians(c.ri_elevation) - obs["beta"]
            d_right = obs["phi"] * math.cos(obs["beta"])
            self.chi_set = math.atan2(d_right, max(d_up, 0.05))
            if obs["beta"] > math.radians(c.tr_elevation) or self.t_phase > c.tr_time_max:
                self._enter(REEL_IN, obs)

        elif ph == REEL_IN:
            self.u_p = c.u_p_ri
            self.chi_set = park_heading(obs, c.k_azimuth)
            # keep the kite away from the zenith: reel in slower when it climbs too high
            over = math.degrees(obs["beta"]) - (c.ri_beta_max - c.ri_beta_band)
            self.winch.v_ri = c.v_ri * min(1.0, max(0.2, 1.0 - over / c.ri_beta_band))
            if obs["l_tether"] <= c.l_min:
                self._enter(REEL_OUT_ENTRY, obs)

        u_s = self.course.steering(self.chi_set, obs, dt, ff)
        v_set = self.winch.update(obs["f_gnd"])
        return u_s, self.u_p, v_set
