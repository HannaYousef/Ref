import math

import numpy as np

from softkite import KiteModel, load_settings


def test_init_and_step_finite():
    m = KiteModel(load_settings()).init()
    for _ in range(250):
        m.step(0.0, 0.4, 0.0)
    assert not m.crashed
    assert np.all(np.isfinite(m.y))
    assert m.out["tether_ok"]


def test_open_loop_turn_rate_matches_law():
    """With constant steering the heading rate follows c1 * v_a * u_s (+ gravity term)."""
    s = load_settings(kcu__use_delay=False)
    m = KiteModel(s).init(elevation=30.0, heading=90.0, depower=0.25)
    for _ in range(50):  # settle
        m.step(0.0, 0.25, 0.0)
    for _ in range(20):
        m.step(0.05, 0.25, 0.0)
    o = m.out
    law = o["c1"] * o["v_a"] * o["u_s_lagged"] + o["c2"] / o["v_a"] * math.sin(o["heading"]) * math.cos(o["beta"])
    assert abs(o["psi_dot"] - law) < 1e-9


def test_course_follows_heading():
    """The side force aligns the flight direction with the heading (crab angle from wind only)."""
    s = load_settings()
    m = KiteModel(s).init(elevation=35.0, heading=90.0, depower=0.3)
    for _ in range(400):
        m.step(0.0, 0.3, 0.0)
    assert abs(math.degrees(m.out["side_slip"])) < 3.0


def test_winch_speed_tracking():
    m = KiteModel(load_settings()).init(depower=0.3, elevation=40.0)
    for _ in range(200):
        m.step(0.0, 0.3, 1.0)
    assert abs(m.out["v_reel"] - 1.0) < 0.05


def test_flight2013_settings_and_log_profile():
    from softkite.settings import DATA

    s = load_settings(DATA / "settings_v3_flight2013.yaml")
    assert s.control.ro_mode == "force" and s.environment.profile_law == "log"
    m = KiteModel(s)
    # wind at 139 m for 5.9 m/s at 6 m and z0 = 0.07 m (van der Vlugt et al. 2019: 10.1 m/s)
    assert abs(m.wind_speed(139.0) - 10.07) < 0.05


def test_constant_force_reel_out():
    from softkite.controllers.winch import WinchController

    w = WinchController(k_v=0.05, f_ro=3000.0, dt=0.02)
    w.set_mode("reel_out_force")
    v1 = w.update(3500.0)
    assert v1 > 0  # force above set-point -> reel out faster
