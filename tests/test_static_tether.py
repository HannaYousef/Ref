"""Quasi-steady tether: regression against the MATLAB reference of Tethers.jl and
physical sanity checks."""

import math
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import brentq

from softkite.static_tether import (StaticSettings, StaticTether, matlab_to_wind, tether_shape,
                                    wind_to_matlab)

DATA = Path(__file__).parent / "data"


def load_fixture():
    v = sio.loadmat(DATA / "input_basic_test.mat", squeeze_me=True, struct_as_record=False)
    r = sio.loadmat(DATA / "basic_test_results.mat", squeeze_me=True, struct_as_record=False)
    T, E = v["T"], v["ENVMT"]
    wind = np.asarray(v["windVel"], float)
    se = StaticSettings(rho=E.rhos, g_earth=abs(E.g), cd_tether=T.CD_tether, d_tether=T.d_tether * 1000,
                        rho_tether=T.rho_t / T.A, c_spring=T.E * T.A, segments=wind.shape[1] + 1)
    b, f = matlab_to_wind(v["stateVec"][0], v["stateVec"][1])
    return v, r, se, wind, (b, f, float(v["stateVec"][2]))


def test_matlab_reference():
    v, r, se, wind, (b, f, Tn) = load_fixture()
    kp = np.asarray(v["kitePos"], float)
    kv = np.asarray(v["kiteVel"], float)
    pos = np.zeros((3, wind.shape[1]))
    out = tether_shape(b, f, Tn, kp, kv, wind, float(v["tetherLength"]), se.segments, se.rho, se.g_earth,
                       se.cd_tether, se.d_tether, se.rho_tether, se.c_spring, pos, True)
    np.testing.assert_allclose(out[0:3], r["Fobj"], rtol=1e-9)
    np.testing.assert_allclose(out[3:6], r["T0"], rtol=1e-9)
    np.testing.assert_allclose(out[6:9], r["p0"], rtol=1e-9)
    np.testing.assert_allclose(pos, r["pj"], rtol=1e-9, atol=1e-9)


def test_solver_converges_on_fixture():
    v, r, se, wind, (b, f, Tn) = load_fixture()
    te = StaticTether(se)
    te.wind_vel[:] = wind
    te.state_vec[:] = (b, f, Tn)
    te.step(np.asarray(v["kitePos"], float), np.asarray(v["kiteVel"], float),
            tether_length=float(v["tetherLength"]))
    assert te.converged
    assert np.linalg.norm(te.p0 - te.kite_pos) < se.abs_tol + se.rel_tol * te.tether_length


def test_angle_conventions_roundtrip():
    for b, f in [(0.3, 0.2), (1.2, -0.7), (0.05, 2.0)]:
        bb, ff = matlab_to_wind(*wind_to_matlab(b, f))
        assert abs(bb - b) < 1e-12 and abs(ff - f) < 1e-12


def test_static_catenary():
    """No wind, kite at rest: the lumped-mass tether converges to the analytic catenary
    as the number of segments grows (inextensible tether)."""
    kp = np.array([100.0, 100.0, 800.0])
    L = 1.05 * np.linalg.norm(kp)
    h, vz = math.hypot(kp[0], kp[1]), kp[2]
    c = brentq(lambda c: 2 * math.sinh(c * h / 2) / c - math.sqrt(L * L - vz * vz), 1e-9, 10.0)
    x_min = -0.5 * (math.log((L + vz) / (L - vz)) / c - h)
    bias = -math.cosh(-x_min * c) / c
    errors = []
    for n in (20, 40, 160):
        se = StaticSettings(segments=n, l_tether=L, c_spring=1e12)
        te = StaticTether(se).init(kite_pos=kp, tether_length=L)
        assert te.converged
        nodes = te.nodes()
        x = np.hypot(nodes[0], nodes[1])
        z_ana = np.cosh((x - x_min) * c) / c + bias
        errors.append(np.max(np.abs(nodes[2] - z_ana)))
    sag = np.max(np.abs(vz * x / h - z_ana))  # deviation of the catenary from the chord
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.005 * sag


def test_vertical_force_balance():
    """At rest without wind: kite-end force minus ground force = tether weight."""
    se = StaticSettings(segments=10)
    kp = np.array([150.0, 0.0, 200.0])
    L = 1.001 * np.linalg.norm(kp)
    te = StaticTether(se).init(kite_pos=kp, tether_length=L)
    b, f, Tn = te.state_vec
    f_gnd = Tn * np.array([math.cos(b) * math.cos(f), math.cos(b) * math.sin(f), math.sin(b)])
    weight = se.rho_tether * math.pi / 4 * (se.d_tether / 1000) ** 2 * L * se.g_earth
    diff = te.force_kite - f_gnd
    assert abs(diff[2] - weight) < 1e-6 * Tn + 1e-6
    assert abs(diff[0]) < 1e-6 * Tn and abs(diff[1]) < 1e-6 * Tn


def test_drag_increases_force():
    se = StaticSettings(segments=10)
    kp = np.array([200.0, 0.0, 150.0])
    L = np.linalg.norm(kp)
    te = StaticTether(se).init(kite_pos=kp, tether_length=L)
    f0 = np.linalg.norm(te.force_kite)
    wind = np.zeros((3, 9))
    wind[0] = 10.0
    te.step(kp, np.array([0.0, 30.0, 0.0]), tether_length=L, wind_vel=wind)
    assert te.converged
    assert np.linalg.norm(te.force_kite) > f0
