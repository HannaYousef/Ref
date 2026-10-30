"""Quasi-steady tether model ("StaticTether").

Python port of ``Tethers.QuasiSteady`` from Tethers.jl (Fechner, Bertozzi, van de Lint,
MIT licence), which implements the quasi-steady cable model of Williams (2017,
J. Guidance, Control, and Dynamics 40(7)) and was ported from a MATLAB implementation.

Given the kite position and velocity, the unstretched tether length and the wind at
each tether node, the model finds the tether elevation ``beta``, azimuth ``phi`` and
tension ``Tn`` at the ground station such that the tether, integrated segment by
segment from the ground station upwards, ends exactly at the kite.  Each node carries
the mass of one segment, its drag, gravity and the centripetal acceleration of a tether
that rotates rigidly with the kite.  Segments stretch elastically with ``EA``.

Conventions (identical to Tethers.jl):

* W frame (wind frame): x downwind, z up, y completes a right-handed frame.
* ``beta``: elevation, 0 at the horizon, 90 deg at zenith.
* ``phi``: azimuth, positive anti-clockwise seen from above.
* ``dir = [cos(beta) cos(phi), cos(beta) sin(phi), sin(beta)]``.
* ``force_kite`` (``T0`` in the Julia code) is the force the kite exerts on the end of
  the tether.  The force of the tether on the kite is ``-force_kite``.

The core routine ``tether_shape`` is written with scalar ``math`` only so that it runs
fast in plain CPython and can be JIT-compiled by numba if numba is installed
(optional, ~15x faster).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:  # optional acceleration
    from numba import njit as _njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba missing
    HAVE_NUMBA = False

    def _njit(*args, **kwargs):  # type: ignore
        if args and callable(args[0]):
            return args[0]

        def deco(f):
            return f

        return deco


__all__ = [
    "StaticSettings",
    "StaticTether",
    "tether_shape",
    "n_nodes",
    "matlab_to_wind",
    "wind_to_matlab",
    "HAVE_NUMBA",
]


def n_nodes(segments: int) -> int:
    """Number of stored tether nodes (the ground station and the kite end are not stored)."""
    return segments - 1


@dataclass
class StaticSettings:
    """Physical, structural and solver parameters (defaults as in Tethers.jl)."""

    segments: int = 8  #: number of tether segments
    elevation: float = 70.0  #: initial elevation [deg]
    azimuth: float = 0.0  #: initial azimuth [deg]
    l_tether: float = 50.0  #: initial unstretched tether length [m]
    slack: float = 0.05  #: initial slack, l_tether = (1 + slack) * kite distance
    rho: float = 1.225  #: air density [kg/m^3]
    g_earth: float = 9.81  #: gravitational acceleration [m/s^2] (acts in -z)
    cd_tether: float = 0.958  #: drag coefficient of the tether
    d_tether: float = 4.0  #: tether diameter [mm]
    rho_tether: float = 724.0  #: tether material density [kg/m^3]
    c_spring: float = 614600.0  #: unit spring constant E*A [N]
    abs_tol: float = 1e-4  #: absolute tolerance on the end-point gap [m]
    rel_tol: float = 1e-4  #: relative tolerance (times tether length)
    max_iter: int = 60  #: maximum Newton iterations


# ---------------------------------------------------------------------------------------
# Core physics (scalar, numba friendly)
# ---------------------------------------------------------------------------------------


@_njit(cache=True)
def _segment_drag(vax, vay, vaz, dx, dy, dz, drag_coeff):
    """Drag on a segment of unit direction d with apparent velocity va (node vel - wind)."""
    if abs(vax) < 1e-3 and abs(vay) < 1e-3 and abs(vaz) < 1e-3:
        return 0.0, 0.0, 0.0
    p = vax * dx + vay * dy + vaz * dz
    nx = vax - p * dx
    ny = vay - p * dy
    nz = vaz - p * dz
    n2 = nx * nx + ny * ny + nz * nz
    if n2 < 1e-24:
        return 0.0, 0.0, 0.0
    f = drag_coeff * math.sqrt(n2)
    return f * nx, f * ny, f * nz


@_njit(cache=True)
def tether_shape(beta, phi, Tn, kp, kv, wind, tether_length, segments, rho, g, cd_t, d_mm,
                 rho_t, EA, pos_out, store):
    """Integrate the quasi-steady tether from the ground station to the kite.

    Parameters
    ----------
    beta, phi, Tn : elevation [rad], azimuth [rad] and tension [N] at the ground station
    kp, kv : (3,) kite position and velocity, W frame
    wind : (3, segments-1) wind velocity at each stored node
    pos_out : (3, segments-1) buffer that receives the node positions if ``store``

    Returns
    -------
    rx, ry, rz : gap ``kite_pos - p0`` [m]
    T0x, T0y, T0z : force from the kite on the end of the tether [N]
    p0x, p0y, p0z : tether end point [m]
    """
    nn = segments - 1
    Ls = tether_length / segments
    d = d_mm / 1000.0
    drag_coeff = -0.5 * rho * Ls * d * cd_t
    A = math.pi / 4.0 * d * d
    mj = rho_t * Ls * A

    sb = math.sin(beta)
    cb = math.cos(beta)
    sp = math.sin(phi)
    cp = math.cos(phi)

    norm_p = math.sqrt(kp[0] * kp[0] + kp[1] * kp[1] + kp[2] * kp[2])
    ux = kp[0] / norm_p
    uy = kp[1] / norm_p
    uz = kp[2] / norm_p
    v_par = kv[0] * ux + kv[1] * uy + kv[2] * uz
    # omega = (kp / |kp|^2) x kv
    n2 = norm_p * norm_p
    ax_ = kp[0] / n2
    ay_ = kp[1] / n2
    az_ = kp[2] / n2
    wx = ay_ * kv[2] - az_ * kv[1]
    wy = az_ * kv[0] - ax_ * kv[2]
    wz = ax_ * kv[1] - ay_ * kv[0]

    # first segment, leaving the ground station
    dx = cb * cp
    dy = cb * sp
    dz = sb
    FTx = Tn * dx
    FTy = Tn * dy
    FTz = Tn * dz
    px = Ls * dx
    py = Ls * dy
    pz = Ls * dz
    if store:
        pos_out[0, nn - 1] = px
        pos_out[1, nn - 1] = py
        pos_out[2, nn - 1] = pz
    # node kinematics: vel = v_par*u + w x p ; acc = w x (w x p)
    cx = wy * pz - wz * py
    cy = wz * px - wx * pz
    cz = wx * py - wy * px
    vx = v_par * ux + cx
    vy = v_par * uy + cy
    vz = v_par * uz + cz
    acx = wy * cz - wz * cy
    acy = wz * cx - wx * cz
    acz = wx * cy - wy * cx
    Fdx, Fdy, Fdz = _segment_drag(vx - wind[0, nn - 1], vy - wind[1, nn - 1], vz - wind[2, nn - 1],
                                  dx, dy, dz, drag_coeff)

    for ii in range(nn, 1, -1):  # ii = nn ... 2 (1-based node index as in Julia)
        m_tot = 1.5 * mj if ii == nn else mj
        FTx = m_tot * acx + FTx - Fdx
        FTy = m_tot * acy + FTy - Fdy
        FTz = m_tot * acz + FTz - Fdz + m_tot * g
        ftn = math.sqrt(FTx * FTx + FTy * FTy + FTz * FTz)
        l_i = (ftn / EA + 1.0) * Ls
        nx_ = px + l_i * FTx / ftn
        ny_ = py + l_i * FTy / ftn
        nz_ = pz + l_i * FTz / ftn
        # drag of this segment uses the velocity of the node below it
        vax = vx - wind[0, ii - 1]
        vay = vy - wind[1, ii - 1]
        vaz = vz - wind[2, ii - 1]
        cx = wy * nz_ - wz * ny_
        cy = wz * nx_ - wx * nz_
        cz = wx * ny_ - wy * nx_
        vx = v_par * ux + cx
        vy = v_par * uy + cy
        vz = v_par * uz + cz
        acx = wy * cz - wz * cy
        acy = wz * cx - wx * cz
        acz = wx * cy - wy * cx
        sx = nx_ - px
        sy = ny_ - py
        sz = nz_ - pz
        sn = math.sqrt(sx * sx + sy * sy + sz * sz)
        Fdx, Fdy, Fdz = _segment_drag(vax, vay, vaz, sx / sn, sy / sn, sz / sn, drag_coeff)
        px = nx_
        py = ny_
        pz = nz_
        if store:
            pos_out[0, ii - 2] = px
            pos_out[1, ii - 2] = py
            pos_out[2, ii - 2] = pz

    # last segment up to the kite attachment point
    T0x = 1.5 * mj * acx + FTx - Fdx
    T0y = 1.5 * mj * acy + FTy - Fdy
    T0z = 1.5 * mj * acz + FTz - Fdz + 1.5 * mj * g
    t0n = math.sqrt(T0x * T0x + T0y * T0y + T0z * T0z)
    l_i = (t0n / EA + 1.0) * Ls
    p0x = px + l_i * T0x / t0n
    p0y = py + l_i * T0y / t0n
    p0z = pz + l_i * T0z / t0n
    return (kp[0] - p0x, kp[1] - p0y, kp[2] - p0z, T0x, T0y, T0z, p0x, p0y, p0z)


@_njit(cache=True)
def _solve3(J, r):
    """Solve the 3x3 system J x = r by Cramer's rule (returns (ok, x0, x1, x2))."""
    a, b, c = J[0, 0], J[0, 1], J[0, 2]
    d, e, f = J[1, 0], J[1, 1], J[1, 2]
    g, h, i = J[2, 0], J[2, 1], J[2, 2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-300:
        return False, 0.0, 0.0, 0.0
    x0 = (r[0] * (e * i - f * h) - b * (r[1] * i - f * r[2]) + c * (r[1] * h - e * r[2])) / det
    x1 = (a * (r[1] * i - f * r[2]) - r[0] * (d * i - f * g) + c * (d * r[2] - r[1] * g)) / det
    x2 = (a * (e * r[2] - r[1] * h) - b * (d * r[2] - r[1] * g) + r[0] * (d * h - e * g)) / det
    return True, x0, x1, x2


@_njit(cache=True)
def _newton(beta, phi, Tn, kp, kv, wind, L, segments, rho, g, cd_t, d_mm, rho_t, EA, tol,
            max_iter, pos_out):
    """Damped Newton solve in (beta, phi, log(Tn/scale)), as in Tethers.jl.

    The tension is solved on a log scale: it spans decades and the residual is far less
    sensitive to it than to the two angles.  Steps are limited to a radius of 2 in the
    scaled unknowns and backtracked if they do not reduce the residual.
    Returns (converged, beta, phi, Tn, iterations, |res|).
    """
    scale = Tn if Tn > 0.0 else 2e-4 * EA
    u0 = beta
    u1 = phi
    u2 = math.log(Tn / scale) if Tn > 0.0 else 0.0
    J = np.empty((3, 3))
    r = np.empty(3)
    dummy = pos_out
    res = tether_shape(u0, u1, scale * math.exp(u2), kp, kv, wind, L, segments, rho, g, cd_t,
                       d_mm, rho_t, EA, dummy, False)
    nrm = math.sqrt(res[0] ** 2 + res[1] ** 2 + res[2] ** 2)
    it = 0
    h = 1e-7
    while nrm > tol and it < max_iter:
        it += 1
        r[0] = res[0]
        r[1] = res[1]
        r[2] = res[2]
        for k in range(3):
            v0, v1, v2 = u0, u1, u2
            if k == 0:
                v0 += h
            elif k == 1:
                v1 += h
            else:
                v2 += h
            rp = tether_shape(v0, v1, scale * math.exp(v2), kp, kv, wind, L, segments, rho, g,
                              cd_t, d_mm, rho_t, EA, dummy, False)
            J[0, k] = (rp[0] - r[0]) / h
            J[1, k] = (rp[1] - r[1]) / h
            J[2, k] = (rp[2] - r[2]) / h
        ok, s0, s1, s2 = _solve3(J, r)
        if not ok:
            break
        s0, s1, s2 = -s0, -s1, -s2
        sn = math.sqrt(s0 * s0 + s1 * s1 + s2 * s2)
        if sn > 2.0:
            s0 *= 2.0 / sn
            s1 *= 2.0 / sn
            s2 *= 2.0 / sn
        lam = 1.0
        accepted = False
        for _ in range(30):
            w0 = u0 + lam * s0
            w1 = u1 + lam * s1
            w2 = u2 + lam * s2
            rn = tether_shape(w0, w1, scale * math.exp(w2), kp, kv, wind, L, segments, rho, g,
                              cd_t, d_mm, rho_t, EA, dummy, False)
            nn_ = math.sqrt(rn[0] ** 2 + rn[1] ** 2 + rn[2] ** 2)
            if nn_ < nrm or lam < 1e-6:
                u0, u1, u2 = w0, w1, w2
                res = rn
                nrm = nn_
                accepted = True
                break
            lam *= 0.5
        if not accepted:
            break
    return nrm <= tol, u0, u1, scale * math.exp(u2), it, nrm


# ---------------------------------------------------------------------------------------
# Object interface
# ---------------------------------------------------------------------------------------


class StaticTether:
    """Stateful quasi-steady tether, mirroring ``Tethers.QuasiSteady.Tether``.

    Usage::

        te = StaticTether(StaticSettings(segments=10, l_tether=200))
        te.init()
        te.step(kite_pos, kite_vel, tether_length=200.0, wind_vel=wind)
        f_on_kite = -te.force_kite
    """

    def __init__(self, settings: StaticSettings | None = None):
        self.set = settings if settings is not None else StaticSettings()
        self.clear()

    # -- state -------------------------------------------------------------------------
    def clear(self):
        n = n_nodes(self.set.segments)
        self.state_vec = np.zeros(3)  # beta [rad], phi [rad], Tn [N]
        self.kite_pos = np.zeros(3)
        self.kite_vel = np.zeros(3)
        self.wind_vel = np.zeros((3, n))
        self.tether_length = 0.0
        self.tether_pos = np.zeros((3, n))
        self.force_gnd = 0.0
        self.force_kite = np.zeros(3)
        self.p0 = np.zeros(3)
        self.converged = True
        self.iterations = 0
        return self

    @property
    def elevation(self):
        return self.state_vec[0]

    @property
    def azimuth(self):
        return self.state_vec[1]

    @property
    def tension(self):
        return self.state_vec[2]

    # -- initialisation ----------------------------------------------------------------
    def init(self, kite_pos=None, tether_length=None):
        """Initialise from ``set.elevation/azimuth/l_tether/slack`` (or the given kite
        position and length) with a catenary initial guess, then solve once."""
        se = self.set
        self.clear()
        L = se.l_tether if tether_length is None else float(tether_length)
        if kite_pos is None:
            b0, f0 = math.radians(se.elevation), math.radians(se.azimuth)
            dist = L / (1 + se.slack)
            kite_pos = np.array([dist * math.cos(b0) * math.cos(f0),
                                 dist * math.cos(b0) * math.sin(f0), dist * math.sin(b0)])
        kite_pos = np.asarray(kite_pos, dtype=float)
        self.state_vec[:] = catenary_initial_guess(kite_pos, L, se)
        return self.step(kite_pos, np.zeros(3), tether_length=L)

    # -- time step -----------------------------------------------------------------------
    def step(self, kite_pos, kite_vel, tether_length=None, wind_vel=None):
        """Move the tether end to ``kite_pos``/``kite_vel`` and re-solve."""
        se = self.set
        kp = np.ascontiguousarray(kite_pos, dtype=float)
        kv = np.ascontiguousarray(kite_vel, dtype=float)
        L = (1 + se.slack) * float(np.linalg.norm(kp)) if tether_length is None else float(tether_length)
        if wind_vel is not None:
            wind_vel = np.asarray(wind_vel, dtype=float)
            if wind_vel.shape != self.wind_vel.shape:
                raise ValueError(f"wind_vel must have shape {self.wind_vel.shape}")
            self.wind_vel[:] = wind_vel
        tol = se.abs_tol + se.rel_tol * L
        g = abs(se.g_earth)
        ok, b, f, T, it, nrm = _newton(self.state_vec[0], self.state_vec[1], self.state_vec[2], kp, kv,
                                      self.wind_vel, L, se.segments, se.rho, g, se.cd_tether,
                                      se.d_tether, se.rho_tether, se.c_spring, tol, se.max_iter,
                                      self.tether_pos)
        if not ok:
            # restart from a catenary guess (robust fallback)
            guess = catenary_initial_guess(kp, L, se)
            ok, b, f, T, it2, nrm = _newton(guess[0], guess[1], guess[2], kp, kv, self.wind_vel, L,
                                           se.segments, se.rho, g, se.cd_tether, se.d_tether,
                                           se.rho_tether, se.c_spring, tol, se.max_iter, self.tether_pos)
            it += it2
        self.converged = bool(ok)
        self.iterations = int(it)
        self.state_vec[:] = (b, f, T)
        out = tether_shape(b, f, T, kp, kv, self.wind_vel, L, se.segments, se.rho, g, se.cd_tether,
                           se.d_tether, se.rho_tether, se.c_spring, self.tether_pos, True)
        self.force_kite[:] = out[3:6]
        self.p0[:] = out[6:9]
        self.force_gnd = T
        self.kite_pos[:] = kp
        self.kite_vel[:] = kv
        self.tether_length = L
        return self

    def nodes(self):
        """All tether points from the ground station to the kite end, shape (3, segments+1)."""
        pts = np.zeros((3, self.set.segments + 1))
        pts[:, 1:-1] = self.tether_pos[:, ::-1]
        pts[:, -1] = self.p0
        return pts


# ---------------------------------------------------------------------------------------
# Catenary initial guess
# ---------------------------------------------------------------------------------------


def catenary_initial_guess(kite_pos, tether_length, se: StaticSettings):
    """(beta, phi, Tn) from the catenary through the origin and ``kite_pos``."""
    from scipy.optimize import brentq

    kp = np.asarray(kite_pos, dtype=float)
    h = math.hypot(kp[0], kp[1])
    v = kp[2]
    phi = math.atan2(kp[1], kp[0])
    w = se.rho_tether * (math.pi / 4 * (se.d_tether / 1000) ** 2) * abs(se.g_earth)
    L = tether_length
    chord = math.hypot(h, v)
    if L <= chord * (1 + 1e-9) or h < 1e-6:
        # taut (or vertical) tether: straight line, tension from weight as lower bound
        beta = math.atan2(v, max(h, 1e-9))
        return np.array([beta, phi, max(w * L, 1.0)])
    target = math.sqrt(L * L - v * v)

    def f(c):
        return 2 * math.sinh(c * h / 2) / c - target

    hi = 1.0
    while f(hi) < 0 and hi < 1e6:
        hi *= 2
    c = brentq(f, 1e-9, hi)
    x_min = -0.5 * (math.log((L + v) / (L - v)) / c - h)
    # slope at the ground station: dz/dx = sinh((0 - x_min) c)
    beta = math.atan(math.sinh(-x_min * c))
    tension = math.sqrt((w / c) ** 2 + (w * L) ** 2)
    return np.array([beta, phi, tension])


# ---------------------------------------------------------------------------------------
# Angle conventions (MATLAB reference fixtures)
# ---------------------------------------------------------------------------------------


def matlab_to_wind(theta_m, phi_m):
    """MATLAB reference convention -> (elevation, wind-frame azimuth)."""
    d1 = math.sin(theta_m) * math.cos(phi_m)
    d2 = math.sin(phi_m)
    d3 = math.cos(theta_m) * math.cos(phi_m)
    n = math.sqrt(d1 * d1 + d2 * d2 + d3 * d3)
    return math.asin(d3 / n), math.atan2(d2 / n, d1 / n)


def wind_to_matlab(beta, phi):
    """Inverse of :func:`matlab_to_wind`."""
    d1 = math.cos(beta) * math.cos(phi)
    d2 = math.cos(beta) * math.sin(phi)
    d3 = math.sin(beta)
    return math.atan2(d1, d3), math.asin(d2)
