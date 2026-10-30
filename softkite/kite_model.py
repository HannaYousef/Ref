"""Rigid-wing point-mass model of a soft (LEI) kite on a quasi-steady tether.

State (W frame: x downwind, z up)::

    y = [p (3), v (3), psi, l_tether, v_reel, u_s_lagged]

Physics per time step
---------------------
* **Tether**: quasi-steady tether (:mod:`softkite.static_tether`) between the ground
  station and the kite; it returns the force on the kite and the tension at the drum.
* **Aerodynamics**: the wing is a rigid body whose angle of attack is its quasi-static
  pitch trim about the bridle point, ``alpha = alpha_trim(u_p)`` (see
  :mod:`softkite.aero`).  Lift is normal to the apparent wind and to the span axis,
  drag is along the apparent wind, and a side force ``CS_beta * beta`` acts in the
  plane normal to the apparent wind when the kite side-slips.  Bridle and KCU drag
  are included in CD; tether drag is in the tether model.
* **Heading**: the span axis lies in the plane tangential to the sphere around the
  ground station and is rotated by the heading ``psi`` (0 = nose to zenith, positive
  = nose to the right seen from the ground station).  ``psi`` follows the turn-rate
  law identified with V3Kite.jl::

      psi_dot = c1(u_p) * v_a * u_s + c2(u_p) / v_a * sin(psi) * cos(elevation)

  The side force then makes the flight direction (course) follow the heading.
* **KCU**: rate-limited steering and depower, steering dead time and first-order
  kite lag from the turn-rate identification.
* **Winch**: drum inertia ``m_eff`` driven by a motor with a speed loop
  (time constant ``tau_speed``) and a force limit ``f_motor_max``.

Integration: fixed-step RK4 with ``substeps`` steps per control period.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .aero import AeroModel, DepowerMap, TurnRateLaw
from .settings import Settings, load_settings
from .static_tether import StaticSettings, StaticTether


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def sphere_frame(p):
    """Unit vectors (e_r, e_up, e_right) at position p; also (elevation, azimuth)."""
    r = math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
    beta = math.asin(max(-1.0, min(1.0, p[2] / r)))
    phi = math.atan2(p[1], p[0])
    sb, cb, sp, cp = math.sin(beta), math.cos(beta), math.sin(phi), math.cos(phi)
    e_r = np.array([cb * cp, cb * sp, sb])
    e_up = np.array([-sb * cp, -sb * sp, cb])
    e_right = np.array([sp, -cp, 0.0])  # = e_r x e_up
    return e_r, e_up, e_right, beta, phi


class KCU:
    """Kite control unit: rate limits and steering dead time (discrete, at the control rate)."""

    def __init__(self, s: Settings, turn: TurnRateLaw, dt: float, depower0: float):
        self.s = s.kcu
        self.turn = turn
        self.dt = dt
        self.u_p = depower0
        self.u_s = 0.0
        n_max = int(math.ceil(0.5 / dt)) + 2
        self.buf = deque([0.0] * n_max, maxlen=n_max)

    def update(self, u_s_set: float, u_p_set: float):
        lim = self.s.max_steering
        u_s_set = min(max(u_s_set, -lim), lim)
        du = self.s.v_steering * self.dt
        self.u_s += min(max(u_s_set - self.u_s, -du), du)
        dp = self.s.v_depower * self.dt
        self.u_p += min(max(u_p_set - self.u_p, -dp), dp)
        self.buf.appendleft(self.u_s)
        if self.s.use_delay:
            n = int(round(self.turn.delay(self.u_p) / self.dt))
            n = min(n, len(self.buf) - 1)
            return self.buf[n], self.u_p
        return self.u_s, self.u_p


class KiteModel:
    """Point-mass soft-wing kite + quasi-steady tether + winch."""

    N_STATE = 10

    def __init__(self, settings: Settings | None = None):
        s = settings if settings is not None else load_settings()
        self.s = s
        k = s.kite
        dmap = DepowerMap(u_p_powered=k.u_p_powered, u_p_depowered=k.u_p_depowered)
        self.aero = AeroModel(s.data_file(k.aero_file), cd_kcu_area=k.cd_kcu_area,
                              cd_extra=k.cd_extra, depower_map=dmap)
        self.turn = TurnRateLaw(s.data_file(k.turn_rate_file))
        self.mass = k.m_wing + k.m_kcu
        t = s.tether
        env = s.environment
        self.tether = StaticTether(StaticSettings(
            segments=t.segments, rho=env.rho, g_earth=env.g, cd_tether=t.cd_tether,
            d_tether=t.d_tether, rho_tether=t.rho_tether, c_spring=t.c_spring,
            l_tether=s.simulation.l_tether0))
        nn = t.segments - 1
        self._wind_nodes = np.zeros((3, nn))
        self._node_frac = np.array([(t.segments - j - 1) / t.segments for j in range(nn)])
        wd = math.radians(env.wind_dir)
        self._wind_unit = np.array([math.cos(wd), math.sin(wd), 0.0])
        self.y = np.zeros(self.N_STATE)
        self.t = 0.0
        self.out: dict = {}
        self.crashed = False

    # -- environment ---------------------------------------------------------------------
    def wind_speed(self, z: float) -> float:
        """Wind speed at height z: power law (default) or logarithmic law."""
        env = self.s.environment
        z = max(z, 0.5)
        if env.profile_law == "log":
            return env.v_wind_ref * math.log(max(z, 1.01 * env.z0) / env.z0) / math.log(env.h_ref / env.z0)
        return env.v_wind_ref * (z / env.h_ref) ** env.profile_exp

    def wind_at(self, p) -> np.ndarray:
        return self.wind_speed(p[2]) * self._wind_unit

    # -- initialisation ------------------------------------------------------------------
    def init(self, l_tether=None, elevation=None, azimuth=None, heading=None, depower=None):
        sim = self.s.simulation
        l0 = sim.l_tether0 if l_tether is None else l_tether
        b = math.radians(sim.elevation0 if elevation is None else elevation)
        f = math.radians(sim.azimuth0 if azimuth is None else azimuth)
        psi = math.radians(sim.heading0 if heading is None else heading)
        self.depower0 = sim.depower0 if depower is None else depower
        r = 0.998 * l0
        p = r * np.array([math.cos(b) * math.cos(f), math.cos(b) * math.sin(f), math.sin(b)])
        self.y[:] = 0.0
        self.y[0:3] = p
        self.y[6] = psi
        self.y[7] = l0
        self.t = 0.0
        self.crashed = False
        self.tether.init(kite_pos=p, tether_length=l0)
        self.kcu = KCU(self.s, self.turn, sim.dt, self.depower0)
        self.inputs = (0.0, self.depower0, 0.0)  # (u_s applied, u_p, v_set)
        self.outputs(self.y, self.inputs)
        return self

    # -- right-hand side -----------------------------------------------------------------
    def _forces(self, y, inputs):
        """Return derivative and a dict of intermediate quantities."""
        u_s, u_p, v_set = inputs
        p = y[0:3]
        v = y[3:6]
        psi, l, v_ro, us_f = y[6], y[7], y[8], y[9]
        s = self.s
        env = s.environment

        # wind along the (approximately straight) tether and at the kite
        vw_kite = self.wind_speed(p[2])
        wn = self._wind_nodes
        for j, fr in enumerate(self._node_frac):
            w = self.wind_speed(p[2] * fr)
            wn[0, j] = w * self._wind_unit[0]
            wn[1, j] = w * self._wind_unit[1]
        te = self.tether.step(p, v, tether_length=l, wind_vel=wn)
        f_tether = -te.force_kite
        f_gnd = te.force_gnd

        e_r, e_up, e_right, beta, phi = sphere_frame(p)
        cps, sps = math.cos(psi), math.sin(psi)
        x_k = cps * e_up + sps * e_right
        y_k = cps * e_right - sps * e_up  # span axis = e_r x x_k (observer's right when the nose is up)

        v_a = vw_kite * self._wind_unit - v
        va = float(np.linalg.norm(v_a))
        va_s = max(va, 1e-3)
        va_hat = v_a / va_s
        alpha = self.aero.alpha_trim(u_p)
        cl = self.aero.cl(alpha)
        cd = self.aero.cd(alpha)
        q = 0.5 * env.rho * va * va
        S = self.aero.area
        lift_dir = np.cross(y_k, v_a)
        ln = np.linalg.norm(lift_dir)
        lift_dir = lift_dir / ln if ln > 1e-9 else e_r
        sin_bs = max(-1.0, min(1.0, float(np.dot(va_hat, y_k))))
        beta_s = math.asin(sin_bs)
        side_dir = y_k - sin_bs * va_hat
        sn = np.linalg.norm(side_dir)
        side_dir = side_dir / sn if sn > 1e-9 else y_k
        cs = self.aero.cs_beta * beta_s
        f_lift = q * S * cl * lift_dir
        f_drag = q * S * cd * va_hat
        f_side = q * S * cs * side_dir
        f_aero = f_lift + f_drag + f_side
        f_grav = np.array([0.0, 0.0, -self.mass * env.g])
        acc = (f_aero + f_tether + f_grav) / self.mass

        # heading (turn-rate law) and steering lag
        psi_dot = self.turn.c1(u_p) * va_s * us_f + self.turn.c2(u_p) / max(va_s, 1.0) * sps * math.cos(beta)
        lag = max(self.turn.kite_lag(u_p), 1e-3)
        us_dot = (u_s - us_f) / lag

        # winch: motor speed loop with force limit
        w = s.winch
        f_motor = f_gnd - w.m_eff * (v_set - v_ro) / w.tau_speed
        f_motor = min(max(f_motor, -w.f_motor_max), w.f_motor_max)
        a_ro = (f_gnd - f_motor) / w.m_eff

        dy = np.empty(self.N_STATE)
        dy[0:3] = v
        dy[3:6] = acc
        dy[6] = psi_dot
        dy[7] = v_ro
        dy[8] = a_ro
        dy[9] = us_dot
        info = dict(beta=beta, phi=phi, e_r=e_r, e_up=e_up, e_right=e_right, x_k=x_k, y_k=y_k,
                    v_a=va, alpha=alpha, cl=cl, cd=cd, cs=cs, side_slip=beta_s, f_aero=f_aero,
                    f_lift=f_lift, f_drag=f_drag, f_tether=f_tether, f_gnd=f_gnd, f_motor=f_motor,
                    v_wind=vw_kite, psi_dot=psi_dot, tether_ok=te.converged)
        return dy, info

    def rhs(self, y, inputs):
        return self._forces(y, inputs)[0]

    # -- stepping ------------------------------------------------------------------------
    def step(self, u_s_set: float, u_p_set: float, v_set: float):
        """Advance one control period with the commanded steering, depower and reel speed."""
        sim = self.s.simulation
        u_s, u_p = self.kcu.update(u_s_set, u_p_set)
        inputs = (u_s, u_p, v_set)
        self.inputs = inputs
        h = sim.dt / sim.substeps
        y = self.y
        for _ in range(sim.substeps):
            k1 = self.rhs(y, inputs)
            k2 = self.rhs(y + 0.5 * h * k1, inputs)
            k3 = self.rhs(y + 0.5 * h * k2, inputs)
            k4 = self.rhs(y + h * k3, inputs)
            y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y[6] = wrap_pi(y[6])
        self.y = y
        self.t += sim.dt
        self.outputs(y, inputs)
        if y[2] < 1.0 or not np.all(np.isfinite(y)):
            self.crashed = True
        return self

    def outputs(self, y, inputs):
        dy, info = self._forces(y, inputs)
        p, v = y[0:3], y[3:6]
        v_t = v - np.dot(v, info["e_r"]) * info["e_r"]
        vt = float(np.linalg.norm(v_t))
        course = math.atan2(float(np.dot(v_t, info["e_right"])), float(np.dot(v_t, info["e_up"]))) if vt > 1e-6 else y[6]
        o = self.out
        o.update(info)
        o.update(time=self.t, pos=p.copy(), vel=v.copy(), heading=y[6], course=course, v_tangential=vt,
                 l_tether=y[7], v_reel=y[8], u_s=inputs[0], u_s_lagged=y[9], u_p=inputs[1], v_set=inputs[2],
                 power=info["f_gnd"] * y[8], distance=float(np.linalg.norm(p)),
                 height=float(p[2]), c1=self.turn.c1(inputs[1]), c2=self.turn.c2(inputs[1]),
                 cm=self.aero.cm(info["alpha"], inputs[1]), tether_nodes=self.tether.nodes().copy())
        return o
