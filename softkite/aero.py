"""Aerodynamic coefficients of the soft wing as functions of the mean angle of attack.

The tables are produced offline (``scripts/identify_aero_vsm.py`` with the Python VSM,
or ``scripts/fit_v3kite_logs.py`` from V3Kite.jl runs) and stored in YAML; this module
only reads and interpolates them, so the simulation has no VSM/Julia dependency.

Pitch trim
----------
A bridled soft wing pitches about its bridle point (the KCU) until the aerodynamic
moment about that point vanishes.  Its pitch dynamics are fast compared to the flight
path, so the reference model treats the angle of attack as the quasi-static trim angle

    alpha = alpha_trim(u_p)   with   CM_KCU(alpha_trim, delta(u_p)) = 0,

where ``u_p`` is the relative depower setting and ``delta`` the resulting shortening
of the front (power) bridle lines.  ``delta(u_p)`` follows the V3 depower-tape
calibration (V3Kite.jl, ``examples/depower_drum_model.jl``, 2025 fit) scaled linearly
between a powered and a depowered reference state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

DATA = Path(__file__).resolve().parent / "data"


def tape_length_2025(u_p: float) -> float:
    """V3 depower-tape length [m] for relative depower ``u_p`` (V3Kite.jl ld_2025)."""
    x = 100.0 * u_p
    return -0.00017237 * x * x + 0.06623795 * x + 0.192


@dataclass
class DepowerMap:
    """Map relative depower ``u_p`` -> front-line shortening ``delta`` [m]."""

    u_p_powered: float = 0.22  #: fully powered setting (KiteControllers.jl min_depower)
    u_p_depowered: float = 0.40  #: depowered setting (KiteControllers.jl max_depower)
    delta_powered: float = -0.036  #: VSM compute_trim_angle.py, powered
    delta_depowered: float = 0.66  #: VSM compute_trim_angle.py, depowered

    def delta(self, u_p: float) -> float:
        l0 = tape_length_2025(self.u_p_powered)
        l1 = tape_length_2025(self.u_p_depowered)
        f = (tape_length_2025(u_p) - l0) / (l1 - l0)
        return self.delta_powered + f * (self.delta_depowered - self.delta_powered)


class AeroModel:
    """CL, CD, CS and CM tables of the wing (incl. bridle and KCU drag).

    Parameters
    ----------
    path : YAML table (default ``softkite/data/v3_aero.yaml``)
    cd_kcu_area : drag area of the KCU, CD*A [m^2], added to CD
    cd_extra : additional, user-calibrated drag coefficient (default 0)
    depower_map : :class:`DepowerMap`
    """

    def __init__(self, path: str | Path | None = None, cd_kcu_area: float = 0.059,
                 cd_extra: float = 0.0, depower_map: DepowerMap | None = None):
        path = Path(path) if path is not None else DATA / "v3_aero.yaml"
        with open(path) as f:
            d = yaml.safe_load(f)
        self.source = str(path)
        self.meta = d.get("provenance", {})
        ref = d["reference"]
        self.area = float(ref["area"])
        self.chord = float(ref["chord"])
        self.span = float(ref.get("span", float("nan")))
        self.alpha_deg = np.asarray(d["alpha_deg"], dtype=float)
        self._cl = np.asarray(d["cl"], dtype=float)
        self._cd = np.asarray(d["cd"], dtype=float)
        self.cd_kcu = cd_kcu_area / self.area
        self.cd_extra = float(cd_extra)
        self.cs_beta = float(d.get("cs_beta", 1.0))
        dp = d["depower"]
        self.dmap = depower_map if depower_map is not None else DepowerMap()
        # table indexed either by delta (VSM) or directly by u_p (V3Kite fit)
        if "u_p" in dp:
            self._dp_axis = np.asarray(dp["u_p"], dtype=float)
            self._dp_is_up = True
        else:
            self._dp_axis = np.asarray(dp["delta_m"], dtype=float)
            self._dp_is_up = False
        self._alpha_trim = np.asarray(dp["alpha_trim_deg"], dtype=float)
        self._cm = np.asarray(dp["cm"], dtype=float) if "cm" in dp else None
        if np.any(np.isnan(self._alpha_trim)):
            ok = ~np.isnan(self._alpha_trim)
            self._dp_axis, self._alpha_trim = self._dp_axis[ok], self._alpha_trim[ok]
            if self._cm is not None:
                self._cm = self._cm[ok]

    # -- coefficients -------------------------------------------------------------------
    def cl(self, alpha_deg: float) -> float:
        return float(np.interp(alpha_deg, self.alpha_deg, self._cl))

    def cd(self, alpha_deg: float) -> float:
        """Total drag coefficient of wing + bridle + KCU (+ cd_extra), tether excluded."""
        return float(np.interp(alpha_deg, self.alpha_deg, self._cd)) + self.cd_kcu + self.cd_extra

    def depower_axis(self, u_p: float) -> float:
        return u_p if self._dp_is_up else self.dmap.delta(u_p)

    def alpha_trim(self, u_p: float) -> float:
        """Trim angle of attack [deg] for relative depower ``u_p``."""
        x = self.depower_axis(u_p)
        return float(np.interp(x, self._dp_axis, self._alpha_trim))

    def cm(self, alpha_deg: float, u_p: float) -> float:
        """Pitching moment coefficient about the KCU (bilinear interpolation)."""
        if self._cm is None:
            return float("nan")
        x = self.depower_axis(u_p)
        i = int(np.clip(np.searchsorted(self._dp_axis, x) - 1, 0, len(self._dp_axis) - 2))
        t = (x - self._dp_axis[i]) / (self._dp_axis[i + 1] - self._dp_axis[i])
        t = min(max(t, 0.0), 1.0)
        c0 = np.interp(alpha_deg, self.alpha_deg, self._cm[i])
        c1 = np.interp(alpha_deg, self.alpha_deg, self._cm[i + 1])
        return float((1 - t) * c0 + t * c1)

    def summary(self, u_ps=(0.22, 0.25, 0.30, 0.35, 0.40)) -> str:
        lines = [f"Aero tables: {self.source}",
                 f"  S = {self.area:.2f} m^2, c = {self.chord:.2f} m, CS_beta = {self.cs_beta:.2f} 1/rad,"
                 f" CD_kcu = {self.cd_kcu:.4f}, cd_extra = {self.cd_extra:.3f}",
                 "   u_p   alpha_trim   CL     CD     L/D"]
        for u in u_ps:
            a = self.alpha_trim(u)
            cl, cd = self.cl(a), self.cd(a)
            lines.append(f"  {u:5.3f}   {a:7.2f}     {cl:5.3f}  {cd:5.3f}  {cl / cd:5.2f}")
        return "\n".join(lines)


class TurnRateLaw:
    """Turn-rate law of the kite, identified with V3Kite.jl (RobustKiteControllers.jl):

        psi_dot = c1 * v_a * u_s + c2 / v_a * sin(psi) * cos(beta)

    ``c1`` [1/m], ``c2`` [m/s^2], the steering dead time ``delay`` and the kite's own
    first-order lag ``kite_lag`` are interpolated in depower (log-linear for c1) from
    ``softkite/data/v3_turn_rate_coeffs.yaml``.  Outside the identified depower range the
    nearest row is used.
    """

    def __init__(self, path: str | Path | None = None, body_damping=(0.0, 0.0, 40.0)):
        path = Path(path) if path is not None else DATA / "v3_turn_rate_coeffs.yaml"
        with open(path) as f:
            d = yaml.safe_load(f)
        rows = [e for e in d["entries"] if list(map(float, e["body_damping"])) == list(body_damping)]
        if not rows:
            raise ValueError(f"no turn-rate rows with body_damping {body_damping}")
        rows.sort(key=lambda e: e["depower"])
        self.u_p = np.array([e["depower"] for e in rows], dtype=float)
        self.log_c1 = np.log([e["c1"] for e in rows])
        self.c2_tab = np.array([e["c2"] for e in rows], dtype=float)
        self.delay_tab = np.array([e["delay"] for e in rows], dtype=float)
        self.lag_tab = np.array([e.get("kite_lag", 0.05) for e in rows], dtype=float)
        self.conditions = d.get("conditions", {})

    def c1(self, u_p):
        return float(math.exp(np.interp(u_p, self.u_p, self.log_c1)))

    def c2(self, u_p):
        return float(np.interp(u_p, self.u_p, self.c2_tab))

    def delay(self, u_p):
        return float(np.interp(u_p, self.u_p, self.delay_tab))

    def kite_lag(self, u_p):
        return float(np.interp(u_p, self.u_p, self.lag_tab))

    def rate(self, u_s, v_a, psi, beta, u_p):
        v = max(v_a, 1.0)
        return self.c1(u_p) * v * u_s + self.c2(u_p) / v * math.sin(psi) * math.cos(beta)
