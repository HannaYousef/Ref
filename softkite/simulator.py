"""Simulation loop, logging and cycle statistics."""

from __future__ import annotations

import math
import time as _time
from pathlib import Path

import numpy as np

from .kite_model import KiteModel
from .settings import Settings, load_settings

SCALARS = ["time", "l_tether", "v_reel", "v_set", "f_gnd", "f_motor", "power", "v_a", "v_wind",
           "alpha", "cl", "cd", "cs", "cm", "side_slip", "heading", "course", "v_tangential", "beta", "phi",
           "u_s", "u_s_lagged", "u_p", "distance", "height", "psi_dot", "c1", "c2"]
VECTORS = ["pos", "vel", "x_k", "y_k", "e_r"]


class SimLog:
    """Column store of a run.  ``log["f_gnd"]`` is an array over time."""

    def __init__(self, data: dict | None = None, meta: dict | None = None):
        self.data = data if data is not None else {}
        self.meta = meta if meta is not None else {}

    def __getitem__(self, k):
        return self.data[k]

    def __contains__(self, k):
        return k in self.data

    def keys(self):
        return self.data.keys()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.data, _meta=np.array(repr(self.meta)))
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        data = {k: z[k] for k in z.files if k != "_meta"}
        meta = {}
        if "_meta" in z.files:
            try:
                import ast

                meta = ast.literal_eval(str(z["_meta"]))
            except Exception:
                meta = {"raw": str(z["_meta"])}
        return cls(data, meta)


class Simulation:
    """Couples a :class:`KiteModel` and a controller and logs the run."""

    def __init__(self, settings: Settings | None = None, controller_cls=None, model: KiteModel | None = None,
                 **controller_kwargs):
        from .controllers import PumpingCycleController

        self.s = settings if settings is not None else load_settings()
        self.model = model if model is not None else KiteModel(self.s)
        self.controller_cls = controller_cls or PumpingCycleController
        self.controller_kwargs = controller_kwargs
        self.controller = None

    def run(self, sim_time: float | None = None, verbose: bool = True, stop_after_cycles: int | None = None,
            **init_kwargs) -> SimLog:
        s = self.s
        sim_time = s.simulation.sim_time if sim_time is None else sim_time
        m = self.model.init(**init_kwargs)
        ctrl = self.controller_cls(s, m, **self.controller_kwargs)
        self.controller = ctrl
        n = int(round(sim_time / s.simulation.dt))
        rec = {k: [] for k in SCALARS + VECTORS + ["phase", "cycle", "chi_set", "tether_nodes", "tether_ok"]}
        from .controllers import PHASES

        t0 = _time.time()
        last_print = -1e9
        for i in range(n):
            o = m.out
            u_s, u_p, v_set = ctrl.update(o)
            m.step(u_s, u_p, v_set)
            o = m.out
            for k in SCALARS:
                rec[k].append(o[k])
            for k in VECTORS:
                rec[k].append(np.array(o[k], dtype=float))
            rec["tether_nodes"].append(o["tether_nodes"])
            rec["tether_ok"].append(o["tether_ok"])
            rec["phase"].append(PHASES.index(ctrl.phase) if ctrl.phase in PHASES else -1)
            rec["cycle"].append(ctrl.cycle)
            rec["chi_set"].append(ctrl.chi_set)
            if verbose and o["time"] - last_print >= 10.0:
                last_print = o["time"]
                print(f"t={o['time']:6.1f}s {ctrl.phase:15s} l={o['l_tether']:6.1f}m "
                      f"el={math.degrees(o['beta']):5.1f} az={math.degrees(o['phi']):6.1f} "
                      f"F={o['f_gnd']:7.0f}N v_ro={o['v_reel']:5.2f} v_a={o['v_a']:5.1f} "
                      f"P={o['power'] / 1000:6.2f}kW")
            if m.crashed:
                print(f"!! kite crashed or state diverged at t = {m.t:.2f} s")
                break
            if stop_after_cycles is not None and ctrl.cycle > stop_after_cycles:
                break
        wall = _time.time() - t0
        if verbose:
            print(f"simulated {m.t:.1f} s in {wall:.1f} s wall time ({m.t / max(wall, 1e-9):.1f}x real time)")
        data = {k: np.asarray(v) for k, v in rec.items()}
        meta = {"settings": s.to_dict(), "phases": PHASES, "wall_time": wall, "crashed": m.crashed,
                "aero_source": m.aero.source, "controller": type(ctrl).__name__}
        if hasattr(ctrl, "transitions"):
            meta["transitions"] = ctrl.transitions
        return SimLog(data, meta)


def run(settings: Settings | None = None, sim_time: float | None = None, controller_cls=None, verbose=True,
        **kwargs) -> SimLog:
    """Convenience wrapper: ``log = run(load_settings(), 300)``."""
    return Simulation(settings, controller_cls).run(sim_time, verbose=verbose, **kwargs)


def cycle_stats(log: SimLog):
    """Per-cycle statistics of a pumping run (complete cycles only).

    A cycle starts with REEL_OUT_ENTRY and ends at the next REEL_OUT_ENTRY.
    Returns a list of dicts with duration, energy, mean power and phase durations.
    """
    from .controllers import PHASES, REEL_OUT_ENTRY

    t = log["time"]
    ph = log["phase"]
    cyc = log["cycle"]
    dt = np.diff(t, prepend=t[0] - (t[1] - t[0] if len(t) > 1 else 0.02))
    p = log["power"]
    stats = []
    for c in range(1, int(cyc.max()) + 1):
        idx = np.where(cyc == c)[0]
        if len(idx) == 0 or c == int(cyc.max()):
            continue  # last cycle incomplete
        e_out = float(np.sum((p * dt)[idx][p[idx] > 0]))
        e_in = float(np.sum((p * dt)[idx][p[idx] < 0]))
        dur = float(t[idx[-1]] - t[idx[0]] + dt[idx[0]])
        d = {"cycle": c, "duration": dur, "energy_out": e_out, "energy_in": e_in,
             "energy": e_out + e_in, "mean_power": (e_out + e_in) / dur,
             "f_max": float(log["f_gnd"][idx].max()), "v_ro_max": float(log["v_reel"][idx].max()),
             "v_ri_max": float(-log["v_reel"][idx].min())}
        for k, name in enumerate(PHASES):
            d["t_" + name] = float(np.sum(dt[idx][ph[idx] == k]))
        ro = idx[ph[idx] == PHASES.index("reel_out")]
        if len(ro):
            d["mean_power_ro"] = float(np.mean(p[ro]))
            d["mean_force_ro"] = float(np.mean(log["f_gnd"][ro]))
            d["mean_v_ro"] = float(np.mean(log["v_reel"][ro]))
            d["mean_cl_ro"] = float(np.mean(log["cl"][ro]))
            d["mean_alpha_ro"] = float(np.mean(log["alpha"][ro]))
            d["mean_va_ro"] = float(np.mean(log["v_a"][ro]))
        ri = idx[ph[idx] == PHASES.index("reel_in")]
        if len(ri):
            d["mean_force_ri"] = float(np.mean(log["f_gnd"][ri]))
            d["mean_v_ri"] = float(-np.mean(log["v_reel"][ri]))
        stats.append(d)
    return stats
