"""Constant-length figures of eight compared with the apparent wind speed that
V3Kite.jl reports in its turn-rate identification runs (softkite/data/v3_turn_rate_coeffs.yaml:
150 m tether, 9.51 m/s wind, 30 deg elevation).

This is a plausibility check of the aero tables against the full V3Kite.jl model, not a
strict validation: the V3Kite numbers come from steering sweeps, not steady eights.
Run from the repository root:  python examples_simple/08_compare_v3kite_vapp.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from softkite import Simulation, load_settings
from softkite.controllers import Fig8Controller
from softkite.settings import DATA

tab = yaml.safe_load(open(DATA / "v3_turn_rate_coeffs.yaml"))
rows = sorted((e["depower"], e["v_app"]) for e in tab["entries"])
print(" u_p   v_app V3Kite   v_app here   AoA here   F here")
for u_p, va_ref in rows:
    s = load_settings(simulation__l_tether0=150.0, simulation__elevation0=60.0, control__f8_elevation=30.0,
                      winch__f_motor_max=1e6)  # hold the length whatever the force
    log = Simulation(s, Fig8Controller, depower=u_p).run(60.0, verbose=False)
    m = log["time"] > 25
    print(f"{u_p:5.3f}  {va_ref:10.1f}   {log['v_a'][m].mean():10.1f}   {log['alpha'][m].mean():8.2f}"
          f"   {log['f_gnd'][m].mean():7.0f}")
