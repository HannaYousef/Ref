"""Aerodynamic tables of the V3 kite: CL, CD, CM about the KCU vs AoA, and the trim AoA
vs depower.  Run from the repository root:  python examples_simple/02_aero_coefficients.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import yaml

from softkite import AeroModel, TurnRateLaw

OUT = Path(__file__).resolve().parents[1] / "output"
OUT.mkdir(exist_ok=True)

aero = AeroModel()
turn = TurnRateLaw()
print(aero.summary())

a = np.linspace(-6, 20, 200)
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
ax = axes[0, 0]
ax.plot(a, [aero.cl(x) for x in a], color="#2a78d6")
ax.set_ylabel("CL [-]")
ax = axes[0, 1]
ax.plot(a, [aero.cd(x) for x in a], color="#2a78d6", label="wing + bridle + KCU")
table = yaml.safe_load(open(aero.source))
if "cd_wing_only" in table:
    ax.plot(table["alpha_deg"], table["cd_wing_only"], ":", color="#52514e", label="wing only")
ax.set_ylabel("CD [-]")
ax.legend(frameon=False, fontsize=8)
ax = axes[1, 0]
for u, col in zip((0.22, 0.27, 0.33, 0.40), ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")):
    ax.plot(a, [aero.cm(x, u) for x in a], color=col, label=f"u_p = {u:.2f}")
    ax.plot([aero.alpha_trim(u)], [0.0], "o", color=col, ms=6)
ax.axhline(0, color="#52514e", lw=0.8)
ax.set_ylabel("CM about the KCU [-]")
ax.set_xlabel("angle of attack [deg]")
ax.legend(frameon=False, fontsize=8)
ax = axes[1, 1]
u = np.linspace(0.2, 0.42, 50)
ax.plot(u, [aero.alpha_trim(x) for x in u], color="#2a78d6")
ax.set_xlabel("relative depower u_p [-]")
ax.set_ylabel("trim AoA [deg]")
for axx in axes.flat:
    axx.grid(True, color="#e4e3df")
fig.suptitle("V3 kite, Python VSM (rigid CAD geometry, CFD polars, bridle drag)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "02_aero_coefficients.png", dpi=130)
print(f"saved {OUT / '02_aero_coefficients.png'}")
print("turn-rate law c1(u_p):", {round(float(x), 3): round(turn.c1(x), 3) for x in turn.u_p})
plt.show()
