"""Quasi-steady tether: shape and forces of a 300 m tether in four situations.

Side view (x-z) and top view (x-y).  The kite sits at (220, 0, 200) m, 1 % closer to
the ground station than the tether is long, so the tether can sag.
Run from the repository root:  python examples_simple/01_static_tether.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import matplotlib.pyplot as plt
import numpy as np

from softkite import StaticSettings, StaticTether

OUT = Path(__file__).resolve().parents[1] / "output"
OUT.mkdir(exist_ok=True)

se = StaticSettings(segments=20)
kite_pos = np.array([220.0, 0.0, 200.0])
L = 1.01 * np.linalg.norm(kite_pos)
wind = np.zeros((3, se.segments - 1))
wind[0] = 10.0  # 10 m/s downwind at all nodes
radial = kite_pos / np.linalg.norm(kite_pos)
cases = [("kite at rest, no wind", np.zeros(3), np.zeros_like(wind), "#2a78d6"),
         ("kite at rest, 10 m/s wind", np.zeros(3), wind, "#eb6834"),
         ("kite flying 30 m/s sideways (+y), 10 m/s wind", np.array([0.0, 30.0, 0.0]), wind, "#1baf7a"),
         ("same + reeling out at 5 m/s", np.array([0.0, 30.0, 0.0]) + 5 * radial, wind, "#6250d6")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
ax1.plot([0, kite_pos[0]], [0, kite_pos[2]], ":", color="#52514e", lw=1, label="straight line")
for label, vel, w, col in cases:
    te = StaticTether(se).init(kite_pos=kite_pos, tether_length=L)
    te.step(kite_pos, vel, tether_length=L, wind_vel=w)
    n = te.nodes()
    ax1.plot(n[0], n[2], "o-", ms=2.5, lw=1.2, color=col, label=label)
    ax2.plot(n[0], n[1], "o-", ms=2.5, lw=1.2, color=col)
    print(f"{label:46s} converged={te.converged}  tension at drum = {te.force_gnd:7.1f} N  "
          f"force on kite = {np.linalg.norm(te.force_kite):7.1f} N  "
          f"elevation at drum = {math.degrees(te.elevation):5.1f} deg")
ax1.set_xlabel("x, downwind [m]"); ax1.set_ylabel("z, height [m]"); ax1.set_title("side view", fontsize=10)
ax2.set_xlabel("x, downwind [m]"); ax2.set_ylabel("y, sideways [m]"); ax2.set_title("top view", fontsize=10)
ax1.legend(frameon=False, fontsize=7.5, loc="upper left")
for ax in (ax1, ax2):
    ax.grid(True, color="#e4e3df")
fig.tight_layout()
fig.savefig(OUT / "01_static_tether.png", dpi=130)
print(f"saved {OUT / '01_static_tether.png'}")
plt.show()
