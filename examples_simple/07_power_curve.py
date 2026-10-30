"""Mean cycle power vs wind speed (a few minutes: one 300 s run per wind speed).

Run from the repository root:  python examples_simple/07_power_curve.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from softkite import Simulation, cycle_stats, load_settings

OUT = Path(__file__).resolve().parents[1] / "output"
OUT.mkdir(exist_ok=True)

winds = [5.0, 6.0, 7.0, 8.0, 9.51, 11.0, 12.5]
rows = []
for vw in winds:
    s = load_settings(environment__v_wind_ref=vw)
    log = Simulation(s).run(320.0, verbose=False)
    st = cycle_stats(log)[1:]  # skip the first cycle (start-up)
    if not st or log.meta["crashed"]:
        print(f"v_wind = {vw:5.2f} m/s: no complete cycle (crashed={log.meta['crashed']})")
        continue
    p = np.mean([c["mean_power"] for c in st])
    rows.append((vw, p, np.mean([c["duration"] for c in st]), max(c["f_max"] for c in st)))
    print(f"v_wind = {vw:5.2f} m/s (at {s.environment.h_ref:.0f} m): mean cycle power {p / 1e3:6.2f} kW, "
          f"cycle time {rows[-1][2]:5.1f} s, F_max {rows[-1][3]:5.0f} N")

r = np.array(rows)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(r[:, 0], r[:, 1] / 1e3, "o-", color="#2a78d6", ms=8, lw=2)
ax.set_xlabel(f"wind speed at {s.environment.h_ref:.0f} m [m/s]")
ax.set_ylabel("mean mechanical cycle power [kW]")
ax.grid(True, color="#e4e3df")
fig.tight_layout()
fig.savefig(OUT / "07_power_curve.png", dpi=130)
np.savetxt(OUT / "07_power_curve.csv", r, delimiter=",", header="v_wind,mean_power,cycle_time,f_max", comments="")
print(f"saved {OUT / '07_power_curve.png'}")
plt.show()
