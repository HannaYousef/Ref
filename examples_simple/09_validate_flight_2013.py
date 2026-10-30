"""Compare the model with the measured pumping cycles of the 25 m^2 TU Delft LEI kite
on 7 May 2013 (van der Vlugt et al. 2019, Renewable Energy 131, arXiv:1705.04133).

Two runs under the flight's conditions (softkite/data/settings_v3_flight2013.yaml):
  A  pure rigid-wing VSM aerodynamics      (kite.cd_extra = 0)
  B  flight-calibrated drag                (kite.cd_extra = 0.067 -> L/D about 3.6)
A few minutes.  Run from the repository root:
    python examples_simple/09_validate_flight_2013.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from softkite import Simulation, cycle_stats, load_settings
from softkite.controllers import PHASES
from softkite.settings import DATA

OUT = Path(__file__).resolve().parents[1] / "output"
OUT.mkdir(exist_ok=True)

# measured values, van der Vlugt et al. (2019), Tables 2 and 3 (moderate wind case)
MEASURED = {"traction power": 6.23, "traction time": 66, "transition power": 3.44, "transition time": 12,
            "retraction power": -2.60, "retraction time": 43, "cycle power": 2.79, "cycle time": 122}


def run(cd_extra):
    s = load_settings(DATA / "settings_v3_flight2013.yaml", kite__cd_extra=cd_extra)
    log = Simulation(s).run(500.0, verbose=False)
    st = cycle_stats(log)[1:]  # skip the start-up cycle
    ph, p, cyc = log["phase"], log["power"], log["cycle"]
    res = {}
    for name, idx in (("traction", [PHASES.index("reel_out_entry"), PHASES.index("reel_out")]),
                      ("transition", [PHASES.index("retract")]), ("retraction", [PHASES.index("reel_in")])):
        pw, tm = [], []
        for c in st:
            m = (cyc == c["cycle"]) & np.isin(ph, idx)
            pw.append(p[m].mean() / 1e3)
            tm.append(m.sum() * s.simulation.dt)
        res[name + " power"], res[name + " time"] = float(np.mean(pw)), float(np.mean(tm))
    res["cycle power"] = float(np.mean([c["mean_power"] for c in st]) / 1e3)
    res["cycle time"] = float(np.mean([c["duration"] for c in st]))
    m = ph == PHASES.index("reel_out")
    res["glide ratio L/D (reel-out)"] = float(np.mean(log["cl"][m] / log["cd"][m]))
    return res


A = run(0.0)
B = run(0.067)
print(f"{'quantity':28s} {'measured':>9s} {'A: VSM':>9s} {'B: calibrated':>14s}")
for k in list(MEASURED) + ["glide ratio L/D (reel-out)"]:
    meas = MEASURED.get(k, float("nan"))
    print(f"{k:28s} {meas:9.2f} {A[k]:9.2f} {B[k]:14.2f}")
for name, r in (("measured", MEASURED), ("A", A), ("B", B)):
    e = r["traction power"] * r["traction time"] + r["transition power"] * r["transition time"] \
        + r["retraction power"] * r["retraction time"]
    print(f"energy per cycle, {name:9s}: {e:6.0f} kJ")

keys = ["traction power", "transition power", "retraction power", "cycle power"]
x = np.arange(len(keys))
fig, ax = plt.subplots(figsize=(8, 4))
for i, (lab, r, col) in enumerate((("measured (7 May 2013)", MEASURED, "#52514e"),
                                   ("A: rigid-wing VSM aero", A, "#2a78d6"),
                                   ("B: flight-calibrated drag", B, "#eb6834"))):
    ax.bar(x + (i - 1) * 0.27, [r[k] for k in keys], width=0.25, color=col, label=lab)
ax.axhline(0, color="#52514e", lw=0.8)
ax.set_xticks(x, ["traction\n(reel-out)", "transition", "retraction\n(reel-in)", "whole cycle"])
ax.set_ylabel("mean mechanical power [kW]")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, axis="y", color="#e4e3df")
fig.tight_layout()
fig.savefig(OUT / "09_validation_2013.png", dpi=130)
print(f"saved {OUT / '09_validation_2013.png'}")
plt.show()
