"""Park the depowered kite near zenith at constant tether length.

Run from the repository root:  python examples_simple/03_parking.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from softkite import Simulation, load_settings
from softkite.controllers import ParkingController
from softkite.viewer import plot_timeseries

OUT = Path(__file__).resolve().parents[1] / "output"
OUT.mkdir(exist_ok=True)

s = load_settings(simulation__azimuth0=15.0)  # start 15 deg off-centre
log = Simulation(s, ParkingController, depower=0.30).run(60.0)
log.save(OUT / "parking.npz")
plot_timeseries(log, OUT / "03_parking.png")
print(f"saved {OUT / '03_parking.png'}")
plt.show()
