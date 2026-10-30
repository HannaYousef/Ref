"""Figures of eight at constant tether length (150 m).

Run from the repository root:  python examples_simple/04_figure_eight.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from softkite import Simulation, load_settings
from softkite.controllers import Fig8Controller
from softkite.viewer import plot_pattern, plot_timeseries

OUT = Path(__file__).resolve().parents[1] / "output"
OUT.mkdir(exist_ok=True)

s = load_settings(simulation__l_tether0=150.0, simulation__elevation0=60.0)
sim = Simulation(s, Fig8Controller)  # depower: control.u_p_f8
log = sim.run(90.0)
log.save(OUT / "fig8.npz")
plot_timeseries(log, OUT / "04_fig8_timeseries.png")
plot_pattern(log, sim.controller.fig8, OUT / "04_fig8_pattern.png")
print(f"saved plots in {OUT}")
plt.show()
