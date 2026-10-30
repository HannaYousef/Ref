"""Full pumping cycles: reel-out in figures of eight, retraction, reel-in, repeat.

Run from the repository root:  python examples_simple/05_pumping_cycle.py [sim_time] [v_wind]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from softkite import Simulation, cycle_stats, load_settings
from softkite.viewer import plot_3d, plot_pattern, plot_timeseries

OUT = Path(__file__).resolve().parents[1] / "output"
OUT.mkdir(exist_ok=True)

sim_time = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
overrides = {"environment__v_wind_ref": float(sys.argv[2])} if len(sys.argv) > 2 else {}
s = load_settings(**overrides)
sim = Simulation(s)
log = sim.run(sim_time)
log.save(OUT / "pumping.npz")

print("\ncycle  duration  reel-out  transition  reel-in   E_out    E_in   mean P   F_max")
print("           [s]      [s]       [s]        [s]     [kJ]    [kJ]    [kW]     [N]")
for c in cycle_stats(log):
    print(f"{c['cycle']:4d} {c['duration']:9.1f} {c['t_reel_out'] + c['t_reel_out_entry']:9.1f} "
          f"{c['t_retract']:10.1f} {c['t_reel_in']:9.1f} {c['energy_out'] / 1e3:8.0f} {c['energy_in'] / 1e3:7.0f} "
          f"{c['mean_power'] / 1e3:8.2f} {c['f_max']:7.0f}")

plot_timeseries(log, OUT / "05_pumping_timeseries.png")
plot_pattern(log, sim.controller.fig8, OUT / "05_pumping_pattern.png")
plot_3d(log, OUT / "05_pumping_3d.png")
print(f"\nlog: {OUT / 'pumping.npz'}  -  replay it with examples_simple/06_replay.py")
plt.show()
