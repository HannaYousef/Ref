import numpy as np

from softkite import Simulation, cycle_stats, load_settings
from softkite.controllers import Fig8Controller


def test_pumping_cycle_runs_and_produces_energy():
    s = load_settings()
    log = Simulation(s).run(220, verbose=False)
    assert not log.meta["crashed"]
    st = cycle_stats(log)
    assert len(st) >= 1
    for c in st:
        assert c["energy"] > 0
        assert c["mean_power"] > 2000
        assert c["f_max"] < 1.35 * s.control.f_max
    assert log["l_tether"].max() < s.control.l_max + 80
    assert log["height"][log["time"] > 5].min() > 20.0


def test_fig8_constant_length():
    s = load_settings(simulation__l_tether0=150.0, simulation__elevation0=60.0)
    log = Simulation(s, Fig8Controller).run(40, verbose=False)
    assert not log.meta["crashed"]
    assert np.ptp(log["l_tether"]) < 1.0
    assert np.ptp(np.degrees(log["phi"][log["time"] > 20])) > 20  # it flies the eight
