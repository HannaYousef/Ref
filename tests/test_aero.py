import numpy as np

from softkite.aero import AeroModel, DepowerMap, TurnRateLaw, tape_length_2025


def test_tables_load_and_trim_monotonic():
    a = AeroModel()
    ups = np.linspace(0.2, 0.42, 12)
    trims = [a.alpha_trim(u) for u in ups]
    assert np.all(np.diff(trims) < 0), "more depower must lower the trim AoA"
    assert 7.0 < a.alpha_trim(0.22) < 10.0
    assert 1.0 < a.alpha_trim(0.40) < 3.0


def test_cm_zero_at_trim():
    a = AeroModel()
    for u in (0.22, 0.27, 0.33, 0.40):
        assert abs(a.cm(a.alpha_trim(u), u)) < 0.01
        # statically stable: CM decreases with alpha
        assert a.cm(a.alpha_trim(u) + 1, u) < a.cm(a.alpha_trim(u) - 1, u)


def test_depower_map_endpoints():
    m = DepowerMap()
    assert abs(m.delta(m.u_p_powered) - m.delta_powered) < 1e-12
    assert abs(m.delta(m.u_p_depowered) - m.delta_depowered) < 1e-12
    assert tape_length_2025(0.4) > tape_length_2025(0.2)


def test_glide_ratio_range():
    a = AeroModel()
    ld = [a.cl(a.alpha_trim(u)) / a.cd(a.alpha_trim(u)) for u in (0.22, 0.25, 0.3, 0.4)]
    assert 5.5 < ld[1] < 7.0  # V3Kite.jl: parked V3 flies at L/D_wing ~ 5.8
    assert ld[-1] < ld[0]


def test_turn_rate_law():
    t = TurnRateLaw()
    assert abs(t.c1(0.2) - 0.348278060619048) < 1e-9
    assert t.c1(0.2) > t.c1(0.3) > t.c1(0.375)
    assert t.c1(0.5) == t.c1(0.375)  # clamped outside the table
    # gravity turns the nose down: positive rate for heading to the right
    assert t.rate(0.0, 30.0, 0.5, 0.3, 0.25) > 0
