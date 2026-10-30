# Third-party material

| What | From | Licence |
|---|---|---|
| `softkite/static_tether.py` (port of the quasi-steady tether model) and `tests/data/*.mat` (MATLAB reference fixtures) | [Tethers.jl](https://github.com/ufechner7/Tethers.jl), Uwe Fechner, Andrea Bertozzi, Bart van de Lint | MIT |
| `data/v3_turn_rate_coeffs.yaml` (turn-rate-law coefficients identified with V3Kite.jl) | [RobustKiteControllers.jl](https://github.com/OpenSourceAWE/RobustKiteControllers.jl) (formerly SimpleKiteControllers.jl), Uwe Fechner and contributors | MIT |
| Structure of the controllers (course loop with 1/(c1 v_a) scheduling, sqrt-force winch law with force limiters) | RobustKiteControllers.jl, WinchControllers.jl, KiteControllers.jl | MIT |
| `data/v3_aero.yaml` was computed with the [Vortex Step Method](https://github.com/awegroup/Vortex-Step-Method) on its TU Delft V3 geometry, polars and bridle files; the depower geometry follows its `examples/TUDELFT_V3_KITE/compute_trim_angle.py` | awegroup / TU Delft (Cayon, Poland et al.) | MIT |
| Depower-tape calibration `ld_2025` and V3 parameters (KCU mass, rate limits) | [V3Kite.jl](https://github.com/OpenSourceAWE/V3Kite.jl) | MPL-2.0 (numbers only) |
