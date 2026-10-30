"""Settings of the reference model, loaded from ``softkite/data/settings.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

import yaml

DATA = Path(__file__).resolve().parent / "data"


@dataclass
class Environment:
    v_wind_ref: float = 9.51
    h_ref: float = 6.0
    profile_exp: float = 0.08163
    profile_law: str = "power"
    z0: float = 0.07
    wind_dir: float = 0.0
    rho: float = 1.225
    g: float = 9.81


@dataclass
class Kite:
    aero_file: str = "v3_aero.yaml"
    turn_rate_file: str = "v3_turn_rate_coeffs.yaml"
    m_wing: float = 10.6
    m_kcu: float = 8.4
    cd_kcu_area: float = 0.059
    cd_extra: float = 0.0
    u_p_powered: float = 0.22
    u_p_depowered: float = 0.40


@dataclass
class KCU:
    v_depower: float = 0.075
    v_steering: float = 2.0
    max_steering: float = 0.32
    use_delay: bool = True


@dataclass
class Tether:
    segments: int = 10
    d_tether: float = 4.0
    cd_tether: float = 0.958
    rho_tether: float = 724.0
    c_spring: float = 614600.0


@dataclass
class Winch:
    m_eff: float = 120.0
    tau_speed: float = 0.1
    f_motor_max: float = 5000.0
    v_ro_max: float = 8.0
    v_ri_max: float = 8.0
    a_max: float = 4.0


@dataclass
class Simulation:
    dt: float = 0.02
    substeps: int = 2
    l_tether0: float = 200.0
    elevation0: float = 70.0
    azimuth0: float = 0.0
    heading0: float = 0.0
    depower0: float = 0.40
    sim_time: float = 400.0


@dataclass
class Control:
    k_psi: float = 2.5
    k_psi_d: float = 0.0
    v_heading: float = 5.0
    v_course: float = 10.0
    f8_azimuth: float = 25.0
    f8_height: float = 9.0
    f8_elevation: float = 28.0
    f8_lookahead: float = 5.0
    f8_up_turns: bool = True
    u_p_ro: float = 0.25
    u_p_ri: float = 0.40
    u_p_tr: float = 0.30
    u_p_f8: float = 0.35
    ri_elevation: float = 70.0
    tr_elevation: float = 55.0
    tr_time_max: float = 12.0
    ro_exit_wait: float = 0.0
    ri_beta_max: float = 80.0
    ri_beta_band: float = 10.0
    k_azimuth: float = 1.5
    l_min: float = 200.0
    l_max: float = 350.0
    f_ro_factor: float = 0.333
    ro_mode: str = "sqrt"
    f_ro: float = 3000.0
    f_max: float = 4000.0
    f_min: float = 300.0
    f_ri: float = 1200.0
    v_ri: float = 7.0
    kp_force: float = 0.002
    ki_force: float = 0.004


@dataclass
class Settings:
    environment: Environment = field(default_factory=Environment)
    kite: Kite = field(default_factory=Kite)
    kcu: KCU = field(default_factory=KCU)
    tether: Tether = field(default_factory=Tether)
    winch: Winch = field(default_factory=Winch)
    simulation: Simulation = field(default_factory=Simulation)
    control: Control = field(default_factory=Control)

    def data_file(self, name: str) -> Path:
        p = Path(name)
        return p if p.is_absolute() else DATA / p

    def to_dict(self):
        return asdict(self)


def load_settings(path: str | Path | None = None, **overrides) -> Settings:
    """Load settings from YAML. ``overrides`` use ``section__key=value``, e.g.
    ``load_settings(environment__v_wind_ref=7.0, control__l_max=400)``."""
    path = Path(path) if path is not None else DATA / "settings.yaml"
    raw = yaml.safe_load(path.read_text()) or {}
    s = Settings()
    for sec in fields(Settings):
        obj = getattr(s, sec.name)
        known = {f.name for f in fields(obj)}
        for k, v in (raw.get(sec.name) or {}).items():
            if k not in known:
                raise KeyError(f"unknown setting {sec.name}.{k} in {path}")
            setattr(obj, k, v)
    for key, v in overrides.items():
        sec, _, name = key.partition("__")
        obj = getattr(s, sec)
        if not hasattr(obj, name):
            raise KeyError(f"unknown setting {sec}.{name}")
        setattr(obj, name, v)
    return s
