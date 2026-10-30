"""softkite - a simple, fast-to-install reference model of a soft-wing (LEI) kite.

Modules
-------
static_tether   quasi-steady tether model (port of Tethers.jl QuasiSteady)
aero            CL, CD, CS, CM tables vs angle of attack, depower trim, turn-rate law
kite_model      rigid-wing point-mass kite + tether + winch + KCU
controllers     course loop, figure-of-eight guidance, winch control, pumping cycle
simulator       run loop and logging
viewer          matplotlib 3-D replay and plots
"""

from .settings import Settings, load_settings
from .static_tether import StaticSettings, StaticTether
from .aero import AeroModel, TurnRateLaw, DepowerMap
from .kite_model import KiteModel
from .simulator import Simulation, SimLog, run, cycle_stats

__version__ = "0.1.1"
