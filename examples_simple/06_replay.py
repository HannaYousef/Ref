"""3-D replay of a logged run (default: output/pumping.npz from 05_pumping_cycle.py).

    python examples_simple/06_replay.py                       # window
    python examples_simple/06_replay.py output/pumping.npz --save output/pumping.mp4
    python examples_simple/06_replay.py --speed 2 --t0 30 --t1 90
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from softkite import SimLog
from softkite.viewer import replay

OUT = Path(__file__).resolve().parents[1] / "output"
ap = argparse.ArgumentParser()
ap.add_argument("log", nargs="?", default=str(OUT / "pumping.npz"))
ap.add_argument("--speed", type=float, default=4.0, help="replay speed vs real time")
ap.add_argument("--fps", type=int, default=20)
ap.add_argument("--t0", type=float, default=None)
ap.add_argument("--t1", type=float, default=None)
ap.add_argument("--save", default=None, help=".mp4 (needs ffmpeg) or .gif")
ap.add_argument("--kite-scale", type=float, default=4.0, help="drawing scale of the wing")
args = ap.parse_args()

if args.save:
    import matplotlib

    matplotlib.use("Agg")
log = SimLog.load(args.log)
t = log["time"]
t_range = (args.t0 if args.t0 is not None else t[0], args.t1 if args.t1 is not None else t[-1])
anim = replay(log, speed=args.speed, fps=args.fps, t_range=t_range, save=args.save,
              show=args.save is None, kite_scale=args.kite_scale)
if args.save:
    print(f"saved {args.save}")
