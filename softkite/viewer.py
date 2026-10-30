"""Matplotlib visualisation: 3-D replay of the flight and summary plots.

Functions
---------
plot_timeseries(log)    stacked time series with the pumping phases shaded
plot_pattern(log)       flight path in the (azimuth, elevation) plane + the target eight
plot_3d(log)            static 3-D flight path coloured by phase
replay(log, ...)        animated 3-D replay (kite, tether, trail) with live time series;
                        shown in a window or saved as MP4 (ffmpeg) or GIF (pillow)
"""

from __future__ import annotations

import math

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Categorical colours (validated reference palette, slots 1-3) + neutral for parking
PHASE_STYLE = {
    0: ("park", "#9a9893"),
    1: ("reel-out", "#2a78d6"),     # reel-out entry, drawn as reel-out
    2: ("reel-out", "#2a78d6"),
    3: ("transition", "#eb6834"),
    4: ("reel-in", "#1baf7a"),
}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)


def _phase_spans(log):
    ph = log["phase"]
    t = log["time"]
    if len(t) == 0:
        return []
    spans, start = [], 0
    for i in range(1, len(ph) + 1):
        if i == len(ph) or ph[i] != ph[start]:
            spans.append((t[start], t[i - 1], int(ph[start])))
            start = i
    return spans


def _shade(ax, spans):
    for t0, t1, p in spans:
        ax.axvspan(t0, t1, color=PHASE_STYLE.get(p, ("", "#cccccc"))[1], alpha=0.10, lw=0)


def _phase_legend(fig, loc="upper center"):
    seen, handles = set(), []
    for _, (name, col) in PHASE_STYLE.items():
        if name not in seen:
            seen.add(name)
            handles.append(Patch(facecolor=col, alpha=0.35, label=name))
    fig.legend(handles=handles, loc=loc, ncol=len(handles), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.0))


def plot_timeseries(log, path=None, t_range=None, show=False):
    t = log["time"]
    sel = slice(None) if t_range is None else (t >= t_range[0]) & (t <= t_range[1])
    t = t[sel]
    spans = _phase_spans(type(log)({k: log[k][sel] for k in ("phase", "time")}))
    panels = [
        ("l_tether", "tether length [m]", None),
        ("beta", "elevation [deg]", np.degrees),
        ("f_gnd", "tether force [kN]", lambda x: x / 1000),
        ("v_reel", "reel speed [m/s]", None),
        ("power", "mech. power [kW]", lambda x: x / 1000),
        ("v_a", "apparent wind [m/s]", None),
        ("alpha", "AoA [deg]", None),
        ("u_p", "depower [-]", None),
        ("u_s", "steering [-]", None),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 1.45 * len(panels)), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, label, fn) in zip(axes, panels):
        y = log[key][sel]
        y = fn(y) if fn else y
        _style(ax)
        _shade(ax, spans)
        ax.plot(t, y, color=INK, lw=1.2)
        if key == "v_reel":
            ax.plot(t, log["v_set"][sel], color="#2a78d6", lw=1.0, ls="--", label="set-point")
            ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax.set_ylabel(label, fontsize=8)
    axes[-1].set_xlabel("time [s]")
    _phase_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if path:
        fig.savefig(path, dpi=130, facecolor=SURFACE)
    if show:
        plt.show()
    return fig


def plot_pattern(log, fig8=None, path=None, show=False):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    az = np.degrees(log["phi"])
    el = np.degrees(log["beta"])
    ph = log["phase"]
    for p in (0, 3, 4, 1):
        m = (ph == p) | ((ph == 2) if p == 1 else False)
        if m.any():
            name, col = PHASE_STYLE[p]
            a = np.where(m, az, np.nan)
            e = np.where(m, el, np.nan)
            ax.plot(a, e, color=col, lw=1.0, label=name)
    if fig8 is not None:
        pa, pb = fig8.points()
        ax.plot(np.degrees(pa), np.degrees(pb), color=INK, lw=1.0, ls=":", label="target eight")
    ax.set_xlabel("azimuth [deg]  (positive = left, seen from the ground station)")
    ax.set_ylabel("elevation [deg]")
    ax.invert_xaxis()  # left on the left, as seen from the ground station
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, facecolor=SURFACE)
    if show:
        plt.show()
    return fig


def _setup_3d(ax, pos):
    ax.set_xlabel("x (downwind) [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    ax.set_zlabel("z [m]", fontsize=8)
    r = max(np.abs(pos).max(), 50.0)
    ax.set_xlim(0, r)
    ax.set_ylim(-r / 2, r / 2)
    ax.set_zlim(0, r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=18, azim=-140)
    ax.tick_params(labelsize=7, colors=INK2)


def plot_3d(log, path=None, show=False):
    fig = plt.figure(figsize=(7, 6))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_subplot(111, projection="3d")
    pos = log["pos"]
    ph = log["phase"]
    for p in (0, 3, 4, 1):
        m = (ph == p) | ((ph == 2) if p == 1 else False)
        if m.any():
            xyz = np.where(m[:, None], pos, np.nan)
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=PHASE_STYLE[p][1], lw=0.8,
                    label=PHASE_STYLE[p][0])
    _setup_3d(ax, pos)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, facecolor=SURFACE)
    if show:
        plt.show()
    return fig


def kite_outline(pos, x_k, y_k, e_r, span=8.3, chord=2.6, scale=1.0):
    """Arc-shaped LEI wing outline (closed polyline, 3 x N) at ``pos``."""
    span, chord = span * scale, chord * scale
    th = np.linspace(-1.1, 1.1, 15)  # arc of the leading edge
    rad = span / 2 / math.sin(1.1)
    le = [pos + rad * math.sin(a) * y_k - rad * (1 - math.cos(a)) * e_r * 0.9 + 0.35 * chord * x_k for a in th]
    te = [p - chord * (1 - 0.5 * abs(a) / 1.1) * x_k for p, a in zip(le, th)]
    pts = np.array(le + te[::-1] + [le[0]]).T
    return pts


def replay(log, speed=4.0, fps=20, tail=30.0, kite_scale=4.0, save=None, t_range=None, show=True,
           dpi=100):
    """Animated 3-D replay.

    speed       replay speed relative to real time
    tail        length of the flight-path trail [s]
    kite_scale  drawing scale of the wing (it is tiny compared to the tether)
    save        file name (.mp4 needs ffmpeg, .gif uses pillow) or None
    """
    t = log["time"]
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.02
    i0 = 0 if t_range is None else int(np.searchsorted(t, t_range[0]))
    i1 = len(t) if t_range is None else int(np.searchsorted(t, t_range[1]))
    step = max(1, int(round(speed / (fps * dt))))
    frames = list(range(i0, i1, step))
    n_tail = int(tail / dt)

    fig = plt.figure(figsize=(12, 6.5))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0.0, 0.0, 0.58, 1.0], projection="3d")
    pos = log["pos"]
    _setup_3d(ax, pos[i0:i1])
    tether_line, = ax.plot([], [], [], color=INK2, lw=1.0)
    trail, = ax.plot([], [], [], color="#2a78d6", lw=0.9, alpha=0.8)
    kite_line, = ax.plot([], [], [], color=INK, lw=1.4)
    gs_marker = ax.scatter([0], [0], [0], color=INK, s=20)
    hud = fig.text(0.02, 0.95, "", fontsize=9, family="monospace", color=INK, va="top")

    series = [("f_gnd", "force [kN]", 1e-3), ("power", "power [kW]", 1e-3), ("l_tether", "length [m]", 1.0),
              ("v_a", "app. wind [m/s]", 1.0)]
    side_axes, cursors = [], []
    spans = _phase_spans(log)
    for k, (key, label, sc) in enumerate(series):
        a = fig.add_axes([0.64, 0.08 + 0.225 * (3 - k), 0.34, 0.18])
        _style(a)
        _shade(a, spans)
        a.plot(t[i0:i1], log[key][i0:i1] * sc, color=INK, lw=0.9)
        a.set_ylabel(label, fontsize=8)
        a.set_xlim(t[i0], t[i1 - 1])
        if k < 3:
            a.set_xticklabels([])
        cursors.append(a.axvline(t[i0], color="#eb6834", lw=1.2))
        side_axes.append(a)
    side_axes[-1].set_xlabel("time [s]", fontsize=8)
    _phase_legend(fig, loc="upper center")

    from .controllers import PHASES

    def update(i):
        nodes = log["tether_nodes"][i]
        tether_line.set_data(nodes[0], nodes[1])
        tether_line.set_3d_properties(nodes[2])
        j0 = max(i0, i - n_tail)
        trail.set_data(pos[j0:i + 1, 0], pos[j0:i + 1, 1])
        trail.set_3d_properties(pos[j0:i + 1, 2])
        pts = kite_outline(pos[i], log["x_k"][i], log["y_k"][i], log["e_r"][i], scale=kite_scale)
        kite_line.set_data(pts[0], pts[1])
        kite_line.set_3d_properties(pts[2])
        ph = int(log["phase"][i])
        hud.set_text(f"t = {t[i]:6.1f} s   {PHASES[ph] if 0 <= ph < len(PHASES) else '':15s}\n"
                     f"F = {log['f_gnd'][i] / 1000:5.2f} kN  P = {log['power'][i] / 1000:6.2f} kW\n"
                     f"l = {log['l_tether'][i]:6.1f} m   v_reel = {log['v_reel'][i]:5.2f} m/s\n"
                     f"el = {math.degrees(log['beta'][i]):5.1f} deg  az = {math.degrees(log['phi'][i]):6.1f} deg\n"
                     f"AoA = {log['alpha'][i]:4.1f} deg  v_a = {log['v_a'][i]:5.1f} m/s")
        for c in cursors:
            c.set_xdata([t[i], t[i]])
        return [tether_line, trail, kite_line, hud] + cursors

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    if save:
        if str(save).endswith(".gif"):
            anim.save(save, writer="pillow", fps=fps, dpi=dpi)
        else:
            anim.save(save, writer="ffmpeg", fps=fps, dpi=dpi, bitrate=2400)
    if show:
        plt.show()
    return anim
