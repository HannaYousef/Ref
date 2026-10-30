"""Figure-of-eight path following on the wind-window sphere.

The path is a lemniscate of Gerono in (azimuth, elevation)::

    phi(s)  = phi_c + A * sin(s)
    beta(s) = beta_c + H * sin(2 s)

with azimuth positive to the left (seen from the ground station).  The kite flies along
increasing ``s`` with down-turns at the sides, or decreasing ``s`` (``up_turns=True``,
default) with up-turns at the sides and a diagonal descent through the centre, which
keeps the force peaks low.

Guidance is "carrot chasing": find the closest point of the path, move a look-ahead
distance along it, and command the course that points at that carrot.  The course rate
of the path at the carrot is fed forward to the course controller.
"""

from __future__ import annotations

import math

import numpy as np


class FigureEight:
    def __init__(self, azimuth_width=25.0, height=9.0, elevation=28.0, lookahead=7.0,
                 up_turns=True, azimuth_center=0.0, ff_gain=0.8):
        self.A = math.radians(azimuth_width)
        self.H = math.radians(height)
        self.beta_c = math.radians(elevation)
        self.phi_c = math.radians(azimuth_center)
        self.lookahead = math.radians(lookahead)
        self.dir = -1.0 if up_turns else 1.0
        self.ff_gain = ff_gain
        self.s = None
        self.cross_track = 0.0
        self.target = (0.0, 0.0)
        self._grid = np.linspace(0, 2 * math.pi, 721)[:-1]

    # path geometry ------------------------------------------------------------------
    def point(self, s):
        return self.phi_c + self.A * np.sin(s), self.beta_c + self.H * np.sin(2 * s)

    def tangent(self, s):
        """(d_up, d_right) unit tangent of the path in the direction of flight."""
        dphi = self.A * math.cos(s) * self.dir
        dbeta = 2 * self.H * math.cos(2 * s) * self.dir
        beta = self.beta_c + self.H * math.sin(2 * s)
        up, right = dbeta, -dphi * math.cos(beta)
        n = math.hypot(up, right)
        return up / n, right / n

    def path_course(self, s):
        up, right = self.tangent(s)
        return math.atan2(right, up)

    def points(self, n=400):
        s = np.linspace(0, 2 * math.pi, n)
        return self.point(s)

    # guidance -------------------------------------------------------------------------
    def _closest(self, phi, beta):
        if self.s is None:
            cand = self._grid
        else:
            cand = self.s + self.dir * np.linspace(-0.3, 0.6, 91)
        p, b = self.point(cand)
        d2 = ((p - phi) * math.cos(beta)) ** 2 + (b - beta) ** 2
        i = int(np.argmin(d2))
        return float(cand[i] % (2 * math.pi)), math.sqrt(d2[i])

    def _advance(self, s, dist):
        """Move ``dist`` [rad of arc] along the path from ``s`` in the flight direction."""
        acc, ds = 0.0, 0.01 * self.dir
        p0, b0 = self.point(s)
        while acc < dist:
            s += ds
            p1, b1 = self.point(s)
            acc += math.hypot((p1 - p0) * math.cos(b1), b1 - b0)
            p0, b0 = p1, b1
        return s

    def guidance(self, obs: dict):
        """Return (chi_set [rad], course-rate feed-forward [rad/s])."""
        phi, beta = obs["phi"], obs["beta"]
        s, d = self._closest(phi, beta)
        self.s = s
        self.cross_track = d
        # longer look-ahead while far from the path (attractor behaviour)
        s_t = self._advance(s, max(self.lookahead, 0.7 * d))
        pt, bt = self.point(s_t)
        self.target = (float(pt), float(bt))
        d_up = bt - beta
        d_right = -(pt - phi) * math.cos(beta)
        chi_set = math.atan2(d_right, d_up)
        # feed-forward: course rate of the path near the carrot
        ds = 0.02 * self.dir
        dchi = (self.path_course(s_t + ds) - self.path_course(s_t - ds))
        dchi = (dchi + math.pi) % (2 * math.pi) - math.pi
        pa, ba = self.point(s_t - ds)
        pb, bb = self.point(s_t + ds)
        arc = math.hypot((pb - pa) * math.cos(bt), bb - ba)
        omega = obs["v_tangential"] / max(obs["distance"], 1.0)  # angular speed [rad/s]
        ff = self.ff_gain * dchi / max(arc, 1e-6) * omega if d < 2 * self.lookahead else 0.0
        return chi_set, ff
