"""Identify CL, CD, CS and CM(alpha, depower) of the TU Delft V3 kite with the Python
Vortex Step Method (VSM, awegroup/Vortex-Step-Method) and write ``softkite/data/v3_aero.yaml``.

This is the Python counterpart of "identify CL, CD, CM as a function of the mean AoA
based on V3Kite.jl": V3Kite.jl computes its aerodynamics with VortexStepMethod.jl on the
same V3 geometry and 2-D polars, so running the Python VSM on that geometry gives the
rigid-wing (CAD shape) coefficients without needing Julia.  The deformation that the
full V3Kite.jl model adds is NOT included; use ``scripts/v3kite_extract_coeffs.jl`` +
``scripts/fit_v3kite_logs.py`` on a machine with V3Kite.jl to replace these tables with
coefficients identified from the full model.

What is computed
----------------
* CL, CD, CS vs alpha for the wing alone and for wing + bridle lines (VSM line drag).
* CMy about the bridle point (KCU) for a set of depower states.  Depower is modelled as
  in VSM's ``examples/TUDELFT_V3_KITE/compute_trim_angle.py``: the KCU is the
  intersection of two circles around the mid-span leading and trailing edge, with radii
  ``L_b`` (front) and ``L_t`` (rear).  Depower shortens ``L_b`` by ``delta`` relative to
  the CAD geometry (delta = -0.036 m powered, +0.66 m depowered in that example).
* The trim angle alpha_trim(delta) where CMy(KCU) = 0 with dCMy/dalpha < 0.
* The side-force derivative CS_beta at the powered trim angle.

Usage::

    pip install "VSM @ git+https://github.com/awegroup/Vortex-Step-Method"
    python scripts/identify_aero_vsm.py            # writes softkite/data/v3_aero.yaml
    python scripts/identify_aero_vsm.py --vsm-data /path/to/Vortex-Step-Method/data

Runtime: about one minute.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Mid-span leading/trailing edge of the CAD geometry (VSM compute_trim_angle.py)
MID_LE = np.array([-1.1557911914860979, 0.0, 11.00491618799836])
MID_TE = np.array([1.443146003226444, 0.0, 11.004972573823276])
L_B0 = float(np.linalg.norm(MID_LE))  # CAD distance KCU -> mid-span LE
L_T0 = float(np.linalg.norm(MID_TE))  # CAD distance KCU -> mid-span TE


def find_vsm_data(user_path: str | None) -> Path:
    cands = []
    if user_path:
        cands.append(Path(user_path))
    try:
        import VSM  # noqa: F401

        pkg = Path(VSM.__file__).resolve()
        cands += [pkg.parents[2] / "data", pkg.parents[1] / "data", pkg.parent / "data"]
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("The VSM package is required: pip install "
                         "'VSM @ git+https://github.com/awegroup/Vortex-Step-Method'") from exc
    for c in cands:
        d = c / "TUDELFT_V3_KITE" / "CAD_derived_geometry" if (c / "TUDELFT_V3_KITE").exists() else c
        if (d / "aero_geometry_CAD_CFD_polars.yaml").exists():
            return d
    raise SystemExit("Could not find the VSM V3 data folder; pass --vsm-data "
                     "<Vortex-Step-Method>/data")


def kcu_position(delta: float) -> np.ndarray:
    """KCU position for a front-line shortening ``delta`` [m] (x-z plane, lower solution)."""
    r0 = L_B0 - delta
    r1 = L_T0
    x0, z0 = MID_LE[0], MID_LE[2]
    x1, z1 = MID_TE[0], MID_TE[2]
    dx, dz = x1 - x0, z1 - z0
    d = math.hypot(dx, dz)
    a = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
    h = math.sqrt(max(r0 * r0 - a * a, 0.0))
    xm, zm = x0 + a * dx / d, z0 + a * dz / d
    p3 = np.array([xm - dz * h / d, 0.0, zm + dx * h / d])
    p4 = np.array([xm + dz * h / d, 0.0, zm - dx * h / d])
    return p3 if p3[2] < p4[2] else p4


def patched_struc_yaml(src: Path, kcu: np.ndarray, out_dir: Path) -> Path:
    """Copy of the bridle geometry with the KCU moved and the 'diameter' header renamed
    to 'd', which is what BodyAerodynamics.instantiate expects."""
    data = yaml.safe_load(src.read_text())
    data["bridle_point_node"] = [float(v) for v in kcu]
    hdr = data["bridle_lines"]["headers"]
    data["bridle_lines"]["headers"] = ["d" if h == "diameter" else h for h in hdr]
    out = out_dir / f"struc_{kcu[0]:+.4f}_{kcu[2]:+.4f}.yaml"
    out.write_text(yaml.safe_dump(data))
    return out


def solve(body, solver, alpha, beta, umag, ref):
    body.va_initialize(Umag=umag, angle_of_attack=alpha, side_slip=beta, reference_point=ref)
    return solver.solve(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vsm-data", default=None)
    ap.add_argument("--n-panels", type=int, default=40)
    ap.add_argument("--umag", type=float, default=20.0)
    ap.add_argument("--out", default=str(ROOT / "softkite" / "data" / "v3_aero.yaml"))
    args = ap.parse_args()

    from VSM.core.BodyAerodynamics import BodyAerodynamics
    from VSM.core.Solver import Solver

    cad = find_vsm_data(args.vsm_data)
    geom = cad / "aero_geometry_CAD_CFD_polars.yaml"
    struc = cad / "struc_geometry_manually_adjusted.yaml"
    print(f"VSM data: {cad}")

    alphas = np.arange(-6.0, 20.0 + 1e-9, 1.0)
    deltas = np.round(np.linspace(-0.10, 0.80, 10), 4)  # covers -0.036 (powered) .. 0.66 (depowered)
    c_ref = float(np.linalg.norm(MID_TE - MID_LE))

    tmp = Path(tempfile.mkdtemp())
    out = {}

    # ---- wing alone ---------------------------------------------------------------
    wing = BodyAerodynamics.instantiate(n_panels=args.n_panels, file_path=geom,
                                        spanwise_panel_distribution="uniform")
    solver = Solver(reference_point=np.zeros(3))
    cl_w, cd_w, cs_w = [], [], []
    for a in alphas:
        r = solve(wing, solver, a, 0.0, args.umag, np.zeros(3))
        cl_w.append(r["cl"]), cd_w.append(r["cd"]), cs_w.append(r["cs"])
    S_ref = float(r["projected_area"])
    span = float(r["wing_span"])
    print(f"projected area {S_ref:.3f} m2, span {span:.3f} m, c_ref {c_ref:.3f} m")

    # ---- wing + bridles, CM about the KCU for each depower state -------------------
    cm = np.zeros((len(deltas), len(alphas)))
    cl_b = np.zeros_like(cm)
    cd_b = np.zeros_like(cm)
    kcu_xz = []
    trims = []
    for i, dl in enumerate(deltas):
        kcu = kcu_position(dl)
        kcu_xz.append([float(kcu[0]), float(kcu[2])])
        body = BodyAerodynamics.instantiate(n_panels=args.n_panels, file_path=geom,
                                            spanwise_panel_distribution="uniform",
                                            bridle_path=patched_struc_yaml(struc, kcu, tmp))
        solver = Solver(reference_point=kcu)
        for j, a in enumerate(alphas):
            r = solve(body, solver, a, 0.0, args.umag, kcu)
            q = r["q_ref"]
            cl_b[i, j], cd_b[i, j] = r["cl"], r["cd"]
            cm[i, j] = r["My"] / (q * S_ref * c_ref)
        # trim: first stable zero crossing of CMy
        a_trim = float("nan")
        for j in range(len(alphas) - 1):
            if cm[i, j] > 0 >= cm[i, j + 1]:
                t = cm[i, j] / (cm[i, j] - cm[i, j + 1])
                a_trim = float(alphas[j] + t * (alphas[j + 1] - alphas[j]))
                break
        trims.append(a_trim)
        print(f"delta = {dl:+.3f} m  KCU = ({kcu[0]:+.3f}, {kcu[2]:+.3f}) m  alpha_trim = {a_trim:6.2f} deg")

    # bridle drag is (nearly) independent of the KCU position: use the CAD (delta=0) row
    i0 = int(np.argmin(np.abs(deltas)))

    # ---- side force derivative at the powered trim -----------------------------------
    kcu = kcu_position(-0.036)
    body = BodyAerodynamics.instantiate(n_panels=args.n_panels, file_path=geom,
                                        spanwise_panel_distribution="uniform",
                                        bridle_path=patched_struc_yaml(struc, kcu, tmp))
    solver = Solver(reference_point=kcu)
    a_pow = float(np.interp(-0.036, deltas, trims))
    betas = np.array([-6.0, -3.0, 0.0, 3.0, 6.0])
    cs_beta = []
    for b in betas:
        r = solve(body, solver, a_pow, b, args.umag, kcu)
        cs_beta.append(r["cs"])
    cs_slope = float(np.polyfit(np.radians(betas), cs_beta, 1)[0])
    print(f"CS_beta at alpha = {a_pow:.2f} deg: {cs_slope:+.3f} 1/rad")

    out = {
        "provenance": {
            "generated_by": "scripts/identify_aero_vsm.py",
            "date": _dt.date.today().isoformat(),
            "method": "Python Vortex Step Method (awegroup/Vortex-Step-Method), "
                      "TU Delft V3 CAD geometry with 2-D CFD polars "
                      "(aero_geometry_CAD_CFD_polars.yaml) and bridle line drag "
                      "(struc_geometry_manually_adjusted.yaml). Rigid wing, no deformation.",
            "n_panels": args.n_panels,
            "v_app": args.umag,
        },
        "reference": {
            "area": round(S_ref, 4),
            "chord": round(c_ref, 4),
            "span": round(span, 4),
            "alpha_definition": "angle between apparent wind and the CAD body x axis "
                                "(mid-span chord), positive nose up [deg]",
            "moment_reference": "bridle point (KCU), CM = My / (q S c)",
        },
        "alpha_deg": alphas.round(3).tolist(),
        "cl": cl_b[i0].round(5).tolist(),
        "cd": cd_b[i0].round(5).tolist(),
        "cl_wing_only": np.round(cl_w, 5).tolist(),
        "cd_wing_only": np.round(cd_w, 5).tolist(),
        "cs_beta": round(cs_slope, 4),
        "depower": {
            "description": "delta = shortening of the front bridle (power) lines relative "
                           "to the CAD geometry [m]; delta > 0 depowers",
            "delta_m": deltas.tolist(),
            "kcu_xz": np.round(kcu_xz, 4).tolist(),
            "alpha_trim_deg": np.round(trims, 3).tolist(),
            "cm": cm.round(5).tolist(),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Generated by scripts/identify_aero_vsm.py - do not edit by hand.\n")
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None, width=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
