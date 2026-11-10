"""PSM1 kinematics in the ECM frame for pre-send joint-limit checks (pure functions, no ROS).

T_ecm(q) = T_base_ECM (from /SUJ/PSM1/measured_cp) @ FK(q), FK = dVRK modified DH from the config files
(kinematic/PSM.json + PROGRASP_FORCEPS_400093.json incl. tooltip_offset), verified against /PSM1/measured_cp
to 0.0006 mm (step1e).  IK = bounded least squares from the current joints, position + orientation.
"""
import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261025/step1e_fk_vs_cp"))
import fk_vs_cp as K  # noqa: E402

TOOL_FILE = "PROGRASP_FORCEPS_400093.json"
_arm = K.load_json(K.CFG + "kinematic/PSM.json")
_tool = K.load_json(K.CFG + "tool/" + TOOL_FILE)
JOINTS = _arm["DH"]["joints"] + _tool["DH"]["joints"]
TOOLTIP = np.array(_tool["tooltip_offset"], float)
QMIN = np.array([j["qmin"] for j in JOINTS])
QMAX = np.array([j["qmax"] for j in JOINTS])
NAMES = ["outer_yaw", "outer_pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw"]


def fk_ecm(q, T_base):
    return T_base @ K.fk(q, JOINTS, TOOLTIP)


def ik_ecm(p_m, R, q0, T_base):
    """joints reaching (p_m, R) in ECM, starting from q0; returns (q, pos_err_mm, rot_err_deg)."""
    def res(q):
        T = fk_ecm(q, T_base)
        return np.r_[(T[:3, 3] - p_m) * 1e3, Rotation.from_matrix(R.T @ T[:3, :3]).as_rotvec() * 57.29578 * 0.2]
    lo, hi = QMIN - 1e-6, QMAX + 1e-6
    sol = least_squares(res, np.clip(q0, lo, hi), bounds=(lo, hi), xtol=1e-12, ftol=1e-12)
    T = fk_ecm(sol.x, T_base)
    return sol.x, float(np.linalg.norm(T[:3, 3] - p_m) * 1e3), float(np.degrees(Rotation.from_matrix(R.T @ T[:3, :3]).magnitude()))


def margins(q):
    """per-joint distance to the nearer limit: degrees for revolute, mm for insertion."""
    m = np.minimum(q - QMIN, QMAX - q)
    return np.array([m[i] * 1e3 if i == 2 else np.degrees(m[i]) for i in range(6)])


def check_pose(p_m, R, q0, T_base, min_deg=3.0, min_mm=5.0):
    """(ok, q, reason): reachable within 0.05 mm / 0.05 deg with every joint margin >= min_deg / min_mm."""
    q, ep, er = ik_ecm(p_m, R, q0, T_base)
    if ep > 0.05 or er > 0.05:
        return False, q, f"IK residual {ep:.3f} mm / {er:.3f} deg (limit-bound)"
    m = margins(q)
    bad = [f"{NAMES[i]} margin {m[i]:.1f}{' mm' if i == 2 else ' deg'}" for i in range(6)
           if m[i] < (min_mm if i == 2 else min_deg)]
    return (not bad), q, "; ".join(bad)


def yaw_gradient(q, T_base):
    """d(outer yaw)/d(position) [rad per m] at fixed orientation, ECM frame (numerical)."""
    T0 = fk_ecm(q, T_base)
    g = np.zeros(3)
    for i in range(3):
        p = T0[:3, 3].copy(); p[i] += 1e-4
        qi, _, _ = ik_ecm(p, T0[:3, :3], q, T_base)
        g[i] = (qi[0] - q[0]) / 1e-4
    return g
