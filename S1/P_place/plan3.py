#!/usr/bin/env python3
"""DRY. Joints q so that the fixed needle-in-tool model (fit_joint) lands on the target needle arc in the camera. Search over tip angle a and heading dh. Sends nothing."""
import json, os, sys
import numpy as np, cv2
from scipy.optimize import least_squares
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", R0 + "/G_place", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K, needle_circle as NC
from rig import FX, CX, CY, FB, TCP_MM
class Model:
    def __init__(s, fit_json, key, measure_json, q_start):
        f = json.load(open(fit_json))[key]; x = np.array(f["x"]); s.x7 = x[:7]; s.dl = np.r_[0, 0, 0, x[7:10]] if len(x) > 7 else np.zeros(6); s.R = 9.96
        cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); s.M = np.array(cal["M"]); s.SUJ = np.array(cal["frozen"]["suj_psm1"]); s.q0 = np.array(q_start)
        st = json.load(open(measure_json))["stickers"]; en, ex = (st[1], st[0]) if st[1]["area"] < st[0]["area"] else (st[0], st[1]); s.e, s.ex = np.array(en["cam"]), np.array(ex["cam"]); n = np.array(en["n25"]); s.n = n if n[2] < 0 else -n
        u = (s.ex - s.e) - ((s.ex - s.e) @ s.n) * s.n; s.bite = float(np.linalg.norm(u)); s.u = u / s.bite; s.side = np.cross(s.n, s.u)
        nv = np.array([np.sin(x[3]) * np.cos(x[4]), np.sin(x[3]) * np.sin(x[4]), np.cos(x[3])]); e1, e2 = NC._basis(nv); s.span = abs(x[6]); th = x[5] + x[6] - np.sign(x[6]) * np.linspace(0, min(s.span, np.radians(90)), 13)   # index 0 = tip, going back along the arc
        s.th = np.degrees(abs(th - th[0])); s.Pt = x[:3][None] + s.R * 1e-3 * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
    def cam(s, q, Pt=None): T = K.fk_ecm(q + s.dl, s.SUJ); Pt = s.Pt if Pt is None else Pt; P = (T[:3, :3] @ Pt.T).T + T[:3, 3]; return np.c_[P * 1e3, np.ones(len(P))] @ s.M
    def target(s, a, dh, h, off=(0, 0, 0)):
        du = np.cos(np.radians(dh)) * s.u + np.sin(np.radians(dh)) * s.side; ta = np.radians(a); t = -np.cos(ta) * s.n + np.sin(ta) * du; w = np.sin(ta) * s.n + np.cos(ta) * du
        c = s.e + np.array(off) + h * s.n + s.R * w; phi = np.radians(s.th); return c[None] + s.R * (np.cos(phi)[:, None] * (-w)[None] - np.sin(phi)[:, None] * t[None])
    def solve(s, tgt, mdeg=3.0, seeds=None):
        lo = K.QMIN + np.r_[np.radians([5, 5]), 0.01, np.radians([mdeg] * 3)]; hi = K.QMAX - np.r_[np.radians([5, 5]), 0.01, np.radians([mdeg] * 3)]; best = None; w = np.r_[[3.0] * 3, [1.0] * (3 * (len(tgt) - 1))]
        S = seeds if seeds is not None else [np.clip(s.q0 + np.r_[0, 0, 0, np.radians([dr, dp, dy])], lo, hi) for dr in (0, -50, 50, -100) for dp, dy in ((0, 0), (-40, -60), (30, -60), (-40, 30))]
        for qi in S:
            r = least_squares(lambda q: np.ravel(s.cam(q) - tgt) * w, qi, bounds=(lo, hi), max_nfev=80); e = np.linalg.norm(s.cam(r.x) - tgt, axis=1)
            if best is None or e.max() < best[0]: best = (float(e.max()), r.x, e)
        return best
def clearance(md, q, disp):
    def h(X):
        u, v = FX * X[0] / X[2] + CX, FX * X[1] / X[2] + CY
        return FB / disp[int(v), int(u)] - X[2] if 0 <= u < disp.shape[1] and 0 <= v < disp.shape[0] else 1e3
    T = K.fk_ecm(q + md.dl, md.SUJ); jaw = min(h(np.r_[(T[:3, 3] + T[:3, :3] @ (TCP_MM * 1e-3 * f)) * 1e3, 1] @ md.M) for f in np.linspace(0, 1, 20)); nd = min(h(X) for X in md.cam(q)); return round(float(jaw), 1), round(float(nd), 1)
if __name__ == "__main__":
    fit_json, key, mj, statej, snap = sys.argv[1:6]; md = Model(fit_json, key, mj, json.load(open(statej))["q"]); disp = np.load(snap + "/fs_disp.npy")
    a0 = np.degrees(np.arccos(min(1, md.bite / (2 * md.R)))); print("bite %.1f a0 %.1f span used %.0f of %.0f deg; entry %s n %s u %s" % (md.bite, a0, md.th[-1], np.degrees(md.span), np.round(md.e, 1).tolist(), np.round(md.n, 3).tolist(), np.round(md.u, 3).tolist()))
    print("a dh | max needle error mm (tip) | wrist deg | margins roll p y | ins mm | clear jaw/needle at h=3")
    for a in (a0 - 15, a0, a0 + 10, a0 + 20, a0 + 30):
        for dh in (0, 15, -15, 30):
            b = md.solve(md.target(a, dh, 3.0)); mg = K.margins(b[1])
            print("%3.0f %+3d | %.2f (%.2f) | %s | %s | %.0f | %s" % (a, dh, b[0], b[2][0], np.round(np.degrees(b[1][3:]), 0).tolist(), np.round(mg[3:], 0).tolist(), b[1][2] * 1e3, clearance(md, b[1], disp)), flush=True)
