#!/usr/bin/env python3
"""DRY. Fixed needle-in-tool transform -> tool pose for the target needle pose -> IK, margins, clearances; overlay of the target on the right image. Sends nothing."""
import json, os, sys
import numpy as np, cv2
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K
from calib import rotation_from_affine, cam2ecm_mm
from rig import FX, CX, CY, FB, TCP_MM
def load(mdir, demo):
    arc = json.load(open(mdir + "/arcfit.json")); st = json.load(open(mdir + "/measure.json"))["stickers"]; dem = json.load(open(demo))
    return arc, st, dem
class Plan:
    def __init__(s, arc, st, dem, snap):
        cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); s.M = np.array(cal["M"]); s.Rc = rotation_from_affine(s.M); s.SUJ = np.array(cal["frozen"]["suj_psm1"]); s.q0 = np.array(dem["js_mean"]); s.T0 = K.fk_ecm(s.q0, s.SUJ)
        C, tip, tan = map(np.array, (arc["centre"], arc["tip_cam"], arc["dir_cam"])); s.Rn = arc["radius_mm"]; s.span = arc["span_deg"]
        x = (tip - C) / np.linalg.norm(tip - C); z = np.cross(x, tan); z /= np.linalg.norm(z); s.Rtn = s.T0[:3, :3].T @ s.Rc.T @ np.c_[x, np.cross(z, x), z]; s.ptn = s.T0[:3, :3].T @ (cam2ecm_mm(C, s.M) * 1e-3 - s.T0[:3, 3])
        en, ex = (st[1], st[0]) if st[1]["area"] < st[0]["area"] else (st[0], st[1]); s.e, s.x = np.array(en["cam"]), np.array(ex["cam"]); n = np.array(en["n25"]); s.n = n if n[2] < 0 else -n; s.nloc = np.array(en["n4"])
        u = (s.x - s.e) - ((s.x - s.e) @ s.n) * s.n; s.bite = float(np.linalg.norm(u)); s.u = u / s.bite; s.side = np.cross(s.n, s.u); s.disp = np.load(snap + "/fs_disp.npy")
    def nd_pts(s, T, k=25):
        """needle points in camera mm for tool pose T, tip first."""
        th = -np.radians(np.linspace(0, s.span, k)); P = s.ptn[None] + s.Rn * 1e-3 * (np.cos(th)[:, None] * s.Rtn[:, 0][None] + np.sin(th)[:, None] * s.Rtn[:, 1][None])
        Q = (T[:3, :3] @ P.T).T + T[:3, 3]; return np.c_[Q * 1e3, np.ones(k)] @ s.M
    def target(s, a, dh, h):
        du = np.cos(np.radians(dh)) * s.u + np.sin(np.radians(dh)) * s.side; ta = np.radians(a); t = -np.cos(ta) * s.n + np.sin(ta) * du; w = np.sin(ta) * s.n + np.cos(ta) * du
        tipc = s.e + h * s.n; xs = -w; zs = np.cross(xs, t); zs /= np.linalg.norm(zs); return np.c_[xs, np.cross(zs, xs), zs], tipc + s.Rn * w
    def tool(s, Rs, Cs):
        R = s.Rc.T @ Rs @ s.Rtn.T; T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cam2ecm_mm(Cs, s.M) * 1e-3 - R @ s.ptn; return T
    def ik(s, T, qs=None):
        best = None; lo, hi = K.QMIN + 1e-4, K.QMAX - 1e-4
        for qi in (qs if qs is not None else [np.clip(s.q0 + np.r_[0, 0, 0, np.radians([dr, dp, dy])], lo, hi) for dr in (0, 60, -60, 120, -120) for dp, dy in ((0, 0), (-40, -40), (40, -40), (-40, 40))]):
            q, ep, er = K.ik_ecm(T[:3, 3], T[:3, :3], qi, s.SUJ)
            if ep > 0.05 or er > 0.05: continue
            mg = K.margins(q); sc = float(np.min(np.r_[mg[:2], mg[2] / 2, mg[3:]]))
            if best is None or sc > best[0]: best = (sc, q, mg)
        return best
    def clear(s, T):
        """heights above the FS surface along the camera axis (mm): jaw hinge->TCP, needle; FS pixels off-image are skipped."""
        def h(X):
            u, v = FX * X[0] / X[2] + CX, FX * X[1] / X[2] + CY
            return FB / s.disp[int(v), int(u)] - X[2] if 0 <= u < s.disp.shape[1] and 0 <= v < s.disp.shape[0] else 1e3
        jaw = min(h(np.r_[(T[:3, 3] + T[:3, :3] @ (TCP_MM * 1e-3 * f)) * 1e3, 1] @ s.M) for f in np.linspace(0, 1, 25)); nd = min(h(X) for X in s.nd_pts(T)); return round(float(jaw), 1), round(float(nd), 1)
if __name__ == "__main__":
    mdir, demo, snap = sys.argv[1:4]; arc, st, dem = load(mdir, demo); p = Plan(arc, st, dem, snap)
    tail_tool = (lambda th: p.ptn + p.Rn * 1e-3 * (np.cos(th) * p.Rtn[:, 0] + np.sin(th) * p.Rtn[:, 1]))(-np.radians(p.span)); tip_tool = p.ptn + p.Rn * 1e-3 * p.Rtn[:, 0]
    print("radius %.2f span %.1f  needle in tool mm: tip %s tail %s centre %s normal %s; TCP %s; tail->TCP %.1f" % (p.Rn, p.span, np.round(tip_tool * 1e3, 1).tolist(), np.round(tail_tool * 1e3, 1).tolist(), np.round(p.ptn * 1e3, 1).tolist(), np.round(p.Rtn[:, 2], 3).tolist(), TCP_MM.tolist(), np.linalg.norm(tail_tool * 1e3 - TCP_MM)))
    a0 = np.degrees(np.arccos(min(1, p.bite / (2 * p.Rn)))); a_loc = np.degrees(np.arccos(np.clip(-p.nloc @ -p.n, -1, 1)))
    print("bite %.1f; angle with circle through both stickers %.1f deg; local entry normal is %.1f deg from n, %.1f deg out of the plane" % (p.bite, a0, a_loc, np.degrees(np.arcsin(abs(p.nloc @ p.side)))))
    print("a dh h | score margins(yaw pit ins roll wp wy) | wrist deg | ins mm | clear jaw/needle")
    rows = []
    for a in (a0 - 10, a0 - 5, a0, a0 + 5, a0 + 10):
        for dh in (0, 5, 10, 15, 20, -5, -10):
            T = p.tool(*p.target(a, dh, 3.0)); b = p.ik(T)
            if b is None: print("%.0f %+d none" % (a, dh)); continue
            print("%.0f %+d  %.1f %s  %s  %.0f  %s" % (a, dh, b[0], np.round(b[2], 0).tolist(), np.round(np.degrees(b[1][3:]), 0).tolist(), b[1][2] * 1e3, p.clear(T)))
