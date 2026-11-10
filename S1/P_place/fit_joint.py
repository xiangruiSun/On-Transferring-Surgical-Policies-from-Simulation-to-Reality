#!/usr/bin/env python3
"""READ-ONLY. One fixed needle-in-tool circle fitted jointly to the arcs of all views, both images: X_cam = [(FK(q + delta) p_tool) mm, 1] @ M.
Models: A no wrist offset; B constant wrist offset delta (roll, pitch, yaw).  Prints per-view residuals in px and leave-one-out prediction error."""
import json, os, sys
import numpy as np, cv2
from scipy.optimize import least_squares
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", R0 + "/G_place", R0 + "/P_place", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib"), os.path.expanduser("~/rl_live/sim2real_20261027/place/code")]
import kin as K, tipmeas, needle_circle as NC
from arc3d import row_offset
from rig import FX, CX, CY, FB
d, hints = sys.argv[1], json.load(open(sys.argv[2])); z = float(sys.argv[3]); radius = 9.96
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); SUJ = np.array(cal["frozen"]["suj_psm1"]); V = []
for k, (tip, jaw) in hints.items():
    v = f"{d}/{k}"; R = cv2.imread(v + "/right_rect.png"); L = cv2.imread(v + "/left_rect.png"); st = json.load(open(v + "/state.json")); tip, jaw = np.array(tip, float), np.array(jaw, float); d0 = FB / z
    mid = (tip + jaw) / 2; pxr = np.linalg.norm(tip - jaw) / 2; rR = tipmeas.detect_roi(R, z, tuple(mid), 300, tuple(jaw)); rL = tipmeas.detect_roi(L, z, (mid[0] - d0, mid[1]), 300, (jaw[0] - d0, jaw[1]))
    if rR is None or rL is None: print(k, "no detection"); continue
    cR, cL = NC.dense_arc(rR, 200), NC.dense_arc(rL, 200)
    if np.linalg.norm(cR[0] - jaw) > np.linalg.norm(cR[-1] - jaw): cR = cR[::-1]
    if np.linalg.norm(cL[0] - (jaw - [d0, 0])) > np.linalg.norm(cL[-1] - (jaw - [d0, 0])): cL = cL[::-1]
    cR = cR[np.linalg.norm(cR - mid, axis=1) <= 1.12 * pxr]; cL = cL[np.linalg.norm(cL - (mid - [d0, 0]), axis=1) <= 1.12 * pxr]; dv = row_offset(R, L, d0)[0]; cL = cL - [0, dv]
    V.append(dict(k=k, q=np.array(st["q"]), cR=cR, cL=cL, dv=dv)); print(k, "arc pts R/L", len(cR), len(cL), "tip R", np.round(cR[-1]).tolist(), "row off %.2f" % dv, flush=True)
def circle(x, n=50):
    nv = np.array([np.sin(x[3]) * np.cos(x[4]), np.sin(x[3]) * np.sin(x[4]), np.cos(x[3])]); e1, e2 = NC._basis(nv); th = x[5] + np.linspace(0, x[6], n)
    return x[:3][None] + radius * 1e-3 * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2), nv, e1, e2
def uv(q, Pt):
    T = K.fk_ecm(q, SUJ); Pe = (T[:3, :3] @ Pt.T).T + T[:3, 3]; X = np.c_[Pe * 1e3, np.ones(len(Pe))] @ M; u = FX * X[:, 0] / X[:, 2] + CX; v = FX * X[:, 1] / X[:, 2] + CY; return np.c_[u, v], np.c_[u - FB / X[:, 2], v]
def ch(A, B): return np.min(np.linalg.norm(A[:, None] - B[None], axis=2), axis=1)
def resid_view(x, W, dl):
    Pt = circle(x)[0]; uR, uL = uv(W["q"] + np.r_[0, 0, 0, dl], Pt)
    return np.r_[ch(uR, W["cR"]), ch(W["cR"][::4], uR), ch(uL, W["cL"]), ch(W["cL"][::4], uL), 3 * np.linalg.norm(uR[-1] - W["cR"][-1]), 3 * np.linalg.norm(uL[-1] - W["cL"][-1])]
def solve(Vs, use_delta, seeds):
    best = None
    for x0 in seeds:
        z0 = np.r_[x0, np.zeros(3)] if use_delta else x0
        fn = lambda y: np.concatenate([resid_view(y[:7], W, y[7:10] if use_delta else np.zeros(3)) for W in Vs])
        try: s = least_squares(fn, z0, loss="soft_l1", f_scale=5.0, x_scale=np.r_[.003, .003, .003, .2, .2, .2, .2, *([.1, .1, .1] if use_delta else [])], max_nfev=60)
        except Exception as e: continue
        c = float(np.mean(np.abs(s.fun)))
        if best is None or c < best[0]: best = (c, s.x)
    return best
def seeds():
    S = []
    for th in np.radians([30, 70, 110, 150]):
        for ph in np.radians([0, 90, 180, 270]):
            nv = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]); e1, e2 = NC._basis(nv); C = np.array([-2.0, 9.6, 17.8]) * 1e-3; tp = np.array([-1.5, 17, 24]) * 1e-3; t1 = np.arctan2((tp - C) @ e2, (tp - C) @ e1)
            for fl in (1, -1): s = fl * np.radians(135); S.append(np.r_[C, th, ph, t1 - s, s])
    return S
res = {}
for name, ud in (("A no offset", False), ("B const wrist offset", True)):
    c, x = solve(V, ud, seeds()); dl = x[7:10] if ud else np.zeros(3); print(f"{name}: mean {c:.2f} px; wrist offset deg {np.round(np.degrees(dl), 1).tolist()}", flush=True)
    Pt, nv, _, _ = circle(x[:7]); print("  tool mm: tip", np.round(Pt[-1] * 1e3, 1).tolist(), "tail", np.round(Pt[0] * 1e3, 1).tolist(), "centre", np.round(x[:3] * 1e3, 1).tolist(), "normal", np.round(nv, 3).tolist(), "span %.0f" % abs(np.degrees(x[6])), flush=True)
    per = {W["k"]: (round(float(np.median(np.abs(resid_view(x[:7], W, dl)[:-2]))), 1), round(float(np.linalg.norm(uv(W["q"] + np.r_[0, 0, 0, dl], Pt)[0][-1] - W["cR"][-1])), 1)) for W in V}; print("  per view (median arc px, tip px):", per, flush=True)
    res[name] = dict(x=x.tolist(), cost=c, per=per, normal=nv.tolist())
    # leave-one-out: fit without the view, predict it
    loo = {}
    for i, W in enumerate(V):
        o = solve(V[:i] + V[i + 1:], ud, [x[:7]]); xo = o[1]; dlo = xo[7:10] if ud else np.zeros(3); Pto = circle(xo[:7])[0]; loo[W["k"]] = round(float(np.linalg.norm(uv(W["q"] + np.r_[0, 0, 0, dlo], Pto)[0][-1] - W["cR"][-1])), 1)
    print("  leave-one-out tip error px:", loo, flush=True); res[name]["loo"] = loo
json.dump(res, open(d + "/joint_fit.json", "w"), indent=1)
