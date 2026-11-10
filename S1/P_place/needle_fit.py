#!/usr/bin/env python3
"""READ-ONLY. Known-radius needle circle from one stereo pair using everything: arc shape in BOTH images (chamfer), tip in both, and NCC depth along the arc.
All branch starts are tried and printed.  Output in camera and in the site frame (u entry->exit, m sideways, n up)."""
import json, os, sys
import numpy as np, cv2
from scipy.optimize import least_squares
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", R0 + "/G_place", R0 + "/P_place", os.path.expanduser("~/rl_live/sim2real_20261027/place/code")]
import tipmeas, needle_circle as NC, needle3d as N3
from ncc_disp import FB
FX, CX, CY = N3.FX, N3.CX, N3.CY; RAD = N3.RAD; DV = 2.2
def nv(y): return np.array([np.sin(y[3]) * np.cos(y[4]), np.sin(y[3]) * np.sin(y[4]), np.cos(y[3])])
def arc3(y, ns=60): n = nv(y); e1, e2 = NC._basis(n); th = y[5] + np.linspace(0, y[6], ns); return y[:3] + RAD * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2), n, e1, e2, th      # [0] tail ... [-1] tip
def ch(A, B): return np.min(np.linalg.norm(A[:, None] - B[None], axis=2), axis=1)
def fit(R, L, tipR, jawR, tipL, jawL, z=106.0, verbose=True):
    tipR, jawR, tipL, jawL = [np.array(v, float) for v in (tipR, jawR, tipL, jawL)]
    def arc2(I, t, j):
        mid = (t + j) / 2; r = tipmeas.detect_roi(I, z, tuple(mid), 340, tuple(j))
        if r is None: return None
        c = NC.dense_arc(r, 400); c = c[::-1] if np.linalg.norm(c[0] - t) < np.linalg.norm(c[-1] - t) else c; a, b = int(np.argmin(np.linalg.norm(c - j, axis=1))), int(np.argmin(np.linalg.norm(c - t, axis=1))); return c[min(a, b):max(a, b) + 1]
    cR, cL = arc2(R, tipR, jawR), arc2(L, tipL, jawL)
    if cR is None or cL is None: return None, f"arc detection R {cR is not None} L {cL is not None}"
    m0, why = N3.measure(R, L, tuple(tipR), tuple(jawR), z_hint=z, verbose=False)
    if m0 is None: return None, "stereo: " + why
    A = m0["A"][m0["keep"]]; cLr = cL - [0, DV]; tipLr = tipL - [0, DV]
    def res(y):
        P = arc3(y)[0]; uR = np.c_[FX * P[:, 0] / P[:, 2] + CX, FX * P[:, 1] / P[:, 2] + CY]; uL = uR - np.c_[FB / P[:, 2], np.zeros(len(P))]
        dm = ch(A[:, :2], uR); ix = np.argmin(np.linalg.norm(A[:, :2][:, None] - uR[None], axis=2), axis=1); dd = (FB / P[ix, 2]) - A[:, 2]
        return np.r_[ch(uR, cR), ch(cR[::5], uR), ch(uL, cLr), ch(cLr[::5], uL), 3 * np.linalg.norm(uR[-1] - tipR), 3 * np.linalg.norm(uL[-1] - tipLr), np.linalg.norm(uR[0] - jawR), np.linalg.norm(uL[0] - (jawL - [0, DV])), 1.0 * dd]
    C0, n0 = m0["centre"], m0["normal"]; ray = C0 / np.linalg.norm(C0); nm = 2 * (n0 @ ray) * ray - n0; sols = []
    for n in (n0, -n0, nm, -nm):
        e1, e2 = NC._basis(n); tp = m0["tip"]; tl = m0["tail"]; t1 = np.arctan2((tp - C0) @ e2, (tp - C0) @ e1); t0 = np.arctan2((tl - C0) @ e2, (tl - C0) @ e1); sp = (t1 - t0 + np.pi) % (2 * np.pi) - np.pi
        for s in (sp, sp - np.sign(sp) * 2 * np.pi):
            y0 = np.r_[C0, np.arccos(np.clip(n[2], -1, 1)), np.arctan2(n[1], n[0]), t1 - s, s]
            try: o = least_squares(res, y0, loss="soft_l1", f_scale=2.0, x_scale=np.r_[1, 1, 2, .1, .1, .1, .1], max_nfev=120)
            except Exception: continue
            sols.append((float(np.mean(np.abs(o.fun))), o.x))
    sols.sort(key=lambda s: s[0]); y = sols[0][1]; P, n, e1, e2, th = arc3(y); tan = np.sign(y[6]) * (-np.sin(th[-1]) * e1 + np.cos(th[-1]) * e2); r = res(y); nA = len(A); ix = np.argmin(np.linalg.norm(A[:, :2][:, None] - np.c_[FX * P[:, 0] / P[:, 2] + CX, FX * P[:, 1] / P[:, 2] + CY][None], axis=2), axis=1)
    uniq = []
    for c, yy in sols:
        if not any(abs(nv(yy) @ nv(u[1])) > 0.97 for u in uniq): uniq.append((c, yy))
    rep = dict(cost_px=sols[0][0], depth_rms_px=float(np.sqrt(np.mean(r[-nA:] ** 2))), span_deg=float(abs(np.degrees(y[6]))), runner_up=[round(u[0], 2) for u in uniq[1:3]], runner_up_normals=[np.round(nv(u[1]), 2).tolist() for u in uniq[1:3]], n_stereo=nA)
    return dict(y=y, tip=P[-1], tail=P[0], centre=y[:3], normal=n, tangent=tan / np.linalg.norm(tan), P=P, report=rep, cR=cR, cL=cL), "ok"
if __name__ == "__main__":
    snap, mj, out = sys.argv[1], sys.argv[2], sys.argv[3]; h = list(map(float, sys.argv[4:12])); os.makedirs(out, exist_ok=True); R = cv2.imread(snap + "/right_rect.png"); L = cv2.imread(snap + "/left_rect.png")
    m, why = fit(R, L, h[:2], h[2:4], h[4:6], h[6:8]); print(why)
    if m is None: sys.exit(1)
    print("report", m["report"]); s = N3.site(json.load(open(mj))["stickers"]); B = np.c_[s["u"], s["m"], s["n"]]; q = dict(tip=B.T @ (m["tip"] - s["e"]), tangent=B.T @ m["tangent"], normal=B.T @ m["normal"], centre=B.T @ (m["centre"] - s["e"]), tail=B.T @ (m["tail"] - s["e"]))
    [print("  ", k, np.round(v, 2).tolist()) for k, v in q.items()]; t, nn = q["tangent"], q["normal"]
    print("tip direction %.0f deg from straight into tissue; along u %.2f m %.2f n %.2f" % (np.degrees(np.arccos(np.clip(-t[2], -1, 1))), t[0], t[1], t[2])); print("plane %.0f deg from vertical; horizontal line of the plane vs entry->exit %.0f deg; tail above tip %.1f mm" % (np.degrees(np.arcsin(abs(nn[2]))), np.degrees(np.arctan2(abs(nn[0]), abs(nn[1]))), q["tail"][2] - q["tip"][2]))
    fs = None
    P = m["P"]; uR = np.c_[FX * P[:, 0] / P[:, 2] + CX, FX * P[:, 1] / P[:, 2] + CY]; uL = uR - np.c_[FB / P[:, 2], -DV * np.ones(len(P))]; oR, oL = R.copy(), L.copy(); cv2.polylines(oR, [np.int32(uR)], False, (0, 255, 0), 1); cv2.polylines(oL, [np.int32(uL)], False, (0, 255, 0), 1); cv2.imwrite(out + "/fit.png", np.hstack([oL[480:800, 520:940], oR[480:800, 620:1040]]))
    json.dump({"y": m["y"].tolist(), "camera": {k: m[k].tolist() for k in ("tip", "tail", "centre", "normal", "tangent")}, "site": {k: v.tolist() for k, v in q.items()}, "report": m["report"], "e": s["e"].tolist(), "bite": s["bite"]}, open(out + "/fit.json", "w"), indent=1)
