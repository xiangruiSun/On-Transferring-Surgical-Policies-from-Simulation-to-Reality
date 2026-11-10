#!/usr/bin/env python3
"""READ-ONLY. Known-radius 3D circle fitted to the needle arc detected separately in BOTH rectified images (chamfer both ways, tip-to-tip, measured
row offset).  All tilt branches are tried and all costs printed, so a wrong branch is visible."""
import json, os, sys
import numpy as np, cv2
from scipy.optimize import least_squares
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", R0 + "/G_place", R0 + "/P_place", os.path.expanduser("~/rl_live/sim2real_20261027/place/code")]
import tipmeas, needle_circle as NC
from arc3d import row_offset
from rig import FX, CX, CY, FB

def basis(n): return NC._basis(n)
def fit(R, L, roi, jaw, roiL=None, jawL=None, z=115.0, radius=9.96, verbose=True, cut=(540, 580), cut_r=155):
    d0 = FB / z; dv, nn, sd = row_offset(R, L, d0)
    rR = tipmeas.detect_roi(R, z, roi, 300, jaw); rL = tipmeas.detect_roi(L, z, roiL or (roi[0] - d0, roi[1]), 300, jawL or (jaw[0] - d0, jaw[1]))
    if rR is None or rL is None: return None, f"detect R {rR is not None} L {rL is not None}"
    cR, cL = NC.dense_arc(rR, 200), NC.dense_arc(rL, 200)
    # orient both tail -> tip: the tip is the end farther from the jaw
    for c in (cR,):
        pass
    if np.linalg.norm(cR[0] - jaw) > np.linalg.norm(cR[-1] - jaw): cR = cR[::-1]
    jl = np.array(jawL or (jaw[0] - d0, jaw[1]))
    if np.linalg.norm(cL[0] - jl) > np.linalg.norm(cL[-1] - jl): cL = cL[::-1]
    # cut the arc at the jaw: the detector runs on along the jaw / thread past the needle tail
    hR = np.array(cut); hL = hR - [FB / z, 0]; cR = cR[np.linalg.norm(cR - hR, axis=1) <= cut_r]; cL = cL[np.linalg.norm(cL - hL, axis=1) <= cut_r]
    cL = cL - [0, dv]                                   # left rows -> right rows
    if verbose: print("row offset %.2f; R tail %s tip %s | L tail %s tip %s | disp tip %.1f tail %.1f" % (dv, np.round(cR[0]).tolist(), np.round(cR[-1]).tolist(), np.round(cL[0]).tolist(), np.round(cL[-1]).tolist(), cR[-1, 0] - cL[-1, 0], cR[0, 0] - cL[0, 0]))
    def arc(x, ns=60):
        n = np.array([np.sin(x[3]) * np.cos(x[4]), np.sin(x[3]) * np.sin(x[4]), np.cos(x[3])]); e1, e2 = basis(n); th = x[5] + np.linspace(0, x[6], ns)
        return x[:3] + radius * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2), n, e1, e2, th
    def resid(x):
        P = arc(x)[0]; uR = np.c_[FX * P[:, 0] / P[:, 2] + CX, FX * P[:, 1] / P[:, 2] + CY]; uL = uR - np.c_[FB / P[:, 2], np.zeros(len(P))]
        f = lambda A, B: np.min(np.linalg.norm(A[:, None] - B[None], axis=2), axis=1)
        return np.r_[f(uR, cR), f(cR[::4], uR), f(uL, cL), f(cL[::4], uL), 3 * np.linalg.norm(uR[-1] - cR[-1]), 3 * np.linalg.norm(uL[-1] - cL[-1]), np.linalg.norm(uR[0] - cR[0]), np.linalg.norm(uL[0] - cL[0])]
    Cuv = cR.mean(0); best = []
    for th0 in np.radians([15, 35, 55, 75]):
        for ph0 in np.radians(np.arange(0, 360, 45)):
            n0 = np.array([np.sin(th0) * np.cos(ph0), np.sin(th0) * np.sin(ph0), np.cos(th0)]); e1, e2 = basis(n0)
            for flip in (1, -1):
                C0 = np.r_[(Cuv[0] - CX) * z / FX, (Cuv[1] - CY) * z / FX, z] + flip * 0
                Pt = np.r_[(cR[-1, 0] - CX) * z / FX, (cR[-1, 1] - CY) * z / FX, z]; t1 = np.arctan2((Pt - C0) @ e2, (Pt - C0) @ e1); span = flip * np.radians(150)
                x0 = np.r_[C0, th0, ph0, t1 - span, span]
                try: s = least_squares(resid, x0, loss="soft_l1", f_scale=2.0, x_scale=np.r_[1, 1, 3, .1, .1, .1, .1], max_nfev=150)
                except Exception: continue
                best.append((float(np.mean(np.abs(s.fun))), s.x))
    # the circle fit alone cannot separate the two tilt branches (costs within 0.2 px): pick by the DIRECT disparity of the arc ends, which are 2D points
    dtip, dtail = cR[-1, 0] - cL[-1, 0], cR[0, 0] - cL[0, 0]
    def end_err(xx): Pp = arc(xx)[0]; return abs(FB / Pp[-1, 2] - dtip) + 0.5 * abs(FB / Pp[0, 2] - dtail)
    best.sort(key=lambda b: b[0]); low = [b for b in best if b[0] < best[0][0] + 1.0]; low.sort(key=lambda b: end_err(b[1])); x = low[0][1]; P, n, e1, e2, th = arc(x); tan = np.sign(x[6]) * (-np.sin(th[-1]) * e1 + np.cos(th[-1]) * e2)
    uniq = []
    for c, xx in best:
        nn2 = arc(xx)[1]
        if not any(abs(nn2 @ u[1]) > 0.985 for u in uniq): uniq.append((c, nn2, arc(xx)[0]))
    if verbose:
        for c, nn2, Pp in uniq[:5]: print("  branch cost %.2f px normal %s tip Z %.1f (disp %.1f, measured %.1f) tail Z %.1f (disp %.1f, measured %.1f)" % (c, np.round(nn2, 3).tolist(), Pp[-1, 2], FB / Pp[-1, 2], dtip, Pp[0, 2], FB / Pp[0, 2], dtail))
    return dict(centre=x[:3], normal=n, radius_mm=radius, tip_cam=P[-1], tail_cam=P[0], dir_cam=tan / np.linalg.norm(tan), span_deg=float(np.degrees(abs(x[6]))), cost_px=low[0][0], end_err_px=end_err(x), row_offset=dv, arc=P, cR=cR, cL=cL + [0, dv],
                margin=(uniq[1][0] - uniq[0][0]) if len(uniq) > 1 else None), "ok"

if __name__ == "__main__":
    snap, rx, ry, jx, jy, out = sys.argv[1], *map(float, sys.argv[2:6]), sys.argv[6]; os.makedirs(out, exist_ok=True)
    R = cv2.imread(snap + "/right_rect.png"); L = cv2.imread(snap + "/left_rect.png")
    m, why = fit(R, L, (rx, ry), (jx, jy)); print(why)
    if m is None: sys.exit(1)
    print({k: np.round(m[k], 3).tolist() for k in ("centre", "normal", "tip_cam", "tail_cam", "dir_cam")}, "span %.1f cost %.2f px, runner-up margin %s" % (m["span_deg"], m["cost_px"], m["margin"]))
    P = m["arc"]; uR = np.c_[FX * P[:, 0] / P[:, 2] + CX, FX * P[:, 1] / P[:, 2] + CY]; uL = uR - np.c_[FB / P[:, 2], -m["row_offset"] * np.ones(len(P))]
    oR, oL = R.copy(), L.copy(); cv2.polylines(oR, [np.int32(uR)], False, (0, 255, 0), 1); cv2.polylines(oL, [np.int32(uL)], False, (0, 255, 0), 1); cv2.polylines(oR, [np.int32(m["cR"])], False, (255, 0, 0), 1); cv2.polylines(oL, [np.int32(m["cL"])], False, (255, 0, 0), 1)
    cv2.imwrite(out + "/arcfit.png", np.hstack([cv2.resize(oL[380:800, 240:800], None, fx=1.1, fy=1.1), cv2.resize(oR[380:800, 330:890], None, fx=1.1, fy=1.1)]))
    json.dump({k: (m[k].tolist() if hasattr(m[k], "tolist") else m[k]) for k in ("centre", "normal", "tip_cam", "tail_cam", "dir_cam", "radius_mm", "span_deg", "cost_px", "end_err_px", "row_offset", "margin")}, open(out + "/arcfit.json", "w"), indent=1)
