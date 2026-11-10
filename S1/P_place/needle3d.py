#!/usr/bin/env python3
"""READ-ONLY. Needle 3D from one stereo pair: right arc samples -> NCC disparity (rows corrected) -> Z continuity outlier rejection -> known-radius circle.
measure() is the reusable part: returns tip, tip tangent, plane normal, centre, tail in camera mm plus a quality report."""
import json, os, sys
import numpy as np, cv2
from scipy.optimize import least_squares
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", R0 + "/G_place", R0 + "/P_place", os.path.expanduser("~/rl_live/sim2real_20261027/place/code")]
import tipmeas, needle_circle as NC
from ncc_disp import disp, FB
FX, CX, CY = 1860.724932, 690.538345, 576.803638; RAD = 9.96
def backproj(a, b, d): Z = FB / d; return np.array([(a - CX) * Z / FX, (b - CY) * Z / FX, Z])
def snap_ridge(g, a, b, nx, ny, w=6):
    """move to the darkest/brightest-contrast centre of the wire across the arc."""
    ts = np.arange(-w, w + 1); v = np.array([cv2.getRectSubPix(g, (1, 1), (float(a + t * nx), float(b + t * ny)))[0, 0] for t in ts]); bg = np.median(v); k = int(np.argmax(np.abs(v - bg))); return a + ts[k] * nx, b + ts[k] * ny
def measure(R, L, tipR, jawR, tipL=None, z_hint=108.0, step_px=12, verbose=True):
    g = cv2.cvtColor(R, 6).astype(np.float32); mid = (np.array(tipR, float) + np.array(jawR, float)) / 2; rr = tipmeas.detect_roi(R, z_hint, tuple(mid), 320, tuple(jawR))
    if rr is None: return None, "no arc"
    c = NC.dense_arc(rr, 400); c = c[::-1] if np.linalg.norm(c[0] - np.array(tipR)) < np.linalg.norm(c[-1] - np.array(tipR)) else c       # tail -> tip
    i_jaw = int(np.argmin(np.linalg.norm(c - np.array(jawR), axis=1))); i_tip = int(np.argmin(np.linalg.norm(c - np.array(tipR), axis=1))); c = c[min(i_jaw, i_tip):max(i_jaw, i_tip) + 1]
    S = [c[0]]
    for p in c[1:]:
        if np.linalg.norm(p - S[-1]) >= step_px: S.append(p)
    S = np.array(S); rows = []
    for i, (a, b) in enumerate(S):
        t = S[min(i + 1, len(S) - 1)] - S[max(i - 1, 0)]; t = t / (np.linalg.norm(t) + 1e-9); a2, b2 = snap_ridge(g, a, b, -t[1], t[0]); best = None
        for hw in (7, 9):
            try: r = disp(R, L, a2, b2, hw=hw, d0=FB / z_hint, rng=30)
            except Exception: continue
            if best is None or r[0] > best[0]: best = r
        slope = abs(t[1])
        if best: rows.append([a2, b2, best[1], best[0], slope])
    A = np.array(rows); keep = (A[:, 3] > 0.75) & (A[:, 4] > 0.15)
    for _ in range(3):        # Z continuity: line-fit residual over arc index (robust)
        ix = np.nonzero(keep)[0]
        if len(ix) < 8: break
        p = np.polyfit(ix, A[ix, 2], 3); res = A[:, 2] - np.polyval(p, np.arange(len(A))); s = max(1.0, 1.4826 * np.median(np.abs(res[ix]))); keep = keep & (np.abs(res) < 2.5 * s)
    if keep.sum() < 8: return None, f"only {int(keep.sum())} good stereo points of {len(A)}"
    P = np.array([backproj(*a[:3]) for a in A[keep]])
    # known-radius circle: x = centre(3), normal angles(2); residual = distance to circle + to plane
    def n_of(y): return np.array([np.sin(y[3]) * np.cos(y[4]), np.sin(y[3]) * np.sin(y[4]), np.cos(y[3])])
    def res3(y): n = n_of(y); d = P - y[:3]; h = d @ n; return np.r_[np.abs(np.linalg.norm(d - h[:, None] * n, axis=1) - RAD), h]
    c0 = P.mean(0); n0 = np.linalg.svd(P - c0, full_matrices=False)[2][2]; best = None
    for sg in (1, -1):
        u0 = np.linalg.svd(P - c0, full_matrices=False)[2][1]
        for nn in (n0, -n0):
            y0 = np.r_[c0 + sg * u0 * RAD * 0.7, np.arccos(np.clip(nn[2], -1, 1)), np.arctan2(nn[1], nn[0])]; s = least_squares(res3, y0, loss="soft_l1", f_scale=0.7, max_nfev=200); cost = float(np.sqrt(np.mean(s.fun ** 2)))
            if best is None or cost < best[0]: best = (cost, s.x)
    cost, y = best; C, n = y[:3], n_of(y)
    # tip: right-image tip ray on the plane, snapped to the circle; tangent toward the tip
    def on_plane(uv): r = np.r_[(uv[0] - CX) / FX, (uv[1] - CY) / FX, 1.0]; return r * ((C @ n) / (r @ n))
    def on_circle(X): d = X - C; d = d - (d @ n) * n; return C + RAD * d / np.linalg.norm(d)
    tip = on_circle(on_plane(tipR)); tail = on_circle(on_plane(jawR)); e1 = (tip - C) / RAD; e2 = np.cross(n, e1); ang = np.arctan2((tail - C) @ e2, (tail - C) @ e1)
    tan = -np.sign(ang) * e2 if ang != 0 else e2          # moving from the tail toward the tip
    rep = dict(n_pts=int(keep.sum()), n_all=len(A), circle_rms_mm=cost, span_deg=float(abs(np.degrees(ang))), Z_tip_side=float(P[-1, 2]), Z_tail_side=float(P[0, 2]), plane_rms=float(np.std((P - C) @ n)))
    return dict(tip=tip, tangent=tan, normal=n, centre=C, tail=tail, P=P, A=A, keep=keep, report=rep), "ok"
def site(st):
    en, ex = (st[1], st[0]) if st[1]["area"] < st[0]["area"] else (st[0], st[1]); e, x = np.array(en["cam"]), np.array(ex["cam"]); n = np.array(en["n25"]); n = n if n[2] < 0 else -n; u = (x - e) - ((x - e) @ n) * n; bite = float(np.linalg.norm(u)); u /= bite
    return dict(e=e, x=x, n=n, u=u, m=np.cross(n, u), bite=bite, nloc=np.array(en["n4"]))
def in_site(s, m): B = np.c_[s["u"], s["m"], s["n"]]; f = lambda v: (B.T @ v); return dict(tip=f(m["tip"] - s["e"]), tangent=f(m["tangent"]), normal=f(m["normal"]), centre=f(m["centre"] - s["e"]), tail=f(m["tail"] - s["e"]))
if __name__ == "__main__":
    snap, mj, tu, tv, ju, jv, out = sys.argv[1], sys.argv[2], *map(float, sys.argv[3:7]), sys.argv[7]; os.makedirs(out, exist_ok=True); R = cv2.imread(snap + "/right_rect.png"); L = cv2.imread(snap + "/left_rect.png")
    m, why = measure(R, L, (tu, tv), (ju, jv)); print(why)
    if m is None: sys.exit(1)
    print("report", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in m["report"].items()}); print("arc samples u v disp ncc:", np.round(m["A"][::3, :4], 1).tolist())
    s = site(json.load(open(mj))["stickers"]); q = in_site(s, m); print("site: bite %.1f" % s["bite"]); [print(" ", k, np.round(v, 2).tolist()) for k, v in q.items()]
    t = q["tangent"]; print("tip direction: %.0f deg from straight into tissue (-n); along entry->exit %.2f, sideways %.2f" % (np.degrees(np.arccos(np.clip(-t[2], -1, 1))), t[0], t[1]))
    nn = q["normal"]; print("plane: %.0f deg from vertical; plane horizontal line vs entry->exit %.0f deg" % (np.degrees(np.arcsin(abs(nn[2]))), np.degrees(np.arctan2(abs(nn[0]), abs(nn[1])))))
    o = R.copy(); C, n, tip = m["centre"], m["normal"], m["tip"]; e1 = (tip - C) / RAD; e2 = np.cross(n, e1); th = np.radians(np.arange(0, 361, 3)); X = C + RAD * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2); uv = np.c_[FX * X[:, 0] / X[:, 2] + CX, FX * X[:, 1] / X[:, 2] + CY]; cv2.polylines(o, [np.int32(uv)], False, (0, 255, 0), 1)
    for a in m["A"][m["keep"]]: cv2.circle(o, (int(a[0]), int(a[1])), 2, (255, 0, 0), -1)
    for a in m["A"][~m["keep"]]: cv2.circle(o, (int(a[0]), int(a[1])), 2, (0, 0, 255), -1)
    tt = (FX * tip[0] / tip[2] + CX, FX * tip[1] / tip[2] + CY); cv2.circle(o, (int(tt[0]), int(tt[1])), 6, (0, 255, 255), 1); cv2.imwrite(out + "/needle3d.png", cv2.resize(o[480:780, 620:980], None, fx=3, fy=3))
    json.dump({"camera": {k: m[k].tolist() for k in ("tip", "tangent", "normal", "centre", "tail")}, "site": {k: v.tolist() for k, v in q.items()}, "report": m["report"], "bite": s["bite"]}, open(out + "/needle3d.json", "w"), indent=1)
