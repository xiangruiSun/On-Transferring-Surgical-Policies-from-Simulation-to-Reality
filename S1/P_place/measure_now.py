#!/usr/bin/env python3
"""READ-ONLY, offline. Held-needle circle (stereo, needle_circle.measure3) and scene (stickers, local + wide plane) from one snapshot."""
import json, os, sys
import numpy as np, cv2
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", R0 + "/G_place"]
import needle_circle as NC
from rig import FX, CX, CY, FB
snap, rx, ry, jx, jy = sys.argv[1], *map(float, sys.argv[2:6]); out = sys.argv[6]
R = cv2.imread(snap + "/right_rect.png"); L = cv2.imread(snap + "/left_rect.png"); disp = np.load(snap + "/fs_disp.npy").astype(np.float32)
Z = FB / np.clip(disp, 1, None); vv, uu = np.mgrid[0:disp.shape[0], 0:disp.shape[1]]; P = np.dstack([(uu - CX) * Z / FX, (vv - CY) * Z / FX, Z])
def proj(X): X = np.atleast_2d(X); return np.c_[FX * X[:, 0] / X[:, 2] + CX, FX * X[:, 1] / X[:, 2] + CY]
ov = R.copy(); res = {}
for f, nm in ((NC.measure3, "m3"), (NC.measure2, "m2")):
    z = float(sys.argv[7]) if len(sys.argv) > 7 else 118.0
    m, why = f(R, L, z, (rx, ry), 300, (jx, jy)); print(nm, why)
    if m is None: continue
    print({k: (np.round(m[k], 3).tolist() if k in m else None) for k in ("tip_cam", "dir_cam", "centre", "normal", "radius_mm")}, "span", m.get("span_deg"), "cost", m.get("cost_px"), "right/left px", m.get("right_px"), m.get("left_px"), "n", m.get("n_points"), "rms", m.get("rms_mm"))
    res[nm] = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items() if k in ("tip_cam", "dir_cam", "centre", "normal", "radius_mm", "span_deg", "cost_px", "right_px", "left_px", "poly")}
    C, n, r, t = np.array(m["centre"]), np.array(m["normal"]), m["radius_mm"], np.array(m["tip_cam"])
    e1 = (t - C) / np.linalg.norm(t - C); e2 = np.cross(n, e1); th = np.radians(np.arange(0, 361, 3))
    X = C + r * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2); col = (0, 255, 0) if nm == "m3" else (255, 255, 0)
    cv2.polylines(ov, [np.int32(proj(X))], False, col, 1); cv2.circle(ov, tuple(np.int32(proj(t)[0])), 6, col, 1)
# stickers: two largest blue blobs
lab = cv2.cvtColor(R, cv2.COLOR_BGR2LAB); blue = cv2.morphologyEx((lab[..., 2] < 120).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
nb, lb, st, ct = cv2.connectedComponentsWithStats(blue); order = sorted(range(1, nb), key=lambda k: -st[k, 4])[:3]
S = []
for k in order:
    if st[k, 4] < 1500: continue
    mask = lb == k; pts = P[mask]; c = np.median(pts, 0); S.append(dict(area=int(st[k, 4]), uv=ct[k].tolist(), cam=c.tolist(), disp=float(np.median(disp[mask])))); print("blob", S[-1])
    cv2.drawContours(ov, cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 255), 1)
res["stickers"] = S
def plane(c, rad):
    d = np.linalg.norm(P - c, axis=2) < rad; Q = P[d][::3]; keep = np.ones(len(Q), bool)
    for _ in range(5):
        q0 = Q[keep].mean(0); nn = np.linalg.svd(Q[keep] - q0, full_matrices=False)[2][2]; r = (Q - q0) @ nn; keep = np.abs(r) < 2.5 * np.std(r[keep]) + 1e-6
    if nn[2] > 0: nn = -nn
    return nn, q0, float(np.std(r[keep]))
for s in S:
    for rad in (4, 8, 15, 25):
        nn, q0, rms = plane(np.array(s["cam"]), rad); print(f"  blob area {s['area']} plane r={rad}: n {np.round(nn, 3).tolist()} rms {rms:.2f} tilt-from-view {np.degrees(np.arccos(-nn[2])):.0f}")
        s[f"n{rad}"] = nn.tolist()
os.makedirs(out, exist_ok=True); cv2.imwrite(out + "/measure.png", ov); json.dump(res, open(out + "/measure.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)); print("->", out)
