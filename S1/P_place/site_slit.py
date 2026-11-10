#!/usr/bin/env python3
"""READ-ONLY. Site frame when only the entry sticker is visible: bite direction u = the demo's angle between u and the slit, applied to the new slit.  Writes a measure.json usable by place_repro.py and draws the demo needle transferred to the new site as a target overlay."""
import json, os, sys
import numpy as np, cv2
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", R0 + "/P_place"]
import needle3d as N3
from rig import FX, CX, CY, FB
def cloud(snap):
    d = np.load(snap + "/fs_disp.npy").astype(np.float32); Z = FB / np.clip(d, 1, None); v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]; return np.dstack([(u - CX) * Z / FX, (v - CY) * Z / FX, Z])
def pts(P, uv): return np.array([np.median(P[int(b) - 3:int(b) + 4, int(a) - 3:int(a) + 4].reshape(-1, 3), 0) for a, b in uv])
def plane(P, c, rad):
    m = np.linalg.norm(P - c, axis=2) < rad; Q = P[m][::3]; k = np.ones(len(Q), bool)
    for _ in range(5):
        q0 = Q[k].mean(0); n = np.linalg.svd(Q[k] - q0, full_matrices=False)[2][2]; r = (Q - q0) @ n; k = np.abs(r) < 2.5 * np.std(r[k]) + 1e-6
    return n if n[2] < 0 else -n, float(np.std(r[k]))
snap, seed, slit_new, out = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3]), sys.argv[4]; os.makedirs(out, exist_ok=True)
sold = N3.site(json.load(open("m0/measure.json"))["stickers"]); Pold = cloud("../G_place/hover_142828/snap"); slit_old = [(745, 395), (800, 440), (860, 500)]; So = pts(Pold, slit_old); do = So[-1] - So[0]; do -= (do @ sold["n"]) * sold["n"]; do /= np.linalg.norm(do); ang = np.arctan2(sold["u"] @ np.cross(sold["n"], do), sold["u"] @ do); side_old = (sold["e"] - So[1]) @ np.cross(sold["n"], do)
print("demo: u vs slit %.1f deg; entry sticker on side %.1f mm of the slit" % (np.degrees(ang), side_old))
P = cloud(snap); R = cv2.imread(snap + "/right_rect.png"); lab = cv2.cvtColor(R, cv2.COLOR_BGR2LAB); blue = cv2.morphologyEx((lab[..., 2] < 122).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); nb, lb, st, ct = cv2.connectedComponentsWithStats(blue); k = lb[seed[1], seed[0]]
if k == 0: raise SystemExit("seed not on a blue blob")
mask = lb == k; e = np.median(P[mask], 0); print("sticker area", st[k, 4], "centroid uv", np.round(ct[k]).tolist(), "cam", np.round(e, 1).tolist(), "disp %.1f" % (FB / e[2]))
for rad in (4, 8, 15, 25): n, r = plane(P, e, rad); print("  plane r=%d n %s rms %.2f tilt-from-view %.0f" % (rad, np.round(n, 3).tolist(), r, np.degrees(np.arccos(-n[2]))))
n25, _ = plane(P, e, 25); n4, _ = plane(P, e, 4); Sn = pts(P, slit_new); dn = Sn[-1] - Sn[0]; dn -= (dn @ n25) * n25; dn /= np.linalg.norm(dn); best = None
for sgn in (1, -1):                                         # slit direction sign is arbitrary: keep the one that puts the sticker on the same side as in the demo and u across the slit away from the sticker
    d = sgn * dn; side = (e - Sn[len(Sn) // 2]) @ np.cross(n25, d)
    if np.sign(side) == np.sign(side_old): best = d; print("new sticker on side %.1f mm of the slit (demo %.1f)" % (side, side_old))
d = best; u = np.cos(ang) * d + np.sin(ang) * np.cross(n25, d); u /= np.linalg.norm(u); x = e + sold["bite"] * u; xuv = (FX * x[0] / x[2] + CX, FX * x[1] / x[2] + CY); print("u %s; virtual exit uv %s; rotation of the site vs demo about n: %.1f deg" % (np.round(u, 3).tolist(), np.round(xuv).tolist(), np.degrees(np.arccos(np.clip(u @ sold["u"], -1, 1)))))
json.dump(dict(stickers=[dict(area=int(st[k, 4]), cam=e.tolist(), n25=n25.tolist(), n4=n4.tolist(), uv=ct[k].tolist()), dict(area=int(st[k, 4]) + 1, cam=x.tolist(), n25=n25.tolist(), n4=n25.tolist(), uv=list(xuv), virtual=True)], snap=snap), open(out + "/measure.json", "w"), indent=1)
# demo needle (right-image samples + NCC depth) transferred through the site frames
A = N3  # noqa
Bo = np.c_[sold["u"], sold["m"], sold["n"]]; snew = N3.site(json.load(open(out + "/measure.json"))["stickers"]); Bn = np.c_[snew["u"], snew["m"], snew["n"]]; cd = json.load(open("c1/needle3d.json"))["camera"]; C, nn, tip = map(np.array, (cd["centre"], cd["normal"], cd["tip"])); tail = np.array(cd["tail"]); RAD = N3.RAD; e1 = (tip - C) / RAD; e2 = np.cross(nn, e1); a_t = np.arctan2((tail - C) @ e2, (tail - C) @ e1); th = np.linspace(0, a_t, 40); X = C + RAD * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2); Xn = snew["e"] + (Bn @ Bo.T @ (X - sold["e"]).T).T; uv = np.c_[FX * Xn[:, 0] / Xn[:, 2] + CX, FX * Xn[:, 1] / Xn[:, 2] + CY]
o = R.copy(); cv2.polylines(o, [np.int32(uv)], False, (0, 255, 0), 2); cv2.circle(o, tuple(np.int32(uv[0])), 6, (0, 255, 255), 2); cv2.arrowedLine(o, tuple(np.int32(ct[k])), tuple(np.int32(xuv)), (255, 255, 0), 2, tipLength=.05); [cv2.circle(o, tuple(int(c) for c in p), 3, (0, 0, 255), -1) for p in slit_new]; cv2.imwrite(out + "/site.png", o); np.save(out + "/target_cam.npy", Xn)
