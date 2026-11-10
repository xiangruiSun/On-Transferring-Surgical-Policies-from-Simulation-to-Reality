#!/usr/bin/env python3
"""READ-ONLY. Smoothed needle depth along the arc for a frame vs the demo, measured from the tip, plus the arc in site coordinates."""
import json, sys
import numpy as np, cv2
sys.path.insert(0, ".")
import needle3d as N3
from ncc_disp import disp, FB
FX, CX, CY = N3.FX, N3.CX, N3.CY
def prof(d, seeds, z, s):
    R = cv2.imread(d + "/right_rect.png"); L = cv2.imread(d + "/left_rect.png"); S = np.array(seeds, float); rr = [max((disp(R, L, a, b, hw=h, d0=FB / z, rng=28) for h in (7, 9)), key=lambda x: x[0]) for a, b in S]; dd = np.array([r[1] for r in rr]); nc = np.array([r[0] for r in rr]); t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(S, axis=0), axis=1))]; keep = nc > 0.7
    for _ in range(3): p = np.polyfit(t[keep], dd[keep], 2); res = dd - np.polyval(p, t); keep = (np.abs(res) < max(1.5, 2 * np.std(res[keep]))) & (nc > 0.7)
    ds = np.polyval(p, t); Z = FB / ds; X = np.c_[(S[:, 0] - CX) * Z / FX, (S[:, 1] - CY) * Z / FX, Z]; B = np.c_[s["u"], s["m"], s["n"]]; Q = (B.T @ (X - s["e"]).T).T; arcl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]; print("  raw d", np.round(dd, 1).tolist(), "ncc", np.round(nc, 2).tolist()); print("  kept %d/%d rms %.2f px" % (keep.sum(), len(S), np.sqrt(np.mean(res[keep] ** 2)))); return arcl, Q
so = N3.site(json.load(open("m0/measure.json"))["stickers"]); print("demo"); ad, Qd = prof("../G_place/correct_143942/snap", [(722, 570), (696, 625), (715, 668), (753, 685), (800, 693), (850, 690), (890, 682)], 108, so)
sn = N3.site(json.load(open(sys.argv[3] if len(sys.argv) > 3 else "m_set2/measure.json"))["stickers"]); print("now"); an, Qn = prof(sys.argv[1], json.loads(sys.argv[2]), 105, sn)
def table(a, Q, nm):
    L = np.linspace(0, min(14, a[-1]), 8); print(nm); [print("   arc %4.1f mm from tip: site u %6.1f  m %6.1f  n %6.1f  (height above tip %5.1f)" % (l, np.interp(l, a, Q[:, 0]), np.interp(l, a, Q[:, 1]), np.interp(l, a, Q[:, 2]), np.interp(l, a, Q[:, 2]) - Q[0, 2])) for l in L]
table(ad, Qd, "DEMO needle in site frame"); table(an, Qn, "CURRENT needle in site frame")
L = np.linspace(0, min(14, ad[-1], an[-1]), 12); f = lambda a, Q: np.c_[[np.interp(L, a, Q[:, i]) for i in range(3)]].T; Pd, Pn = f(ad, Qd), f(an, Qn); c1, c2 = Pn.mean(0), Pd.mean(0); U, S, Vt = np.linalg.svd((Pn - c1).T @ (Pd - c2)); D = np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]); Rr = Vt.T @ D @ U.T
from scipy.spatial.transform import Rotation as Ro
rv = Ro.from_matrix(Rr).as_rotvec(); print("rotation needed current -> demo: %.0f deg about site axis (u m n) %s; shape rms %.2f mm" % (np.degrees(np.linalg.norm(rv)), np.round(rv / (np.linalg.norm(rv) + 1e-9), 2).tolist(), np.sqrt(np.mean(np.sum((Pd - c2 - (Rr @ (Pn - c1).T).T) ** 2, 1))))); print("tip offset demo-now site mm", np.round(Pd[0] - Pn[0], 1).tolist()); json.dump(dict(rotvec_site=rv.tolist()), open(sys.argv[1] + "/rot_needed.json", "w"))
