#!/usr/bin/env python3
"""READ-ONLY (2026-11-05 setting 2). grip_fit_pts.py with one fix: the demo needle is put in the tool frame with the DEMO's own calibration
(C_r6cal_try2 M + demo SUJ), not the current one. usage: grip_fit_s2.py '<json [[u,v,disp],..] tail->tip>' <new calib json> <new q json list> [out.json]
Also prints what the old mixed-calibration computation gives, for comparison."""
import json, os, sys
import numpy as np, cv2
sys.path[:0] = [os.path.expanduser("~/rl_live/sim2real_20261106/S1/code"), os.path.expanduser("~/rl_live/sim2real_20261106/S1/P_place"), os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K
from ncc_disp import disp, FB
FX, CX, CY = 1860.724932, 690.538345, 576.803638
def c2e(x, M): return (np.asarray(x) - M[3]) @ np.linalg.inv(M[:3])
DEMO = os.path.expanduser("~/rl_live/sim2real_20261106/S1/G_place/correct_143942"); dem = json.load(open(DEMO + "/demo.json")); calD = json.load(open(os.path.expanduser("~/rl_live/sim2real_20261104/C_r6cal_try2/calibration_tcp.json")))
seeds = [(890, 682), (850, 690), (800, 693), (753, 685), (715, 668), (696, 625), (722, 570)]
R = cv2.imread(DEMO + "/snap/right_rect.png"); L = cv2.imread(DEMO + "/snap/left_rect.png"); S = np.array(seeds, float)
rr = [max((disp(R, L, a, b, hw=h, d0=FB / 108, rng=28) for h in (7, 9)), key=lambda x: x[0]) for a, b in seeds]; dd = np.array([r[1] for r in rr]); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(S, axis=0), axis=1))]; keep = np.ones(len(S), bool)
for _ in range(3): p = np.polyfit(s[keep], dd[keep], 2); res = dd - np.polyval(p, s); keep = np.abs(res) < max(1.5, 2.0 * np.std(res[keep]))
Zd = FB / np.polyval(p, s); Xd = np.c_[(S[:, 0] - CX) * Zd / FX, (S[:, 1] - CY) * Zd / FX, Zd]
def tool(X, q, M, SUJ): T = K.fk_ecm(np.array(q), SUJ); return np.array([T[:3, :3].T @ (c2e(x, M) - T[:3, 3] * 1e3) for x in X])
def fit(An, Ad):
    sn = np.r_[0, np.cumsum(np.linalg.norm(np.diff(An, axis=0), axis=1))]; sd = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Ad, axis=0), axis=1))]; Ln = min(sd[-1], sn[-1]); t = np.linspace(0, Ln, 12)
    f = lambda P, s, off: np.c_[[np.interp(t + off, s, P[:, i]) for i in range(3)]].T; Pd = f(Ad, sd, sd[-1] - Ln); Pn = f(An, sn, sn[-1] - Ln); c1, c2 = Pn.mean(0), Pd.mean(0); U, _, Vt = np.linalg.svd((Pn - c1).T @ (Pd - c2)); D = np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]); Rd = Vt.T @ D @ U.T
    rms = float(np.sqrt(np.mean(np.sum(((Rd @ (Pn - c1).T).T - (Pd - c2)) ** 2, 1)))); ang = float(np.degrees(np.arccos(np.clip((np.trace(Rd) - 1) / 2, -1, 1)))); return Rd, rms, ang, sn[-1], sd[-1]
UVD = np.array(json.loads(sys.argv[1]), float); cal = json.load(open(sys.argv[2])); q = json.loads(sys.argv[3]); Mn, SUJn = np.array(cal["M"]), np.array(cal["frozen"]["suj_psm1"]); Zn = FB / UVD[:, 2]; Xn = np.c_[(UVD[:, 0] - CX) * Zn / FX, (UVD[:, 1] - CY) * Zn / FX, Zn]; An = tool(Xn, q, Mn, SUJn)
A_old = tool(Xd, dem["js_mean"], Mn, SUJn); A_new = tool(Xd, dem["js_mean"], np.array(calD["M"]), np.array(dem["suj"]) if np.array(dem["suj"]).shape == (4, 4) else np.array(calD["frozen"]["suj_psm1"]))
for nm, Ad in (("OLD (demo with the current calibration)", A_old), ("FIXED (demo with its own calibration)", A_new)):
    Rd, rms, ang, ln, ld = fit(An, Ad); print("%s: grip %.1f deg from demo, shape rms %.2f mm, arc new %.1f demo %.1f mm" % (nm, ang, rms, ln, ld))
print("tip in tool frame mm: new", np.round(An[-1], 2).tolist(), "demo fixed", np.round(A_new[-1], 2).tolist(), "demo old", np.round(A_old[-1], 2).tolist())
if len(sys.argv) > 4:
    Rd, rms, ang, _, _ = fit(An, A_new); json.dump(dict(R_delta=Rd.tolist(), tip_tool_mm=An[-1].tolist(), tip_tool_demo_mm=A_new[-1].tolist(), rms=rms, angle=ang, method="grip_fit_s2 fixed"), open(sys.argv[4], "w"), indent=1)
