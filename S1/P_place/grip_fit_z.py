#!/usr/bin/env python3
"""READ-ONLY. Rotation R_delta (tool frame) taking the new grip's needle onto the demo grip's needle: arc points -> smoothed depth -> resampled by arc length -> Kabsch.  Target tool orientation = demo orientation * R_delta."""
import json, os, sys
import numpy as np, cv2
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", R0 + "/P_place", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K
from calib import cam2ecm_mm
from ncc_disp import disp, FB
FX, CX, CY = 1860.724932, 690.538345, 576.803638; cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); SUJ = np.array(cal["frozen"]["suj_psm1"])
def arc(d, q, seeds, z):
    R = cv2.imread(d + "/right_rect.png"); L = cv2.imread(d + "/left_rect.png"); T = K.fk_ecm(np.array(q), SUJ); S = np.array(seeds, float); rr = [max((disp(R, L, a, b, hw=h, d0=FB / z, rng=28) for h in (7, 9)), key=lambda x: x[0]) for a, b in S]; dd = np.array([r[1] for r in rr]); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(S, axis=0), axis=1))]; keep = np.ones(len(S), bool)
    for _ in range(3): p = np.polyfit(s[keep], dd[keep], 2); res = dd - np.polyval(p, s); keep = np.abs(res) < max(1.5, 2.0 * np.std(res[keep]))
    ds = np.polyval(p, s); Z = FB / ds; X = np.c_[(S[:, 0] - CX) * Z / FX, (S[:, 1] - CY) * Z / FX, Z]; print("  kept %d/%d, disp residual rms %.2f px, smoothed Z %s" % (keep.sum(), len(S), np.sqrt(np.mean(res[keep] ** 2)), np.round(Z, 1).tolist()))
    Pt = np.array([T[:3, :3].T @ (cam2ecm_mm(x, M) * 1e-3 - T[:3, 3]) * 1e3 for x in X]); sq = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pt, axis=0), axis=1))]; return Pt, sq, X
dem = json.load(open("../G_place/correct_143942/demo.json")); st = json.load(open(sys.argv[2])); dnew = os.path.dirname(sys.argv[2])
print("demo"); Ad, sd, Xd = arc("../G_place/correct_143942/snap", dem["js_mean"], [(890, 682), (850, 690), (800, 693), (753, 685), (715, 668), (696, 625), (722, 570)], 108)   # tail -> tip
print("new"); An, sn, Xn = arc(dnew, st["q"], json.loads(sys.argv[1]), float(sys.argv[4]) if len(sys.argv) > 4 else 112)   # 2026-11-05 copy: depth guess as argv[4]
L = min(sd[-1], sn[-1]); t = np.linspace(0, L, 12); f = lambda P, s, off: np.c_[[np.interp(t + off, s, P[:, i]) for i in range(3)]].T; Pd = f(Ad, sd, sd[-1] - L); Pn = f(An, sn, sn[-1] - L)   # aligned at the tip end
print("arc length used %.1f mm (demo %.1f, new %.1f)" % (L, sd[-1], sn[-1])); c1, c2 = Pn.mean(0), Pd.mean(0); U, S, Vt = np.linalg.svd((Pn - c1).T @ (Pd - c2)); D = np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]); Rd = Vt.T @ D @ U.T; rms = np.sqrt(np.mean(np.sum((Pd - c2 - (Rd @ (Pn - c1).T).T) ** 2, 1))); ang = np.degrees(np.arccos(np.clip((np.trace(Rd) - 1) / 2, -1, 1))); from scipy.spatial.transform import Rotation as Ro; ax = Ro.from_matrix(Rd).as_rotvec(); print("R_delta %.0f deg about tool axis %s; shape rms %.2f mm; singular values %s" % (ang, np.round(ax / np.linalg.norm(ax), 2).tolist(), rms, np.round(S, 0).tolist()))
json.dump(dict(R_delta=Rd.tolist(), tip_tool_mm=An[-1].tolist(), tip_tool_demo_mm=Ad[-1].tolist(), rms=rms, angle=ang, tip_cam_demo=Xd[-1].tolist()), open(sys.argv[3], "w"), indent=1)
