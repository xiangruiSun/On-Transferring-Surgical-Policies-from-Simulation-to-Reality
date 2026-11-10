"""READ-ONLY: wrist angles that lay the held needle flat (needle plane ~ parallel to the pad) with the hinge at right-image pixel (u, v),
h mm above the pad. Needle shape in the tool frame from the trial-3 explicit stereo points (tr3/gview). Prints the best q lines."""
import sys, os, json, numpy as np
from scipy.optimize import least_squares
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K
from calib import cam2ecm_mm
FX, CX, CY = 1860.724932, 690.538345, 576.803638; FB = FX * 5.4476353950418295
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); SUJ = np.array(cal["frozen"]["suj_psm1"]); pad = json.load(open(R0 + "/D_pad/pad.json")); nP = np.array(pad["normal"]); dP = pad["d"] + pad["margin_mm"] * 1e-3
UVD = np.array([[760,400,146.3],[737,385,146.4],[715,370,146.3],[693,355,149.0],[672,340,150.5],[651,325,152.3],[630,310,155.7],[598,288,173.4],[590,284,171.3],[598,297,176.1],[607,308,177.8],[616,318,181.8],[625,328,181.7],[632,336,182.7]], float)
st = json.load(open("tr3/gview/state.json")); Tg = K.fk_ecm(np.array(st["js"][:6]), SUJ); Z = FB / UVD[:, 2]; Xc = np.c_[(UVD[:, 0] - CX) * Z / FX, (UVD[:, 1] - CY) * Z / FX, Z]
Pt = np.array([Tg[:3, :3].T @ (cam2ecm_mm(x, M) * 1e-3 - Tg[:3, 3]) for x in Xc]); c = Pt.mean(0); nt = np.linalg.svd(Pt - c)[2][2]
disp = np.load("padcomp/fs_disp.npy"); u, v, h = map(float, sys.argv[1:4]); rolls_avoid = [float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else []
Zp = FB / np.median(disp[int(v) - 5:int(v) + 6, int(u) - 5:int(u) + 6]); Zh = Zp - h; g = cam2ecm_mm(np.array([(u - CX) * Zh / FX, (v - CY) * Zh / FX, Zh]), M) * 1e-3
res = []
for r in range(-240, 241, 20):
    for p in range(-70, 71, 10):
        for y in range(-70, 71, 10):
            wr = np.radians([r, p, y]); s = least_squares(lambda x: (K.fk_ecm(np.r_[x, wr], SUJ)[:3, 3] - g) * 1e3, np.array([-1.3, -0.45, 0.14]), bounds=(K.QMIN[:3] + [0.05, 0.05, 0.01], K.QMAX[:3] - [0.05, 0.05, 0.01]))
            q = np.r_[s.x, wr]; T = K.fk_ecm(q, SUJ)
            if np.linalg.norm(T[:3, 3] - g) > 1e-4: continue
            m = K.margins(q)
            if m[3:].min() < 5 or m[:2].min() < 5 or m[2] < 5: continue
            P = (T[:3, :3] @ Pt.T).T + T[:3, 3]; hmin = (P @ nP + dP).min() * 1e3; tilt = np.degrees(np.arccos(min(1, abs((T[:3, :3] @ nt) @ nP))))
            if hmin < 2: continue
            from rig import TCP_MM
            Q = np.vstack([P, T[:3, 3], T[:3, 3] + T[:3, :3] @ (TCP_MM * 1e-3)]); X = np.c_[Q * 1e3, np.ones(len(Q))] @ M; uu, vv = FX * X[:, 0] / X[:, 2] + CX, FX * X[:, 1] / X[:, 2] + CY
            if uu.min() < 80 or uu.max() > 1220 or vv.min() < 80 or vv.max() > 940: continue   # 2026-11-05: needle + jaw inside the camera view
            res.append((tilt, hmin, r, p, y, q))
res.sort(key=lambda x: x[0] + 0.3 * max(0, x[1] - 5) - 2 * min(min([abs(x[2] - a) for a in rolls_avoid] or [90]), 90) / 90)
for t, hm, r, p, y, q in res[:6]: print("%.4f %.4f %.1f %d %d %d   # needle plane tilt %.0f deg, needle lowest %.1f mm above pad" % (q[0], q[1], q[2] * 1e3, r, p, y, t, hm))
