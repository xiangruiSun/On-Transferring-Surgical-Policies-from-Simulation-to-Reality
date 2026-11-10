"""READ-ONLY: joints (j0 j1 ins) that put the jaw hinge at right-image pixel (u, v), h mm above the pad (pad-only depth), wrist given. Prints the q line for start_pose.py."""
import sys, os, json, numpy as np
from scipy.optimize import least_squares
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K
from calib import cam2ecm_mm
FX, CX, CY = 1860.724932, 690.538345, 576.803638; FB = FX * 5.4476353950418295
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); SUJ = np.array(cal["frozen"]["suj_psm1"]); disp = np.load(sys.argv[1])
u, v, h, r, p, y = map(float, sys.argv[2:8]); Zp = FB / np.median(disp[int(v) - 5:int(v) + 6, int(u) - 5:int(u) + 6]); Z = Zp - h; X = np.array([(u - CX) * Z / FX, (v - CY) * Z / FX, Z]); g = cam2ecm_mm(X, M) * 1e-3
wr = np.radians([r, p, y]); f = lambda x: (K.fk_ecm(np.r_[x, wr], SUJ)[:3, 3] - g) * 1e3
s = least_squares(f, np.array([-1.25, -0.45, 0.13]), bounds=(K.QMIN[:3] + [0.05, 0.05, 0.01], K.QMAX[:3] - [0.05, 0.05, 0.01])); q = np.r_[s.x, wr]
print("%.4f %.4f %.1f %.0f %.0f %.0f   # residual %.2f mm, pad Z %.1f, margins %s" % (s.x[0], s.x[1], s.x[2] * 1e3, r, p, y, np.linalg.norm(f(s.x)), Zp, np.round(K.margins(q), 1).tolist()))
