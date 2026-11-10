import json, os, sys
import numpy as np, cv2
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K
from rig import FX, CX, CY, FB, TCP_MM
snapdir, demo, out = sys.argv[1:4]
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); d = json.load(open(demo))
T = K.fk_ecm(np.array(d["js_mean"]), np.array(d["suj"])); print("margins", np.round(K.margins(np.array(d["js_mean"])), 0).tolist())
R = cv2.imread(snapdir + "/right_rect.png"); L = cv2.imread(snapdir + "/left_rect.png")
def cam(p_tool_mm): return np.r_[(T[:3, 3] + T[:3, :3] @ (np.asarray(p_tool_mm) * 1e-3)) * 1e3, 1] @ M
def uv(X): return np.array([FX * X[0] / X[2] + CX, FX * X[1] / X[2] + CY])
for name, p, col in (("hinge", [0, 0, 0], (0, 255, 255)), ("tcp", TCP_MM, (0, 255, 0)), ("mid", TCP_MM * 0.5, (255, 255, 0))):
    X = cam(p); a = uv(X); print(name, "cam", np.round(X, 1).tolist(), "right uv", np.round(a).tolist(), "disp %.1f" % (FB / X[2]), "left uv", np.round(a - [FB / X[2], 0]).tolist())
    if 0 <= a[0] < R.shape[1] and 0 <= a[1] < R.shape[0]: cv2.drawMarker(R, tuple(np.int32(a)), col, cv2.MARKER_CROSS, 40, 2); cv2.putText(R, name, tuple(np.int32(a + 12)), 0, 0.7, col, 2)
pts = [uv(cam(TCP_MM * f)) for f in np.linspace(-2, 1, 13)]; cv2.polylines(R, [np.int32(pts)], False, (0, 200, 0), 1)
cv2.imwrite(out, R)
