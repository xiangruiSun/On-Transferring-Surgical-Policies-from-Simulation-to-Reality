import json, os, sys
import numpy as np, cv2
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K
from rig import FX, CX, CY, FB, TCP_MM
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); SUJ = np.array(cal["frozen"]["suj_psm1"])
A = 10.7e-3
for vd in sys.argv[1:]:
    st = json.load(open(vd + "/state.json")); q = np.array(st["q"]); T = K.fk_ecm(q, SUJ); R = cv2.imread(vd + "/right_rect.png")
    # wrist pitch axis point: FK with the last link: hinge at cp, wrist-pitch centre is A behind along the link; use q with yaw/pitch zero for the shaft end
    q0 = q.copy(); q0[4:] = 0; T0 = K.fk_ecm(q0, SUJ)
    def c(p, TT=T): return np.r_[(TT[:3, 3] + TT[:3, :3] @ (np.asarray(p) * 1e-3)) * 1e3, 1] @ M
    def uv(X): return np.array([FX * X[0] / X[2] + CX, FX * X[1] / X[2] + CY])
    pts = dict(hinge=c([0, 0, 0]), tcp=c(TCP_MM), mid=c(TCP_MM * .5), hinge_zerowrist=c([0, 0, 0], T0))
    print(vd, "wrist", np.round(np.degrees(q[3:]), 0).tolist(), {k: (np.round(uv(v)).tolist(), round(float(v[2]), 1)) for k, v in pts.items()})
    for (k, v), col in zip(pts.items(), ((0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255))):
        p = uv(v)
        if 0 <= p[0] < R.shape[1] and 0 <= p[1] < R.shape[0]: cv2.drawMarker(R, tuple(np.int32(p)), col, cv2.MARKER_CROSS, 40, 2); cv2.putText(R, k, tuple(np.int32(p + 12)), 0, .6, col, 2)
    cv2.imwrite(vd + "/kin.png", R)
