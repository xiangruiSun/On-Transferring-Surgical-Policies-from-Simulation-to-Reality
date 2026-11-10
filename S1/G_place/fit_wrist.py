#!/usr/bin/env python3
"""Fit today's wrist-joint offsets delta (roll, pitch, yaw) and the needle in the tool frame from the held-needle arc seen at several
wrist poses (collect_wrist.py output).  Model: X_cam = [ (SUJ x FK(q + delta) x p_tool) mm, 1 ] @ M  (C_r6cal affine);
needle = circle of radius --radius in the tool frame (centre c, normal angles, tip angle, span).  Residuals in the right image: model
arc -> detected arc, detected arc -> model arc, tip -> tip.  Multi-start over delta.  No robot access.

    python3 fit_wrist.py --data wrist3 --roi 330 720 --jaw 80 920
"""
import argparse, json, os, sys
import numpy as np
from scipy.optimize import least_squares
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code"); sys.path.insert(0, R0 + "/G_place")
sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib"))
import cv2
import kin as K
import tipmeas, needle_circle as NC
from rig import FX, CX, CY

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--roi", type=float, nargs=2, required=True)
ap.add_argument("--jaw", type=float, nargs=2, required=True); ap.add_argument("--radius", type=float, default=9.7); a = ap.parse_args()
D = a.data if os.path.isabs(a.data) else R0 + "/G_place/" + a.data
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); SUJ = np.array(cal["frozen"]["suj_psm1"])
col = json.load(open(D + "/collect.json"))
obs = []
roi, jaw = np.array(a.roi), np.array(a.jaw)
for r in col["rows"]:
    img = cv2.imread(f"{D}/p{r['k']:02d}/right_rect.png"); det = tipmeas.detect_roi(img, 110.0, roi, 320, jaw)
    if det is None and r.get("roi") is not None: det = tipmeas.detect_roi(img, 110.0, np.array(r["roi"]), 320, np.array(r["jaw"]))
    if det is None: print("pose", r["k"], "no arc"); continue
    arc = NC.dense_arc(det, 120); obs.append(dict(k=r["k"], q=np.array(r["js"]), arc=arc, tip=arc[-1]))
    roi = arc.mean(0); jaw = arc[0]
print(f"{len(obs)} poses with an arc")

def proj_ecm(P_m):
    X = np.c_[P_m * 1e3, np.ones(len(P_m))] @ M; return np.c_[FX * X[:, 0] / X[:, 2] + CX, FX * X[:, 1] / X[:, 2] + CY]

def needle_tool(x, n_s=24):
    c = x[3:6] * 1e-3; th, ph = x[6], x[7]; nrm = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    e1, e2 = NC._basis(nrm); t = x[8] + np.sign(x[9]) * np.radians(60) * np.linspace(0, 1, n_s)   # 13:5x: only the 60 deg next to the tip (detection is partial); sign of x[9] = direction
    return c[None] + a.radius * 1e-3 * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)   # [0] = tip

def resid(x):
    out = []; d = np.r_[0, 0, 0, x[0], x[1], x[2]]; Pt = needle_tool(x)
    for o in obs:
        T = K.fk_ecm(o["q"] + d, SUJ); P = (T[:3, :3] @ Pt.T).T + T[:3, 3]; uv = proj_ecm(P)
        d1 = np.min(np.linalg.norm(uv[:, None] - o["arc"][None, ::4], axis=2), axis=1)
        out += [d1, 6.0 * (uv[0] - o["tip"])]
    return np.concatenate([np.ravel(v) for v in out])

# init of the needle centre: tip of pose 0 is ~ on the ray; put the needle ~25 mm ahead of the hinge along tool z, radius off-axis
cands = []
for dr in np.radians([-135, -90, -45, 0, 45, 90, 135, 180]):
    for dp in np.radians([-40, 0, 40]):
        for dy in np.radians([-40, 0, 40]):
            for nrm0 in ((0.0, 0.0), (np.pi / 2, 0.0), (np.pi / 2, np.pi / 2)):
                for sp in (np.radians(140), -np.radians(140)):
                    x0 = np.r_[dr, dp, dy, 0.0, -5.0, 30.0, nrm0[0], nrm0[1], 0.0, sp]
                    try: s_ = least_squares(resid, x0, loss="soft_l1", f_scale=5.0, max_nfev=40, x_scale=np.r_[0.1, 0.1, 0.1, 3, 3, 3, 0.1, 0.1, 0.1, 0.1])
                    except Exception: continue
                    cands.append((float(np.median(np.abs(s_.fun))), s_.x))
cands.sort(key=lambda t: t[0]); print("coarse best", [round(c_[0], 2) for c_ in cands[:6]], flush=True)
best = None
for c0, x0 in cands[:6]:
    s_ = least_squares(resid, x0, loss="soft_l1", f_scale=3.0, max_nfev=600, x_scale=np.r_[0.1, 0.1, 0.1, 3, 3, 3, 0.1, 0.1, 0.1, 0.1]); c_ = float(np.median(np.abs(s_.fun)))
    print(f"  refine {c0:.2f} -> {c_:.2f} px, delta {np.round(np.degrees(s_.x[:3]), 1).tolist()}", flush=True)
    if best is None or c_ < best[0]: best = (c_, s_.x)
c, x = best
s = least_squares(resid, x, loss="soft_l1", f_scale=3.0, max_nfev=2000, x_scale=np.r_[0.1, 0.1, 0.1, 3, 3, 3, 0.1, 0.1, 0.1, 0.1]); x = s.x
r = resid(x); per = []
for i, o in enumerate(obs):
    d = np.r_[0, 0, 0, x[0], x[1], x[2]]; T = K.fk_ecm(o["q"] + d, SUJ); Pt = needle_tool(x); uv = proj_ecm((T[:3, :3] @ Pt.T).T + T[:3, 3])
    per.append(dict(k=o["k"], tip_err_px=float(np.linalg.norm(uv[0] - o["tip"])), arc_med_px=float(np.median(np.min(np.linalg.norm(uv[:, None] - o["arc"][None], axis=2), axis=1)))))
res = dict(delta_deg=np.degrees(x[:3]).tolist(), needle_x=x.tolist(), radius_mm=a.radius, median_px=float(np.median(np.abs(r))), per_pose=per, n_poses=len(obs), data=D)
json.dump(res, open(D + "/fit.json", "w"), indent=1)
print("delta (roll, pitch, yaw) deg", np.round(np.degrees(x[:3]), 1).tolist(), "| median residual %.2f px" % res["median_px"])
for p in per: print(f"  pose {p['k']}: tip {p['tip_err_px']:.1f} px, arc median {p['arc_med_px']:.1f} px")
# overlay for pose 0 and the extreme poses
for o in obs[:1] + obs[-2:]:
    d = np.r_[0, 0, 0, x[0], x[1], x[2]]; T = K.fk_ecm(o["q"] + d, SUJ); uv = proj_ecm((T[:3, :3] @ needle_tool(x, 60).T).T + T[:3, 3])
    img = cv2.imread(f"{D}/p{o['k']:02d}/right_rect.png"); cv2.polylines(img, [np.int32(o["arc"])], False, (0, 255, 255), 2); cv2.polylines(img, [np.int32(uv)], False, (255, 0, 255), 2)
    cv2.imwrite(f"{D}/fit_p{o['k']:02d}.png", img)
print("->", D + "/fit.json")
