#!/usr/bin/env python3
"""Orientation for the revised placing definition with the fitted wrist model (fit_wrist.py): joints q such that
FK(q + delta) x needle_tool puts the needle tip at e + h n with the first 60 deg of the arc in span(n, u) rising from the tip
(tip tangent -n).  IK = least squares over the 6 joints from the current ones (joint limits, margins).  --send: one move_jp to q
(the dVRK interpolates), then a snapshot with the predicted vs detected needle.  Run high (h >= 20 mm) first.

    python3 orient_v9.py --scene scene3/scene.json --fit wrist3/fit.json --h 25 [--send]
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.optimize import least_squares
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code"); sys.path.insert(0, R0 + "/G_place")
sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib"))
import cv2
import kin as K
import needle_circle as NC
from rig import RigIO, FX, CX, CY
from sensor_msgs.msg import JointState

ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--fit", required=True); ap.add_argument("--h", type=float, default=25.0)
ap.add_argument("--send", action="store_true"); ap.add_argument("--ins-extra-mm", type=float, default=0.0); ap.add_argument("--out", default=None); a = ap.parse_args()
G = R0 + "/G_place"; sc = json.load(open(G + "/" + a.scene)); fit = json.load(open(G + "/" + a.fit))
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); SUJ = np.array(cal["frozen"]["suj_psm1"])
e, n, u = np.array(sc["entry"]), np.array(sc["normal"]), np.array(sc["u"]); x = np.array(fit["needle_x"]); Rn = fit["radius_mm"]; dl = np.r_[0, 0, 0, x[0], x[1], x[2]]
out = a.out or G + f"/v9_h{a.h:g}_" + time.strftime("%H%M%S"); os.makedirs(out, exist_ok=True)

def needle_tool(n_s=13):
    c = x[3:6] * 1e-3; th, ph = x[6], x[7]; nrm = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    e1, e2 = NC._basis(nrm); t = x[8] + np.sign(x[9]) * np.radians(60) * np.linspace(0, 1, n_s)
    return c[None] + Rn * 1e-3 * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)
Pt = needle_tool()
phi = np.radians(np.linspace(0, 60, 13)); cstar = e + a.h * n + Rn * u
target = cstar[None] + Rn * (-np.cos(phi)[:, None] * u[None] + np.sin(phi)[:, None] * n[None])      # [0] = tip, rising along +n toward u
def cam(q):
    T = K.fk_ecm(q + dl, SUJ); P = (T[:3, :3] @ Pt.T).T + T[:3, 3]; return np.c_[P * 1e3, np.ones(len(P))] @ M
def proj(X): X = np.atleast_2d(X); return np.c_[FX * X[:, 0] / X[:, 2] + CX, FX * X[:, 1] / X[:, 2] + CY]
io = RigIO(images=True); t0 = time.time()
while io.get("js") is None and time.time() - t0 < 10: time.sleep(0.1)
time.sleep(0.5); q0 = np.array(io.get("js")[1][:6])
lo, hi = K.QMIN + np.r_[np.radians([5, 5]), 0.01, np.radians([5, 5, 5])], K.QMAX - np.r_[np.radians([5, 5]), 0.01, np.radians([5, 5, 5])]
best = None
for droll in np.radians([0, 90, -90, 180]):
    qi = np.clip(q0 + np.r_[0, 0, 0, droll, 0, 0], lo, hi)
    s = least_squares(lambda q: np.ravel(cam(q) - target) * np.r_[[3.0] * 3, [1.0] * 36], qi, bounds=(lo, hi))
    err = np.linalg.norm(cam(s.x) - target, axis=1)
    if best is None or err.max() < best[0]: best = (float(err.max()), s.x, err)
emax, q, err = best
print(f"IK: max needle-point error {emax:.2f} mm (tip {err[0]:.2f}); joints {np.round(q, 4).tolist()}; change from now {np.round(np.degrees(q - q0), 1).tolist()} deg/(ins in rad)")
print("margins", np.round(K.margins(q), 1).tolist())
img = cv2.imread(sc["snap"] + "/right_rect.png")
if a.send:
    if emax > 2.0: raise SystemExit("IK error > 2 mm: not sent")
    q = q.copy(); q[2] -= a.ins_extra_mm * 1e-3                                                  # extra retraction along the shaft (safety), translations come back
    pub = io.node.create_publisher(JointState, '/PSM1/move_jp', 1); m = JointState(); m.position = q.tolist(); pub.publish(m)
    time.sleep(2.0 + 2.5 * float(np.max(np.abs(q - q0)[[0, 1, 3, 4, 5]]))); print("moved; measured", np.round(io.get("js")[1][:6], 4).tolist(), "jaw eff %.3f" % io.get("jaw_eff")[1][0])
    b = io.frames(8); img = np.mean([z[1] for z in b], 0).round().astype(np.uint8); cv2.imwrite(out + "/right_rect.png", img); cv2.imwrite(out + "/left_rect.png", np.mean([z[0] for z in b], 0).round().astype(np.uint8))
ov = img.copy(); cv2.polylines(ov, [np.int32(proj(target))], False, (0, 255, 0), 2); cv2.polylines(ov, [np.int32(proj(cam(q)))], False, (255, 0, 255), 1)
cv2.circle(ov, tuple(np.int32(proj(e)[0])), 7, (0, 0, 255), 2); cv2.circle(ov, tuple(np.int32(proj(target[0])[0])), 6, (0, 255, 0), 2)
cv2.imwrite(out + "/overlay.png", ov); json.dump(dict(q=q.tolist(), q0=q0.tolist(), err=err.tolist(), h=a.h), open(out + "/orient.json", "w"), indent=1)
print("->", out, flush=True); sys.stdout.flush(); os._exit(0)
