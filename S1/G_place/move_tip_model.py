#!/usr/bin/env python3
"""Translate (orientation fixed) so that the MODEL-predicted needle tip (fitted wrist model, fit_wrist.py) goes to e + h n.
Mover guards (pad clearance with this tip, IK, tracking).  Then a snapshot with the predicted needle drawn.

    python3 move_tip_model.py --scene scene3/scene.json --fit wrist3/fit.json --h 10 [--send]
"""
import argparse, json, os, sys, time
import numpy as np
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code"); sys.path.insert(0, R0 + "/G_place")
sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib"))
import cv2
import kin as K
import needle_circle as NC
from rig import RigIO, Mover, load_kin, TCP_MM, FX, CX, CY
from calib import cam2ecm_mm

ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--fit", required=True); ap.add_argument("--h", type=float, required=True)
ap.add_argument("--send", action="store_true"); ap.add_argument("--out", required=True); ap.add_argument("--max-mm", type=float, default=40.0); a = ap.parse_args()
G = R0 + "/G_place"; sc = json.load(open(G + "/" + a.scene)); fit = json.load(open(G + "/" + a.fit)); os.makedirs(a.out, exist_ok=True)
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); SUJ = np.array(cal["frozen"]["suj_psm1"]); pad = json.load(open(R0 + "/D_pad/pad.json"))
e, n, u = np.array(sc["entry"]), np.array(sc["normal"]), np.array(sc["u"]); x = np.array(fit["needle_x"]); Rn = fit["radius_mm"]; dl = np.r_[0, 0, 0, x[0], x[1], x[2]]
def needle_tool(n_s=13):
    c = x[3:6] * 1e-3; th, ph = x[6], x[7]; nrm = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    e1, e2 = NC._basis(nrm); t = x[8] + np.sign(x[9]) * np.radians(60) * np.linspace(0, 1, n_s)
    return c[None] + Rn * 1e-3 * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)
Pt = needle_tool()
def ecm_pts(q): T = K.fk_ecm(q + dl, SUJ); return (T[:3, :3] @ Pt.T).T + T[:3, 3]
def proj(X): X = np.atleast_2d(X); return np.c_[FX * X[:, 0] / X[:, 2] + CX, FX * X[:, 1] / X[:, 2] + CY]
def log(r): pass
io = RigIO(images=True); kin = load_kin(); t0 = time.time()
while io.get("js") is None and time.time() - t0 < 10: time.sleep(0.1)
time.sleep(1.0); mv = Mover(io, kin, log)
mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = SUJ; mv.pad = (np.array(pad["normal"]), pad["d"]); mv.cube = None
q = np.array(io.get("js")[1][:6]); P = ecm_pts(q); tip_e = P[0]; tip_c = np.r_[tip_e * 1e3, 1] @ M
S = mv.setpoint(); mv.tip_off_tool = S[:3, :3].T @ (tip_e - S[:3, 3]) - TCP_MM * 1e-3
goal_c = e + a.h * n; d_c = goal_c - tip_c; d_e = d_c @ np.linalg.inv(M[:3]) * 1e-3
print(f"model tip (cam) {np.round(tip_c, 1).tolist()} -> goal {np.round(goal_c, 1).tolist()}: move {np.linalg.norm(d_c):.1f} mm (cam); tip pad clearance now {mv.pad_dist_mm(mv.tip(S)):.1f} mm", flush=True)
if np.linalg.norm(d_c) > a.max_mm: raise SystemExit("move too large")
if a.send: mv.translate(d_e, tag="model_tip", lim=(2.0, 3.0)); time.sleep(0.5)
b = io.frames(8); img = np.mean([z[1] for z in b], 0).round().astype(np.uint8); cv2.imwrite(a.out + "/right_rect.png", img); cv2.imwrite(a.out + "/left_rect.png", np.mean([z[0] for z in b], 0).round().astype(np.uint8))
q = np.array(io.get("js")[1][:6]); Pc = np.c_[ecm_pts(q) * 1e3, np.ones(len(Pt))] @ M
ov = img.copy(); cv2.polylines(ov, [np.int32(proj(Pc))], False, (255, 0, 255), 2); cv2.circle(ov, tuple(np.int32(proj(Pc[0])[0])), 7, (255, 0, 255), 2)
cv2.circle(ov, tuple(np.int32(proj(e)[0])), 7, (0, 0, 255), 2); cv2.line(ov, tuple(np.int32(proj(e)[0])), tuple(np.int32(proj(e + 8 * n)[0])), (0, 255, 0), 2)
cv2.imwrite(a.out + "/overlay.png", ov); json.dump(dict(q=q.tolist(), tip_cam=Pc[0].tolist()), open(a.out + "/state.json", "w"))
print("->", a.out, flush=True); sys.stdout.flush(); os._exit(0)
