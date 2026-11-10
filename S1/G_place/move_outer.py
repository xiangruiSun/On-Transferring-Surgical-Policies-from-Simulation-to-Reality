#!/usr/bin/env python3
"""Translate with the OUTER joints only (yaw / pitch / insertion), wrist joints unchanged (2026-10-30 ~15:0x: keeping the orientation
in Cartesian needs wrist yaw beyond its margin; the shaft re-orients by a few degrees instead).  Target = current cp + camera vector
(mm, optionally projected on the pad plane) mapped with C_r6cal M; q1..q3 by least squares; move_jp in sub-steps of <= 2 mm,
joint tracking checked after each; cp pad clearance must stay >= 5 mm.  Then a stereo snapshot.

    python3 move_outer.py --v 18 12 0 --pad-parallel --out DIR [--send]
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.optimize import least_squares
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code"); sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib"))
import cv2
import kin as K
from sensor_msgs.msg import JointState
from rig import RigIO

ap = argparse.ArgumentParser(); ap.add_argument("--v", type=float, nargs=3, required=True); ap.add_argument("--pad-parallel", action="store_true")
ap.add_argument("--out", required=True); ap.add_argument("--send", action="store_true"); ap.add_argument("--min-clear", type=float, default=5.0); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); SUJ = np.array(cal["frozen"]["suj_psm1"]); pad = json.load(open(R0 + "/D_pad/pad.json"))
nE = np.array(pad["normal"]); d_e = np.array(a.v) @ np.linalg.inv(M[:3]) * 1e-3
if a.pad_parallel: d_e = d_e - (d_e @ nE) * nE
if np.linalg.norm(d_e) * 1e3 > 30.0: raise SystemExit("move > 30 mm refused")
io = RigIO(images=True); t0 = time.time()
while (io.get("js") is None or io.get("spj") is None) and time.time() - t0 < 10: time.sleep(0.1)
time.sleep(1.0); pub = io.node.create_publisher(JointState, '/PSM1/move_jp', 1)
ecm = np.asarray(io.get("ecm")[1]); assert np.abs(ecm - np.array(cal["frozen"]["ecm_js"])).max() < 2e-3, "ECM moved"
q0 = np.array(io.get("spj")[1][:6]); p0 = K.fk_ecm(q0, SUJ)[:3, 3]; goal = p0 + d_e
lo, hi = K.QMIN[:3] + np.r_[np.radians([3, 3]), 0.005], K.QMAX[:3] - np.r_[np.radians([3, 3]), 0.005]
s = least_squares(lambda x: (K.fk_ecm(np.r_[x, q0[3:]], SUJ)[:3, 3] - goal) * 1e3, q0[:3], bounds=(lo, hi))
q1 = np.r_[s.x, q0[3:]]; err = float(np.linalg.norm(s.fun)); clr = lambda q: float((nE @ K.fk_ecm(q, SUJ)[:3, 3] + pad["d"]) * 1e3)
R0m, R1m = K.fk_ecm(q0, SUJ)[:3, :3], K.fk_ecm(q1, SUJ)[:3, :3]; rot = float(np.degrees(np.arccos(np.clip((np.trace(R0m.T @ R1m) - 1) / 2, -1, 1))))
print(f"cp move {np.linalg.norm(d_e) * 1e3:.1f} mm (ECM {np.round(d_e * 1e3, 1).tolist()}), IK err {err:.2f} mm; outer joints change {np.round(np.degrees(q1[:2] - q0[:2]), 2).tolist()} deg, "
      f"ins {1e3 * (q1[2] - q0[2]):.1f} mm; tool re-orients {rot:.1f} deg; cp pad clearance {clr(q0):.1f} -> {clr(q1):.1f} mm", flush=True)
log = open(a.out + "/log.jsonl", "a"); log.write(json.dumps(dict(t=time.time(), q0=q0.tolist(), q1=q1.tolist(), d_e=d_e.tolist(), ik_err=err)) + "\n")
if a.send:
    if err > 0.2 or clr(q1) < a.min_clear: raise SystemExit("not sent: IK error / clearance")   # 15:2x --min-clear 0: kinematic cp vs pad ~8 mm pessimistic vs jaw stereo at the site
    n = max(1, int(np.ceil(np.linalg.norm(d_e) * 1e3 / 2.0)))
    for k in range(1, n + 1):
        q = q0 + (q1 - q0) * k / n; m = JointState(); m.position = q.tolist(); pub.publish(m); time.sleep(0.8)
        js = np.array(io.get("js")[1][:6]); je = js - q
        if abs(je[2]) * 1e3 > 0.6 or np.degrees(np.abs(je[:2])).max() > 1.0:
            print(f"STOP at sub-step {k}/{n}: tracking ins {je[2] * 1e3:.2f} mm, outer {np.round(np.degrees(je[:2]), 2).tolist()} deg", flush=True); break
    else: print(f"moved in {n} sub-steps; jaw eff {io.get('jaw_eff')[1][0]:.3f}", flush=True)
b = io.frames(8); cv2.imwrite(a.out + "/right_rect.png", np.mean([z[1] for z in b], 0).round().astype(np.uint8)); cv2.imwrite(a.out + "/left_rect.png", np.mean([z[0] for z in b], 0).round().astype(np.uint8))
json.dump(dict(js=np.asarray(io.get("js")[1][:6]).tolist()), open(a.out + "/state.json", "w"))
print("->", a.out, flush=True); sys.stdout.flush(); os._exit(0)
