#!/usr/bin/env python3
"""Wrist-offset data (2026-10-30): with the needle held and the arm lifted, move_jp through wrist-joint offsets around the current
joints (roll / wrist pitch / wrist yaw), capture stereo frames + joints at each, return to the start joints.  Output <out>/pNN/."""
import argparse, json, os, sys, time, threading
import numpy as np
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code")
import cv2, rclpy
from sensor_msgs.msg import JointState
from rig import RigIO
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--deg", type=float, default=20.0); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
io = RigIO(images=True); t0 = time.time()
while (io.get("js") is None or io.get("jaw_eff") is None) and time.time() - t0 < 10: time.sleep(0.1)
time.sleep(1.0)
pub = io.node.create_publisher(JointState, '/PSM1/move_jp', 1) if hasattr(io, "node") else None
if pub is None: raise SystemExit("RigIO has no node")
q0 = np.array(io.get("spj")[1][:6]) if io.get("spj") else np.array(io.get("js")[1][:6])
d = np.radians(a.deg)
offs = [np.zeros(6)]
for j, s in ((3, 1), (3, -1), (4, 1), (4, -1), (5, 1), (5, -1)):
    o = np.zeros(6); o[j] = s * d; offs.append(o)
offs.append(np.r_[0, 0, 0, d, d, 0]); offs.append(np.r_[0, 0, 0, -d, 0, d]); offs.append(np.r_[0, 0, 0, 0, -d, -d])
rows = []
for k, o in enumerate(offs):
    q = q0 + o; m = JointState(); m.position = q.tolist(); pub.publish(m); time.sleep(2.5 + 3.0 * np.abs(o).max())
    b = io.frames(6); R = np.mean([x[1] for x in b], 0).round().astype(np.uint8); L = np.mean([x[0] for x in b], 0).round().astype(np.uint8)
    dd = f"{a.out}/p{k:02d}"; os.makedirs(dd, exist_ok=True); cv2.imwrite(dd + "/right_rect.png", R); cv2.imwrite(dd + "/left_rect.png", L)
    js = np.array(io.get("js")[1][:6]); rows.append(dict(k=k, offset=o.tolist(), q_cmd=q.tolist(), js=js.tolist(), jaw=float(io.get("jaw")[1][0]), jaw_eff=float(io.get("jaw_eff")[1][0])))
    json.dump(rows[-1], open(dd + "/state.json", "w")); print(k, np.round(np.degrees(o[3:]), 0).tolist(), "js err deg", np.round(np.degrees(js[3:] - q[3:]), 2).tolist(), "jaw eff %.3f" % rows[-1]["jaw_eff"], flush=True)
m = JointState(); m.position = q0.tolist(); pub.publish(m); time.sleep(4.0)
json.dump(dict(q0=q0.tolist(), rows=rows), open(a.out + "/collect.json", "w"), indent=1); print("back at q0", flush=True); sys.stdout.flush(); os._exit(0)
