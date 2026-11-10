#!/usr/bin/env python3
"""Wrist probes (2026-10-30 ~14:4x): move_jp to q0 + listed wrist offsets (deg: roll pitch yaw), stereo snapshot each, back to q0.
Wrist-only (outer joints / insertion unchanged); offsets clipped 5 deg inside the joint limits; jaw effort logged.

    python3 probe_wrist2.py --out DIR --offs "0,45,0;0,-45,0;..."
"""
import argparse, json, os, sys, time
import numpy as np
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code"); sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib"))
import cv2
import kin as K
from sensor_msgs.msg import JointState
from rig import RigIO
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--offs", required=True); ap.add_argument("--no-return", action="store_true"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
offs = [np.radians([float(v) for v in s.split(",")]) for s in a.offs.split(";")]
io = RigIO(images=True); t0 = time.time()
while (io.get("js") is None or io.get("jaw_eff") is None) and time.time() - t0 < 10: time.sleep(0.1)
time.sleep(1.0); pub = io.node.create_publisher(JointState, '/PSM1/move_jp', 1)
q0 = np.array(io.get("spj")[1][:6]) if io.get("spj") else np.array(io.get("js")[1][:6])
lo, hi = K.QMIN + np.radians(5), K.QMAX - np.radians(5); rows = []; qp = q0
for k, o in enumerate(offs):
    q = q0.copy(); q[3:] = np.clip(q0[3:] + o, lo[3:], hi[3:]); m = JointState(); m.position = q.tolist(); pub.publish(m)
    time.sleep(2.0 + 2.0 * float(np.abs(q - qp).max())); qp = q
    b = io.frames(6); dd = f"{a.out}/p{k:02d}"; os.makedirs(dd, exist_ok=True)
    cv2.imwrite(dd + "/right_rect.png", np.mean([x[1] for x in b], 0).round().astype(np.uint8)); cv2.imwrite(dd + "/left_rect.png", np.mean([x[0] for x in b], 0).round().astype(np.uint8))
    js = np.array(io.get("js")[1][:6]); rows.append(dict(k=k, off_deg=np.degrees(o).tolist(), q_cmd=q.tolist(), js=js.tolist(), jaw_eff=float(io.get("jaw_eff")[1][0])))
    json.dump(rows[-1], open(dd + "/state.json", "w"))
    print(k, np.round(np.degrees(q[3:] - q0[3:]), 0).tolist(), "js err deg", np.round(np.degrees(js[3:] - q[3:]), 2).tolist(), "jaw eff %.3f" % rows[-1]["jaw_eff"], flush=True)
if not a.no_return:
    m = JointState(); m.position = q0.tolist(); pub.publish(m); time.sleep(2.0 + 2.0 * float(np.abs(qp - q0).max()))
json.dump(dict(q0=q0.tolist(), rows=rows), open(a.out + "/probe.json", "w"), indent=1); print("done", flush=True); sys.stdout.flush(); os._exit(0)
