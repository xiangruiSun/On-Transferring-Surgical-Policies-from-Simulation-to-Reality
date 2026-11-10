#!/usr/bin/env python3
"""READ-ONLY: record the user's demonstrated placing pose (revised definition 2026-10-30: needle plane vertical, tip lowest at the
entry sticker, arc toward the exit sticker).  Saves joints (PSM1 + jaw, 2 s mean and spread), setpoint / measured cp, SUJ, ECM,
a stereo snapshot (+ FS on the 3090) and the YOLO sticker boxes.  Sends no command.

    python3 record_demo.py --name demo1
"""
import argparse, json, os, sys, time
import numpy as np
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code"); sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261027/place/code"))
import cv2
from rig import RigIO
import calib as C

ap = argparse.ArgumentParser(); ap.add_argument("--name", default="demo1"); ap.add_argument("--no-fs", action="store_true"); a = ap.parse_args()
d = f"{R0}/G_place/{a.name}"
if os.path.exists(d): raise SystemExit(f"{d} exists")
io = RigIO(images=True); time.sleep(2.5)
J, W = [], []
t0 = time.time()
while time.time() - t0 < 2.0:
    J.append(np.asarray(io.get("js")[1])[:6]); W.append(float(io.get("jaw")[1][0])); time.sleep(0.05)
J = np.array(J); C.save_snap(io, d + "/snap")
rec = dict(t=time.time(), js_mean=J.mean(0).tolist(), js_std=J.std(0).tolist(), jaw_mean=float(np.mean(W)), jaw_eff=float(io.get("jaw_eff")[1][0]),
           sp=np.asarray(io.get("sp")[1]).tolist(), cp=np.asarray(io.get("cp")[1]).tolist(), suj=np.asarray(io.get("suj")[1]).tolist(),
           ecm=np.asarray(io.get("ecm")[1]).tolist(), op=io.get("opstate")[1] if io.get("opstate") else None)
json.dump(rec, open(d + "/demo.json", "w"), indent=1)
print("joints", np.round(J.mean(0), 4).tolist(), "std", np.round(J.std(0), 5).tolist(), "jaw", round(rec["jaw_mean"], 3), "eff", round(rec["jaw_eff"], 3), flush=True)
if not a.no_fs:
    try: C.run_fs([d + "/snap"]); print("FS ok", flush=True)
    except Exception as e: print("FS failed:", e, flush=True)
    try:
        from remote import Remote, deploy
        deploy(); yo = Remote("yolo", d); img = cv2.imread(d + "/snap/right_rect.png"); ym, _ = yo.call({"imgsz": 960, "conf": 0.05}, {"image": img}); yo.close()
        json.dump(dict(boxes=ym["boxes"]), open(d + "/yolo.json", "w")); print("YOLO boxes", [[round(x, 2) for x in b[:5]] for b in ym["boxes"][:4]], flush=True)
    except Exception as e: print("YOLO failed:", e, flush=True)
print("->", d, flush=True); sys.stdout.flush(); os._exit(0)
