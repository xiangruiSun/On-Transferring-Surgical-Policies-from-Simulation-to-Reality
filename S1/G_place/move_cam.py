#!/usr/bin/env python3
"""Translate PSM1 (orientation fixed) by a CAMERA-frame vector (mm), optionally projected onto the pad plane, through Mover
(hinge-only pad check: wrist FK is off today; LIM_NEAR sub-steps; tracking check as goto_demo: wrist-only < 3 deg accepted when the
hinge is >= 25 mm above the pad).  Then a stereo snapshot.  Refuses > 25 mm.

    python3 move_cam.py --v 10 0 0 --pad-parallel --out DIR [--send]
"""
import argparse, json, os, sys, time
import numpy as np
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code"); sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib"))
import cv2
from rig import RigIO, Mover, load_kin, Guard, TCP_MM

ap = argparse.ArgumentParser(); ap.add_argument("--v", type=float, nargs=3, required=True); ap.add_argument("--pad-parallel", action="store_true")
ap.add_argument("--out", required=True); ap.add_argument("--send", action="store_true"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); SUJ = np.array(cal["frozen"]["suj_psm1"]); pad = json.load(open(R0 + "/D_pad/pad.json"))
nE = np.array(pad["normal"]); Minv = np.linalg.inv(M[:3])
d_e = np.array(a.v) @ Minv * 1e-3
if a.pad_parallel: d_e = d_e - (d_e @ nE) * nE
if np.linalg.norm(d_e) * 1e3 > 25.0: raise SystemExit("move > 25 mm refused")
log_f = open(a.out + "/log.jsonl", "a")
def log(r): log_f.write(json.dumps(dict(t=time.time(), **r)) + "\n"); log_f.flush()
io = RigIO(images=True); kin = load_kin(); t0 = time.time()
while (io.get("js") is None or io.get("sp") is None) and time.time() - t0 < 10: time.sleep(0.1)
time.sleep(1.0); mv = Mover(io, kin, log)
mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = SUJ; mv.pad = (nE, pad["d"]); mv.cube = None; mv.tip_off_tool = -TCP_MM * 1e-3
_step = mv.step
def step_free(T, lim, contact_ok=False, tag=""):
    try: return _step(T, lim, contact_ok=contact_ok, tag=tag)
    except Guard as e:
        js = mv._need("js"); spj = mv._need("spj"); je = js[:6] - spj[:6]; h = mv.pad_dist_mm(mv.setpoint()[:3, 3])
        if "joint tracking" in str(e) and h >= 25.0 and abs(je[2]) * 1e3 <= 0.5 and np.degrees(np.abs(je[:2])).max() <= 1.0 and np.degrees(np.abs(je[3:6])).max() < 3.0:
            log(dict(ev="track_accepted_free_space", why=str(e), hinge_mm=h)); return mv.measured()
        raise
mv.step = step_free
mv.state_ok(); S = mv.setpoint()
print(f"ECM delta {np.round(d_e * 1e3, 2).tolist()} mm ({np.linalg.norm(d_e) * 1e3:.1f}); hinge pad clearance {mv.pad_dist_mm(S[:3, 3]):.1f} -> {mv.pad_dist_mm(S[:3, 3] + d_e):.1f} mm", flush=True)
if a.send:
    try: mv.translate(d_e, tag="move_cam"); print("moved; jaw eff %.3f" % io.get("jaw_eff")[1][0], flush=True)
    except Guard as e: print("GUARD:", e, flush=True); log(dict(ev="guard", why=str(e)))
b = io.frames(8); cv2.imwrite(a.out + "/right_rect.png", np.mean([z[1] for z in b], 0).round().astype(np.uint8)); cv2.imwrite(a.out + "/left_rect.png", np.mean([z[0] for z in b], 0).round().astype(np.uint8))
json.dump(dict(js=np.asarray(io.get("js")[1][:6]).tolist(), sp=mv.setpoint().tolist()), open(a.out + "/state.json", "w"))
print("->", a.out, flush=True); sys.stdout.flush(); os._exit(0)
