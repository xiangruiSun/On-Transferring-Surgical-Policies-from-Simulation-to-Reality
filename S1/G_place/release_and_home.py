#!/usr/bin/env python3
"""User 17:5x: open the jaw where the arm is (the needle drops), then back to the calibration base with Mover.safe_goto
(lift >= 15 mm above the pad, rotate about the tip, translate; all of today's guards), close the jaw at the base."""
import json, os, sys, time
import numpy as np
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code")
from rig import RigIO, Mover, load_kin
out = R0 + "/G_place/release_home_" + time.strftime("%H%M%S"); os.makedirs(out)
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); pad = json.load(open(R0 + "/D_pad/pad.json"))
logf = open(out + "/log.jsonl", "a"); log = lambda r: logf.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
try:
    io = RigIO(images=False); kin = load_kin(); time.sleep(2); mv = Mover(io, kin, log)
    mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = np.array(cal["frozen"]["suj_psm1"]); mv.pad = (np.array(pad["normal"]), pad["d"])
    j = mv.jaw_to(0.8, n_steps=1, dt=1.8, tag="release_open"); print("jaw opened", round(j, 3), flush=True); time.sleep(1.0)
    print("tip above pad (incl margin) %.1f mm" % mv.pad_dist_mm(mv.tip(mv.setpoint())), flush=True)
    mv.safe_goto(np.array(cal["base_T"]), tag="release_to_base"); mv.jaw_to(0.0, n_steps=1, dt=1.8, tag="close_at_base")
    print("at base: %.3f mm, commands %d" % (np.linalg.norm(mv.setpoint()[:3, 3] - np.array(cal["base_T"])[:3, 3]) * 1e3, mv.n_cmd), flush=True)
except BaseException as e:
    import traceback; traceback.print_exc(); print("STOPPED:", e, flush=True)
sys.stdout.flush(); os._exit(0)
