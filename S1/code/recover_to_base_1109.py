#!/usr/bin/env python3
"""Bring PSM1 back to the calibration base with Mover.safe_goto (user OK '继续' 13:4x after the A04 pad-guard stop)."""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig import RigIO, Mover, load_kin, tip_of
R = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); out = R + "/A_check/roll_data_1109/"
cal = json.load(open(R + "/C_r6cal/calibration_tcp.json")); pad = json.load(open(R + "/D_pad/pad.json"))
logf = open(out + "/recover_to_base.jsonl", "a"); log = lambda r: logf.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
try:
    io = RigIO(images=False); kin = load_kin(); time.sleep(2); mv = Mover(io, kin, log)
    mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = np.array(cal["frozen"]["suj_psm1"]); mv.pad = (np.array(pad["normal"]), pad["d"])
    print("tip above pad (incl margin) %.1f mm" % mv.pad_dist_mm(tip_of(mv.setpoint())), flush=True)
    mv.safe_goto(np.array(cal["base_T"]), tag="recover_to_base"); mv.jaw_to(0.0, n_steps=1, dt=1.8, tag="close_at_base")
    print("at base: %.3f mm, commands %d" % (np.linalg.norm(mv.setpoint()[:3, 3] - np.array(cal["base_T"])[:3, 3]) * 1e3, mv.n_cmd), flush=True)
except BaseException as e:
    import traceback; traceback.print_exc(); print("STOPPED:", e, flush=True)
sys.stdout.flush(); os._exit(0)
