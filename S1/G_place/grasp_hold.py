#!/usr/bin/env python3
"""Grasp with the frozen v1 chain, then KEEP the needle in the jaw (user 17:2x: 'execute the grasping and do not reset,
do not lose the jaw').  v1 code is imported unchanged; only Attempt.putdown_release (put down -> open -> lift -> base) is
replaced by a hold: frames + jaw state are recorded and no command is sent.  A miss (grasp check) still retreats to the
base as in v1; hardware events stop as in v1.  Not a v1 validation attempt.

    python3 grasp_hold.py --n 41
"""
import argparse, os, sys, time
CODE = os.path.expanduser("~/rl_live/sim2real_20261106/S1/code")
sys.path.insert(0, CODE); os.chdir(CODE)
import numpy as np
import attempt as A
import session_driver as SD


def hold(self):
    self.save_frames("hold", 10, all_frames=True)
    je = self.io.get("jaw_eff")
    self.phase("hold", note="17:2x user: needle kept in the jaw for placing (no put down / release / base)",
               jaw_rad=float(self.io.get("jaw")[1][0]), jaw_eff=None if je is None else float(je[1][0]), sp=self.mv.setpoint(),
               tip_pad_mm=self.mv.pad_dist_mm(self.mv.tip(self.mv.setpoint())))


A.Attempt.putdown_release = hold
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, required=True); a = ap.parse_args()
try:
    SD.stage_attempt(SD.ROOT, a.n, user_ok=True, where_note="place-prep: grasp and HOLD (not V1 validation), same needle as A03")
except SystemExit as e:
    print("STOP:", e, flush=True); sys.stdout.flush(); os._exit(2)
except BaseException as e:
    import traceback; traceback.print_exc(); print("STOP:", type(e).__name__, e, flush=True); sys.stdout.flush(); os._exit(1)
sys.stdout.flush(); os._exit(0)
