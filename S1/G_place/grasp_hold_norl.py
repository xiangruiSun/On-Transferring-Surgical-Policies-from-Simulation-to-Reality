#!/usr/bin/env python3
"""grasp_hold.py without the RL policy (user 2026-10-30: 'do not rely on SurgicAI first' = no RL Reach).
Same v2 chain; only Attempt.reach_start / reach_ladder are replaced by a logged no-op, so hover() takes its existing
'Reach end not near H10' branch: safe_goto to the template pose (shifted by -c_off) -> R6 hover -> descend -> close -> lift.
This is the route of A29-A33 (user-judged grasps) minus the Reach detour.  Needle kept in the jaw (hold), as grasp_hold.py.

    python3 grasp_hold_norl.py --n 1
"""
import argparse, os, sys
CODE = os.path.expanduser("~/rl_live/sim2real_20261106/S1/code")
sys.path.insert(0, CODE); os.chdir(CODE)
import attempt as A
import session_driver as SD


def hold(self):
    self.save_frames("hold", 10, all_frames=True)
    je = self.io.get("jaw_eff")
    self.phase("hold", note="needle kept in the jaw for placing (no put down / release / base)",
               jaw_rad=float(self.io.get("jaw")[1][0]), jaw_eff=None if je is None else float(je[1][0]), sp=self.mv.setpoint(),
               tip_pad_mm=self.mv.pad_dist_mm(self.mv.tip(self.mv.setpoint())))


def no_reach_start(self, cells):
    self.T_start = self.mv.setpoint().copy(); self.used_cell = None
    self.phase("reach_start", cell=None, deviation="2026-10-30 user: no RL -> Reach skipped, no command")


def no_reach_ladder(self):
    self.handover = None; self.reach_note = "2026-10-30 user: no RL -> Reach skipped"
    self.phase("reach", rungs=[], handover=None, note=self.reach_note)


A.Attempt.putdown_release = hold
A.Attempt.reach_start = no_reach_start
A.Attempt.reach_ladder = no_reach_ladder
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, required=True); a = ap.parse_args()
    try:
        SD.stage_attempt(SD.ROOT, a.n, user_ok=True, where_note="2026-10-30 grasp and HOLD for placing, no RL Reach")
    except SystemExit as e:
        print("STOP:", e, flush=True); sys.stdout.flush(); os._exit(2)
    except BaseException as e:
        import traceback; traceback.print_exc(); print("STOP:", type(e).__name__, e, flush=True); sys.stdout.flush(); os._exit(1)
    sys.stdout.flush(); os._exit(0)
