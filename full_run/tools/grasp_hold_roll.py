#!/usr/bin/env python3
"""2026-11-10: S1/G_place/grasp_hold.py UNCHANGED, with the roll-model constants of attempt.py (ROLL_A_MM, ROLL_E_XY_MM: jaw midpoint -
kinematic TCP vs roll, used when the jaw-tip template falls back to the model) replaced by the fit for the CURRENT calibration
(full_run_1109/setting/roll_model.json = S1/A_check/roll_model_<date>.json).  Nothing in S1/code is edited.
Why: new setup of 11-10 needs grasp rolls of +78..+85 deg; the 11-08 constants are 108-125 px (5-6 mm) off there (fit range -75..+50)."""
import json, os, sys, runpy
import numpy as np
CODE = os.path.expanduser("~/rl_live/sim2real_20261106/S1/code"); sys.path.insert(0, CODE); os.chdir(CODE)
import attempt as A
m = json.load(open(os.path.expanduser("~/rl_live/full_run_1109/setting/roll_model.json")))
cal = json.load(open(os.path.expanduser("~/rl_live/sim2real_20261106/S1/C_r6cal/calibration_tcp.json")))
assert np.allclose(np.array(m["Rref"]), np.array(cal["base_T"])[:3, :3], atol=1e-6), "roll model was fitted for another calibration (Rref differs) -> refit (A_check/collect_roll.py + fit_roll_*.py)"
print("roll model: a %s e_xy %s (attempt.py default a %s e_xy %s)" % (np.round(m["a"], 2).tolist(), np.round(m["e_xy"], 2).tolist(), A.ROLL_A_MM.tolist(), A.ROLL_E_XY_MM.tolist()), flush=True)
A.ROLL_A_MM = np.array(m["a"], float); A.ROLL_E_XY_MM = np.array(m["e_xy"], float)
# 2026-11-10 (A172: jaw tips exactly on the grasp point, needle 1 mm from the tips -> slipped out at the close): aim deeper into the jaw
back = float(os.environ.get("AIM_BACK_MM", "2.5")); print("AIM_BACK_MM %.1f (attempt.py default %.1f)" % (back, A.AIM_BACK_MM), flush=True); A.AIM_BACK_MM = back
runpy.run_path(os.path.expanduser("~/rl_live/sim2real_20261106/S1/G_place/grasp_hold.py"), run_name="__main__")
