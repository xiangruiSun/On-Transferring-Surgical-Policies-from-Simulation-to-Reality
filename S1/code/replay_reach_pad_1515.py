#!/usr/bin/env python3
"""Offline (no robot) closed-loop Reach on today's goals with reach_real's own FakeRobot (measured = command), to compare the R2-E TCP
pad test (deploy package as is) with the camera-corrected one (attempt.patch_reach_pad, 15:1x, user option 1).

Per attempt with a moving Reach rung: goal.json + pad.json of the attempt, start = first measured pose of the logged s0.5 rung,
tip_off_tool = the attempt's template value.  Check: the uncorrected FakeRobot rollout should stop at about the logged real step.
Out: ../F_attempts/replay_reach_pad_1515.json"""
import glob, json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); DEP = R0 + "/deploy/deploy_export"
sys.path.insert(0, DEP); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(DEP)
import reach_real, run_reach_dvrk
from reach_real import run, goal_hinge, FakeRobot, LIM
from surgicai_rl_deploy.frames import Pose
from r6_actor import actor_mlp
import attempt

layers = actor_mlp.load("r6_actor/actor_weights.npz")
pre_orig = reach_real.precheck


def rollout(g, pad, start, scale, corrected, off):
    reach_real.precheck = pre_orig
    if corrected: attempt.patch_reach_pad(off)
    try:
        gid, p_grasp, R_goal, _ = run_reach_dvrk.load_goal(g)
        r = run(FakeRobot(start, 1.0), layers, goal_hinge(p_grasp, R_goal), p_grasp, pad=pad, scale=scale, dry=False, lim=dict(LIM))
    finally:
        reach_real.precheck = pre_orig; reach_real._ORIG_precheck = pre_orig
    e = [x["e_mm"] for x in r["rows"]]; gdeg = [x["e_rpy_deg"] for x in r["rows"]]
    return dict(status=r["status"], steps=r["steps"], e_start=e[0], e_min=min(e), e_end=e[-1], rpy_start=gdeg[0], rpy_end=gdeg[-1], rpy_min=min(gdeg))


res = []
for d in sorted(glob.glob(R0 + "/F_attempts/A*_*/")):
    a = os.path.basename(os.path.normpath(d)); logs = sorted(glob.glob(d + "reach/reach_*_s0.5_*.json"))
    if not logs or not os.path.exists(d + "goal.json"): continue
    L = json.load(open(logs[0])); r0 = L["rows"][0]
    start = Pose(np.array(r0["meas_ecm_p"]), Rotation.from_quat(r0["meas_ecm_quat"]).as_matrix(), 0.0)
    pad = run_reach_dvrk.load_pad(d + "pad.json")
    tpl = next((json.loads(l) for l in open(d + "trial_log.jsonl") if '"name": "template"' in l and '"ev": "phase"' in l), None)
    off = np.array(tpl.get("tip_off_tool_mm") or [3.19, -0.64, -7.87]) * 1e-3 if tpl else np.array([3.19, -0.64, -7.87]) * 1e-3
    row = dict(attempt=a, real_s05=dict(status=L["status"], steps=L["steps"]), off_tool_mm=(off * 1e3).tolist())
    for sc in (0.5, 1.0):
        row[f"old_s{sc}"] = rollout(d + "goal.json", pad, start, sc, False, off)
        row[f"new_s{sc}"] = rollout(d + "goal.json", pad, start, sc, True, off)
    res.append(row)
json.dump(res, open(R0 + "/F_attempts/replay_reach_pad_1515.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
f = lambda x: f"{x['status'][:22]:22s} {x['steps']:3d} st  e {x['e_start']:4.1f}->{x['e_min']:4.1f}->{x['e_end']:4.1f} mm  rpy {x['rpy_start']:3.0f}->{x['rpy_min']:3.0f}->{x['rpy_end']:3.0f}"
for r in res:
    print(f"{r['attempt']:9s} real s0.5: {r['real_s05']['status']} {r['real_s05']['steps']} st")
    for k in ("old_s0.5", "new_s0.5", "old_s1.0", "new_s1.0"): print(f"    {k:8s} {f(r[k])}")
