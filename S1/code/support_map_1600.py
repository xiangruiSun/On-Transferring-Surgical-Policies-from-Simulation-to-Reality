#!/usr/bin/env python3
"""Offline (no robot): where on today's rig can the Reach policy start?  (user 15:5x: 'compare the training range with the real range')

1. starts sampled in the policy (training) frame from the contract statistics of the 50 demo episodes: start offset in the goal frame
   uniform in R6_START_OFFSET_GOAL_MIN..MAX, start->goal rotation = rotvec along R6_START_TO_GOAL_ROTVEC_MEAN (axis jitter <= 20 deg),
   angle uniform 26-100 deg; kept only if inside the tool-frame support box (APPROACH_R6.in_support), as the training starts were.
   The rotation convention is chosen on R6_REPRESENTATIVE_START/GOAL_VEC7.
2. today's goal (attempt folder: goal.json, pad.json, template tip_off_tool) -> FrameBridge.rebase, as reach_real does.
3. real-rig filters at the start: IK / joint margins (kin.check_pose, as rig.check_target), reach_real's +-40 mm cube on the hinge,
   pad clearance of hinge and camera-corrected tip >= 3 mm.
4. closed loop with reach_real.FakeRobot (measured = command): (a) no pad -> does FakeRobot reproduce the sim success rate on
   in-support starts? (b) with the pad (camera-corrected test, attempt.patch_reach_pad) + IK of every 5th command.
Out: ../F_attempts/support_map_1600.json"""
import argparse, json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); DEP = R0 + "/deploy/deploy_export"
sys.path.insert(0, DEP); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig, attempt
os.chdir(DEP)
import reach_real, run_reach_dvrk
from reach_real import run, goal_hinge, FakeRobot, LIM
from surgicai_rl_deploy.frames import FrameBridge, Pose
from surgicai_rl_deploy import contract as K
from r6_actor import actor_mlp

ap = argparse.ArgumentParser(); ap.add_argument("--attempt", default="A18_Prep"); ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--align", action="store_true"); ap.add_argument("--cube_mm", type=float, default=40.0); ap.add_argument("--extra_standoff_mm", type=float, default=0.0)
a = ap.parse_args()
_bo = reach_real.build_observation
if a.align: reach_real.build_observation = lambda m, g, align_rpy_branch=False: _bo(m, g, align_rpy_branch=True)   # 16:0x analysis: align_rpy_branch ON
LIMX = dict(LIM); LIMX["ws_half_m"] = a.cube_mm * 1e-3
rng = np.random.default_rng(a.seed); layers = actor_mlp.load("r6_actor/actor_weights.npz")
xyz = lambda v: Rotation.from_euler("xyz", v[3:6]).as_matrix()

# --- rotation convention on the representative demo pair
Rs, Rg = xyz(K.R6_REPRESENTATIVE_START_VEC7), xyz(K.R6_REPRESENTATIVE_GOAL_VEC7); m = K.R6_START_TO_GOAL_ROTVEC_MEAN / np.linalg.norm(K.R6_START_TO_GOAL_ROTVEC_MEAN)
conv = {"world (Rg = Exp(r) Rs)": Rotation.from_matrix(Rg @ Rs.T).as_rotvec(), "body (Rg = Rs Exp(r))": Rotation.from_matrix(Rs.T @ Rg).as_rotvec()}
cos = {k: float(v @ m / np.linalg.norm(v)) for k, v in conv.items()}; WORLD = cos["world (Rg = Exp(r) Rs)"] >= cos["body (Rg = Rs Exp(r))"]
print("rotation convention (cos to the mean rotvec):", {k: round(c, 3) for k, c in cos.items()}, "->", "world" if WORLD else "body")

# --- sample training-like starts in the policy frame
G = Pose.from_vec7(K.R6_TRAINED_GOAL_VEC7).with_jaw(0.0)
def sample():
    while True:
        o = rng.uniform(K.R6_START_OFFSET_GOAL_MIN, K.R6_START_OFFSET_GOAL_MAX) * 1e-2
        ax = Rotation.from_rotvec(np.radians(rng.uniform(0, 20)) * np.cross(m, rng.normal(size=3)) / 1.0).apply(m); ax /= np.linalg.norm(ax)
        ang = np.radians(rng.uniform(26, 100)); E = Rotation.from_rotvec(ax * ang).as_matrix()
        R_s = E.T @ G.R if WORLD else G.R @ E.T
        p_s = G.p + G.R @ o
        off = R_s.T @ (G.p - p_s) * 100; rot = float(np.degrees(Rotation.from_matrix(R_s.T @ G.R).magnitude()))
        if not K.APPROACH_R6.in_support(off, rot): return Pose(p_s, R_s, 0.0), off, rot

# --- today's goal / rig
d = f"{R0}/F_attempts/{a.attempt}/"; cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json"))
gid, p_grasp, R_goal, _ = run_reach_dvrk.load_goal(d + "goal.json"); pad = run_reach_dvrk.load_pad(d + "pad.json"); n_, d_ = pad
tpl = next(json.loads(l) for l in open(d + "trial_log.jsonl") if '"name": "template"' in l and '"ev": "phase"' in l)
off_tool = np.array(tpl["tip_off_tool_mm"]) * 1e-3; tcp_c = reach_real.TCP_TOOL_M + off_tool
hinge = goal_hinge(p_grasp, R_goal); hinge = Pose(hinge.p - hinge.R[:, 2] * a.extra_standoff_mm * 1e-3, hinge.R, 0.0); X = FrameBridge.rebase(hinge, G)   # 16:0x: goal backed off further (policy frame unchanged)
kin = rig.load_kin(); js0 = np.array(json.load(open(d + "open25/state.json"))["js"])[:6]; suj = np.array(cal["frozen"]["suj_psm1"])
padd = lambda p: float((n_ @ p + d_) * 1e3)

def ik_ok(T): ok, q, why = kin.check_pose(T.p, T.R, js0, suj, min_deg=3.0, min_mm=5.0); return ok, why

def rollout(start, with_pad, sc=1.0):
    reach_real.precheck = getattr(reach_real, "_ORIG_precheck", reach_real.precheck)
    if with_pad: attempt.patch_reach_pad(off_tool)
    r = run(FakeRobot(start, 1.0), layers, hinge, p_grasp, pad=pad if with_pad else None, scale=sc, dry=False, lim=dict(LIMX))
    reach_real.precheck = getattr(reach_real, "_ORIG_precheck", reach_real.precheck)
    rows = r["rows"]; ik_bad = None
    if with_pad:
        for k in range(0, len(rows), 5):
            if rows[k].get("cmd_ecm_p") is None: continue
            T = Pose(np.array(rows[k]["cmd_ecm_p"]), Rotation.from_quat(rows[k]["cmd_ecm_quat"]).as_matrix(), 0.0); ok, why = ik_ok(T)
            if not ok: ik_bad = f"step {k}: {why}"; break
    hp = [padd(np.array(x["cmd_ecm_p"])) for x in rows if x.get("cmd_ecm_p") is not None]
    return dict(status=r["status"], steps=r["steps"], e_end=rows[-1]["e_mm"], rpy_end=rows[-1]["e_rpy_deg"], first_ok=r["first_contract_1cm_10deg_step"], min_hinge_pad_mm=min(hp) if hp else None, ik_bad=ik_bad)

out = []
for i in range(a.n):
    S_pol, off, rot = sample(); S = X.to_robot(S_pol)
    row = dict(i=i, off_tool_cm=off.tolist(), rot_deg=rot, start_hinge_ecm=S.p.tolist(), start_quat=Rotation.from_matrix(S.R).as_quat().tolist())
    ok, why = ik_ok(S); row["ik"] = ok or why
    row["cube_mm"] = float(np.abs(S.p - p_grasp).max() * 1e3)
    row["pad_start_mm"] = dict(hinge=padd(S.p), tip=padd(S.p + S.R @ tcp_c))
    row["start_ok"] = bool(ok and row["cube_mm"] <= a.cube_mm and min(row["pad_start_mm"].values()) >= 3)
    row["nopad"] = rollout(S, False)
    if row["start_ok"]: row["pad"] = rollout(S, True)
    out.append(row)
json.dump(dict(attempt=a.attempt, rotation_convention="world" if WORLD else "body", rows=out), open(R0 + f"/F_attempts/support_map_1600_align{int(a.align)}_cube{int(a.cube_mm)}_so{int(a.extra_standoff_mm)}.json", "w"), indent=1,
          default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
N = len(out); succ = lambda k, rs: sum(1 for r in rs if r.get(k, {}).get("status") == "success")
print(f"sampled {N} in-support starts (goal of {a.attempt}), align_rpy_branch {a.align}, cube +-{a.cube_mm:.0f} mm, extra standoff {a.extra_standoff_mm:.0f} mm, goal corrected-tip pad clearance {padd(hinge.p + hinge.R @ tcp_c):.1f} mm")
print(f"  (a) FakeRobot, no pad: success {succ('nopad', out)}/{N}   [sim in-support: 764/793]")
st = [r for r in out if r["start_ok"]]
print(f"  start filters: IK ok {sum(1 for r in out if r['ik'] is True)}, cube<={a.cube_mm:.0f} {sum(1 for r in out if r['cube_mm'] <= a.cube_mm)}, pad>=3 {sum(1 for r in out if min(r['pad_start_mm'].values()) >= 3)}, all {len(st)}")
good = [r for r in st if r["pad"]["status"] == "success" and r["pad"]["ik_bad"] is None]
print(f"  (b) of the {len(st)} feasible starts, with pad + IK along the path: success {len(good)}   status counts:",
      {s: sum(1 for r in st if r['pad']['status'] == s) for s in set(r['pad']['status'] for r in st)}, " ik_bad", sum(1 for r in st if r['pad']['ik_bad']))
for r in sorted(good, key=lambda r: r["pad"]["steps"])[:8]:
    print(f"    #{r['i']:3d} off_tool {np.round(r['off_tool_cm'], 2).tolist()} rot {r['rot_deg']:.0f}  steps {r['pad']['steps']} e_end {r['pad']['e_end']:.1f} mm  min hinge-pad {r['pad']['min_hinge_pad_mm']:.1f}")
