#!/usr/bin/env python3
"""Reach -> grasp -> place, one trial, in the 2026-11-09 setting (blue sticker = entry).

Everything a trial needs is behind this one entry point; outputs go to trials/<trial>/. The robot code itself is the frozen
S1 code (~/rl_live/sim2real_20261106/S1), called unchanged, so its guards (pad, joint margins, efforts, ECM/SUJ freeze) all apply.

Stages (run in this order; every moving stage does a dry run first and only sends if the dry run passes):
  grasp   N                       RL Reach + hover + descend + close + lift (S1/trials/grasp_rl.sh, trial T<N>); checks the grasp
  snap    NAME [--fs]             stereo snapshot (+ FoundationStereo disparity on the 3090) and a gridded view for reading the tip
  shift   NAME dx dy dz           Cartesian tool shift in ECM mm (bring the needle into view), then a snapshot
  needle  NAME u,v u,v ...        needle points tail -> tip read on snapshot NAME -> grip fit -> best reachable placing pose
                                  -> prints the rotation (axis, degrees) still needed
  rot     NAME DEG tipR body tipZ rotate about the needle tip by DEG (<= 30) toward the planned pose; tip/body read on the last snapshot
  down    NAME H tipR tipL        move the tip to entry + H mm along the site normal (15 re-centre, 8, then 3.5 final)
  metric  tipR tipL               final tip read on the last snapshot -> height above / lateral offset from the entry
  release                         open the jaw and go back to the base (S1/trials/release.sh)
tipR / tipL / body are 'u,v' pixels read on the right (reference) / left image of the previous snapshot (left uses only u).

Operator in the loop: tip pixels are read by eye from the gridded views; there is no automatic held-needle detector yet.
Setting files (measured once for this phantom position): setting/pad.json (also written to S1/D_pad/pad.json), setting/site/measure.json.
"""
import json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.expanduser("~/rl_live/sim2real_20261106/S1")
PP = S1 + "/P_place"                                   # placing scripts (run with this cwd)
SITE = HERE + "/setting/site/measure.json"
CAL = S1 + "/C_r6cal/calibration_tcp.json"
FS_DIR = os.path.expanduser("~/Desktop/surgicai_grasp_integration/surgic_ai_deployment_unpublished/deploy")
ENV = f"source {S1}/trials/env.sh; export NEEDLE_GREY_VMAX=235; "   # VMAX=235: washed-out pad, used for all 11-09 trials
PLACE_TOL = (HERE + "/tools/") if os.environ.get("PLACE_TOL") == "1" else ""; _TOL = "_tol" if PLACE_TOL else ""   # 2026-11-10 18:4x: PLACE_TOL=1 -> tools/rot_tip_tol.py / phase2_tol.py (wrist settle tolerance 2.5 deg); unset = S1/P_place originals
TRIAL = os.environ.get("TRIAL", "current")              # output folder name: trials/$TRIAL
OUT = f"{HERE}/trials/{TRIAL}"


def sh(cmd, cwd=PP, check=True):
    """run a shell command with the S1 environment, echo it, return stdout (also logged)."""
    print("$", cmd, flush=True)
    p = subprocess.run(["bash", "-c", ENV + cmd], cwd=cwd, capture_output=True, text=True)
    out = "\n".join(l for l in (p.stdout + p.stderr).splitlines() if "Warn" not in l)
    with open(OUT + "/log.txt", "a") as f: f.write(f"\n$ {cmd}\n{out}\n")
    if check and p.returncode not in (0, None): sys.exit(f"command failed ({p.returncode})")
    return out


def last(out, n=3): print("\n".join(out.strip().splitlines()[-n:]), flush=True)


def uv(s): return [float(x) for x in s.split(",")]


def grid_view(snap):
    """right image with a 50-px grid + both images side by side, for reading pixels."""
    import cv2
    R = cv2.imread(snap + "/right_rect.png"); L = cv2.imread(snap + "/left_rect.png")
    for I in (R, L):
        for x in range(0, 1300, 50): cv2.line(I, (x, 0), (x, 1024), (0, 255, 0) if x % 100 == 0 else (0, 90, 0), 1)
        for y in range(0, 1024, 50): cv2.line(I, (0, y), (1300, y), (0, 255, 0) if y % 100 == 0 else (0, 90, 0), 1)
        for x in range(0, 1300, 100): cv2.putText(I, str(x), (x + 2, 18), 0, .6, (0, 0, 255), 2)
        for y in range(100, 1024, 100): cv2.putText(I, str(y), (2, y - 3), 0, .6, (0, 0, 255), 2)
    cv2.imwrite(snap + "_grid_R.jpg", R); cv2.imwrite(snap + "_grid_L.jpg", L); print("views:", snap + "_grid_R.jpg", snap + "_grid_L.jpg")


def snap(name, fs=False):
    d = f"{OUT}/{name}"; sh(f"python3 {S1}/code/r6_local/r6_snap.py {d} >/dev/null 2>&1")
    if fs: last(sh(f"timeout 240 bash tools/remote_fs.sh {d}", cwd=FS_DIR), 1)
    grid_view(d); return d


def move_dry_then_send(script_args, out_name):
    """dry run; send only if it printed the 'checks passed' line; then the script itself saves right/left images in its out dir."""
    dry = sh(f"timeout 150 python3 {script_args} --out {OUT}/{out_name}_dry", check=False); last(dry, 2)
    if "passed" not in dry: sys.exit("dry run did not pass -> nothing sent")
    res = sh(f"timeout 240 python3 {script_args} --out {OUT}/{out_name} --send", check=False); last(res, 2)
    if "STOPPED" in res: sys.exit("guard stop -> see log")
    grid_view(f"{OUT}/{out_name}")


def main():
    os.makedirs(OUT, exist_ok=True); a = sys.argv[1:]
    if not a: sys.exit(__doc__)
    st, a = a[0], a[1:]
    if st == "grasp":                                   # ---- reach + grasp (frozen S1 runner, RL Reach)
        n = int(a[0]); out = sh(f"bash {os.environ.get('GRASP_SH', 'trials/grasp_rl.sh')} {n}", cwd=S1, check=False); last(out, 6)   # 2026-11-10: GRASP_SH = tools/grasp_rl_roll.sh uses setting/roll_model.json
        ok = "grasp_check True" in out
        json.dump(dict(trial=n, grasped=ok, time=time.strftime("%H:%M:%S")), open(OUT + "/grasp.json", "w"))
        print("GRASPED" if ok else "NOT GRASPED -> run 'release', re-place the needle, new trial")
    elif st == "snap": snap(a[0], fs="--fs" in a)
    elif st == "shift":                                 # ---- Cartesian shift to bring the needle into view
        last(sh(f"python3 code/cp_step.py {a[1]} {a[2]} {a[3]}", cwd=S1), 2); snap(a[0], fs=True)
    elif st == "needle":                                # ---- needle shape -> grip -> best reachable placing pose
        d = f"{OUT}/{a[0]}"; pts = [uv(p) for p in a[1:]]
        import cv2
        sys.path.insert(0, PP); from ncc_disp import disp
        R = cv2.imread(d + "/right_rect.png"); L = cv2.imread(d + "/left_rect.png"); D = np.load(d + "/fs_disp.npy"); S = []
        for u, v in pts:                                # NCC disparity on the thin wire, seeded by FS (as on 11-06 p25)
            fs = float(D[int(v), int(u)]); best = None
            for h in (7, 9):
                for d0 in (fs, fs - 15, fs + 15):
                    try:
                        r = disp(R, L, u, v, hw=h, d0=d0, rng=12)
                        if best is None or r[0] > best[0]: best = r
                    except Exception: pass
            S.append((u, v, best[1] if best and best[0] > 0.5 else fs))
        S = np.array(S); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(S[:, :2], axis=0), axis=1))]; S[:, 2] = np.polyval(np.polyfit(s, S[:, 2], 2), s)
        json.dump(S.tolist(), open(OUT + "/uvd.json", "w")); q = json.load(open(d + "/state.json"))["js"][:6]; json.dump(q, open(OUT + "/q.json", "w"))
        print("needle disparities", np.round(S[:, 2], 1).tolist())
        last(sh(f"timeout 60 python3 grip_fit_s2.py '{json.dumps(S.tolist())}' {CAL} '{json.dumps(q)}' {OUT}/grip.json"), 2)
        os.makedirs(OUT + "/padonly", exist_ok=True); sh(f"timeout 60 python3 padonly_disp.py {d} {OUT}/padonly")
        br = sh(f"timeout 180 python3 best_reach.py --measure {SITE} --grip {OUT}/grip.json --fssnap {OUT}/padonly --hover 15 --min-clear 2.5 --out {OUT}/br_dry", check=False)
        print("\n".join(l for l in br.splitlines() if "now wrist" in l or "STOP" in l)[:600])
        sys.path[:0] = [S1 + "/code", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
        import kin as K; from calib import rotation_from_affine; from scipy.spatial.transform import Rotation as Ro
        cal = json.load(open(CAL)); Rc = rotation_from_affine(np.array(cal["M"])); SUJ = np.array(cal["frozen"]["suj_psm1"])
        pl = [json.loads(l) for l in open(OUT + "/br_dry/log.jsonl") if '"plan"' in l][-1]; q1 = np.array(pl["q1"])
        rv = Ro.from_matrix(Rc @ K.fk_ecm(q1, SUJ)[:3, :3] @ K.fk_ecm(np.array(q), SUJ)[:3, :3].T @ Rc.T).as_rotvec()
        ax = rv / np.linalg.norm(rv); plan = dict(q1=q1.tolist(), deg=float(np.degrees(np.linalg.norm(rv))), axis=ax.tolist()); json.dump(plan, open(OUT + "/rot_plan.json", "w"))
        print("ROTATION to the best reachable placing pose: %.1f deg about cam axis %s -> rot steps of <= 30 deg" % (plan["deg"], " ".join("%.4f" % x for x in ax)))
    elif st == "rot":                                   # ---- rotate about the tip (axis from the plan)
        name, deg, tipR, body, tipZ = a[0], float(a[1]), uv(a[2]), uv(a[3]), float(a[4]); frames = a[5] if len(a) > 5 else None
        ax = json.load(open(OUT + "/rot_plan.json"))["axis"]; frames = frames or sorted([d for d in os.listdir(OUT) if os.path.exists(f"{OUT}/{d}/right_rect.png") and not d.endswith("_dry")], key=lambda d: os.path.getmtime(f"{OUT}/{d}/right_rect.png"))[-1]
        move_dry_then_send(f"{PLACE_TOL}rot_tip{_TOL}.py --measure {SITE} --deg {deg} --axis {ax[0]:.4f} {ax[1]:.4f} {ax[2]:.4f} --tipR {tipR[0]:.0f} {tipR[1]:.0f} --body {body[0]:.0f} {body[1]:.0f} --tipZ {tipZ} --frames {OUT}/{frames}", name)
    elif st == "down":                                  # ---- tip to entry + H along the normal
        name, h, tR, tL = a[0], float(a[1]), uv(a[2]), uv(a[3]); extra = " --ins-eff 0.8" if h >= 15 else ""
        move_dry_then_send(f"{PLACE_TOL}phase2{_TOL}.py --tipR {tR[0]:.0f} {tR[1]:.0f} --tipL {tL[0]:.0f} {tL[1]:.0f} --h {h}{extra} --measure {SITE}", name)
    elif st == "metric":                                # ---- final check: tip read on the final snapshot -> height / lateral
        tR, tL = uv(a[0]), uv(a[1]); last(sh(f"python3 tip_metric.py {SITE} {tR[0]} {tR[1]} {tL[0]}"), 3)
    elif st == "release": last(sh("bash trials/release.sh", cwd=S1, check=False), 3)
    else: sys.exit(__doc__)


if __name__ == "__main__":
    main()
