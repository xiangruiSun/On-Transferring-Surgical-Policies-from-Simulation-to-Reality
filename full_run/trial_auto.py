#!/usr/bin/env python3
"""One full trial, reach -> grasp -> place, FULLY AUTOMATIC (2026-11-10): no clicks, no questions.

    cd ~/rl_live/full_run_1109 && python3 trial_auto.py            # needle lying flat on the pad, in view, away from the jaw
    python3 trial_auto.py --hold                                   # keep holding the needle at the end (no release / back to base)

Same robot steps as trial.py / trials/T01 (run.py stages, frozen S1 code, every moving stage dry-run first); the operator's clicks
are replaced by detectors:
  * lying needle + which end is the tail: S1 needle_detect inside the grasp chain (as in trials T45-T62), no tail hint;
  * needle points after the grasp (snapshot s0): needle_detect on the lifted needle, ordered jaw -> tip with the kinematic jaw position;
  * tip (right + left) and the body point before every placing move: tools/held_needle.py (FoundationStereo disparity vs the
    pre-grasp snapshot -> thin structure attached to the instrument -> far end; tip disparity from NCC along the needle).
    Offline on trials/T01 (5 stages): right tip within 0-5 px of the operator's clicks.
Automatic stops (after a grasp attempt: release + back to the base, unless --hold): rig not ready, grasp failed, a detector not confident,
tip jump after a rotation > 600 px, tip height / lateral offset after a 'down' move not where it was commanded, any guard of run.py.
Verdict (machine): SUCCESS if the final tip is within 2 mm lateral of the sticker centre and 0.5 .. 6.5 mm above it (commanded 3.5).
"""
import glob, json, os, re, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); S1 = os.path.expanduser("~/rl_live/sim2real_20261106/S1")
sys.path[:0] = [HERE + "/tools", S1 + "/code", S1 + "/P_place", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import held_needle as HN
FX, CX, CY = 1860.724932, 690.538345, 576.803638; FB = FX * 5.4476353950418295; DV = 2.2
ROT_STEP = 35.0; HEIGHTS = (15.0, 8.0, 3.5); H_TOL = {15.0: 4.5, 8.0: 3.0}; LAT_TOL = 4.0
ENVSH = f"source {S1}/trials/env.sh; export NEEDLE_GREY_VMAX=235"
FS_DIR = os.path.expanduser("~/Desktop/surgicai_grasp_integration/surgic_ai_deployment_unpublished/deploy")
SITE = HERE + "/setting/site/measure.json"; CAL = S1 + "/C_r6cal/calibration_tcp.json"
HOLD = "--hold" in sys.argv
RESUME = sys.argv[sys.argv.index("--resume") + 1:][:2] if "--resume" in sys.argv else None   # --resume <run dir name>: needle already held, new snapshot (interrupted run), continue from the placing plan
ROT_CLEAR = 3.0   # mm above the pad plane of S1/D_pad (= 5 mm above the fitted surface) for the jaw during a rotation
MAX_ROT = 180.0   # deg; 2026-11-10 user "发送": was 110 (A175 119, A176 176 deg in the 11-10 setup)


def say(*a): print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def bash(cmd, cwd=None, timeout=900):
    p = subprocess.run(["bash", "-c", f"{ENVSH}; {cmd}"], cwd=cwd, capture_output=True, text=True, timeout=timeout); return p.returncode, p.stdout + p.stderr


def run(stage_args, trial, must_pass=True):
    """one run.py stage; prints the tail of its output; returns it."""
    env = dict(os.environ, TRIAL=trial); env.pop("NEEDLE_TAIL_HINT", None)
    if os.path.exists(HERE + "/setting/roll_model.json"): env["GRASP_SH"] = HERE + "/tools/grasp_rl_roll.sh"     # roll model fitted for the current calibration
    p = subprocess.run(["bash", "-c", f"{ENVSH}; python3 {HERE}/run.py {stage_args}"], env=env, capture_output=True, text=True)
    out = "\n".join(l for l in (p.stdout + p.stderr).splitlines() if not l.startswith("$") and "autocast" not in l and "Warn" not in l)
    print(out.strip()[-1200:], flush=True)
    if must_pass and p.returncode != 0: stop(f"stage '{stage_args.split()[0]}' stopped (exit {p.returncode})")
    return out


class Stop(Exception): pass


def stop(why): raise Stop(why)


def fs(d):
    rc, out = bash(f"timeout 240 bash tools/remote_fs.sh {d}", cwd=FS_DIR)
    if not os.path.exists(d + "/fs_disp.npy"): stop(f"FoundationStereo failed on {d}: {out[-200:]}")


def fmt(p): return f"{p[0]:.0f},{p[1]:.0f}"


def site():
    m = json.load(open(SITE)); en = min(m["stickers"], key=lambda s: s["area"]); e = np.array(en["cam"]); n = np.array(en["n25"]); return e, (n if n[2] < 0 else -n)


def height_lateral(tip_cam):
    e, n = site(); d = np.asarray(tip_cam) - e; return float(d @ n), float(np.linalg.norm(d - (d @ n) * n))


def tip_of(out_dir, name, bg, prior=None):
    """held-needle detection on snapshot `name` (runs FS first); stops the trial if it is not confident."""
    d = f"{out_dir}/{name}"
    if not os.path.exists(d + "/fs_disp.npy") or os.path.getmtime(d + "/fs_disp.npy") < os.path.getmtime(d + "/right_rect.png"): fs(d)   # stale disparity of an earlier run in the same folder (resume, r1 12:17)
    r = HN.detect(d, bg, prior, d + "_needle.png"); json.dump(r, open(d + "_needle.json", "w"), indent=1)
    if not r.get("ok"): stop(f"held-needle detector not confident on {name}: {r.get('why')} (see {d}_needle.png)")
    h, lat = height_lateral(r["tip_cam"])
    say(f"{name}: tip right ({r['tipR'][0]:.0f},{r['tipR'][1]:.0f}) disparity {r['disp']:.1f} ({r['disp_src']}; FS {r.get('disp_fs_fit', float('nan')):.1f}) -> depth {FB / r['disp']:.1f} mm; "
        f"above the entry {h:.1f} mm, lateral {lat:.1f} mm; needle {r['len_px']:.0f} px long")
    return r


def rot_clearance(out_dir, snap, tipR, Z, deg):
    """lowest predicted height (mm above the pad plane S1/D_pad, which already includes its 2 mm margin) of the jaw (TCP, 1.15 x TCP) while the
    tool turns `deg` about the needle tip (same kinematics as rot_tip.py); and the free-space direction in ECM.  2026-11-10: rot_tip.py has no pad guard."""
    import kin as K; from calib import rotation_from_affine, cam2ecm_mm; from scipy.spatial.transform import Rotation as Ro
    cal = json.load(open(CAL)); M = np.array(cal["M"]); Rc = rotation_from_affine(M); SUJ = np.array(cal["frozen"]["suj_psm1"]); pad = json.load(open(S1 + "/D_pad/pad.json")); n = np.array(pad["normal"])
    st = json.load(open(f"{out_dir}/{snap}/state.json")); q = np.array((st.get("js") or st.get("q"))[:6]); T = K.fk_ecm(q, SUJ); tcp = np.array(cal["TCP_tool_mm"]) * 1e-3
    tip = np.array([(tipR[0] - CX) * Z / FX, (tipR[1] - CY) * Z / FX, Z]); tip_e = cam2ecm_mm(tip, M) * 1e-3; ax = np.array(json.load(open(out_dir + "/rot_plan.json"))["axis"]); low = 1e9
    for f in np.linspace(0, 1, 6):
        Re = Rc.T @ Ro.from_rotvec(ax * np.radians(deg * f)).as_matrix() @ Rc
        for k in (1.0, 1.15): low = min(low, float((n @ (tip_e + Re @ (T[:3, 3] + T[:3, :3] @ (tcp * k) - tip_e)) + pad["d"]) * 1e3))
    return low, n


def needle_points(out_dir, name, bg):
    """needle points jaw -> tip on snapshot `name` (lifted needle). First choice tools/held_needle.py (attached to the instrument, so it
    cannot run onto the thread: A175); fallback S1 needle_detect ordered with the kinematic jaw position (held_needle finds nothing
    when the needle is still over the place the instrument's base pose covered, as on T01)."""
    import kin as K
    d = f"{out_dir}/{name}"; P = None; r = HN.detect(d, bg, None, d + "_needle.png"); json.dump(r, open(d + "_needle.json", "w"), indent=1)
    if r.get("ok"): P = np.array(r["polyR"], float); src = "held_needle"
    else:
        say(f"held_needle on {name}: {r.get('why')} -> needle_detect")
        cal = json.load(open(CAL)); M = np.array(cal["M"]); SUJ = np.array(cal["frozen"]["suj_psm1"]); q = np.array(json.load(open(d + "/state.json"))["js"][:6])
        T = K.fk_ecm(q, SUJ); X = np.r_[T[:3, 3] * 1e3 + T[:3, :3] @ np.array(cal["TCP_tool_mm"]), 1] @ M; jaw = np.array([FX * X[0] / X[2] + CX, FX * X[1] / X[2] + CY])
        rc, out = bash(f"python3 {S1}/code/r6_local/needle_detect.py --snap {d} --out {d}/needle_detect.json", timeout=150)
        j = json.load(open(d + "/needle_detect.json")) if os.path.exists(d + "/needle_detect.json") else {}
        if not (j.get("ok") and j.get("polyline_px")): stop(f"needle not found on {name} (held_needle: {r.get('why')}; needle_detect: {(out.strip().splitlines() or ['no output'])[-1][:120]})")
        P = np.array(j["polyline_px"], float); src = "needle_detect"
        if np.linalg.norm(P[-1] - jaw) < np.linalg.norm(P[0] - jaw): P = P[::-1]; src += " (reversed)"
        L = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
        if np.linalg.norm(P[0] - jaw) > 220 or L > 560: stop(f"needle_detect line on {name} is not the held needle (starts {np.linalg.norm(P[0] - jaw):.0f} px from the jaw, {L:.0f} px long)")
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]; t = np.linspace(0, s[-1], 12); P = np.c_[np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])]
    say(f"{name} needle points ({src}): jaw side ({P[0, 0]:.0f},{P[0, 1]:.0f}) -> tip ({P[-1, 0]:.0f},{P[-1, 1]:.0f}), {s[-1]:.0f} px")
    return P, s[-1]


EDGE = 70   # px: a needle tip closer than this to the image border is brought toward the centre (A177: tip at v = 1016 -> rot_tip's NCC window left the image)


def view_shift(u, v):
    """tool shift (ECM x, y mm; both opposite to image x, y) that brings a point near the image border toward the centre; (0, 0) if it is inside."""
    return (-8.0 if u < EDGE else 8.0 if u > 1300 - EDGE else 0.0), (-8.0 if v < EDGE else 8.0 if v > 1024 - EDGE else 0.0)


def safe_px(p):
    """pixel kept inside the area where S1/P_place/ncc_disp.py can cut its windows (rot_tip.py matches the 'body' point even when --tipZ is given)."""
    return [float(np.clip(p[0], 170, 1275)), float(np.clip(p[1], 25, 995))]


def needle_in_view(out_dir, trial, bg):
    """snapshot s0 (+FS) and the needle points; if the far end of the needle is at the image border, shift the tool (<= 8 mm, twice at
    most) toward the image centre and look again.  ECM x / y are opposite to image x / y.  Returns (snapshot name, points)."""
    name = "s0"; run("snap s0 --fs", trial)
    for k in range(3):
        P, L = needle_points(out_dir, name, bg); u, v = P[-1]; dx = dy = 0.0
        dx, dy = view_shift(u, v)
        if not (dx or dy):
            if L < 150: stop(f"visible needle only {L:.0f} px long on {name}")
            return name, P
        if k == 2: stop("needle tip still at the image border after two shifts")
        say(f"needle runs out of the image at ({u:.0f},{v:.0f}) -> shift the tool {dx:+.0f} / {dy:+.0f} mm (ECM x / y)"); name = f"s{k + 1}"
        run(f"shift {name} {dx} {dy} 0", trial)


def main():
    os.chdir(HERE)
    n_att = max([int(m[0]) for p in glob.glob(S1 + "/F_attempts/A1??_Prep") for m in [re.findall(r"A1(\d\d)_Prep", p)] if m] + [68]) + 1
    name = RESUME[0] if RESUME else time.strftime("auto_%m%d_%H%M%S"); trial = f"../dev_unrecorded/{name}"; out_dir = f"{HERE}/dev_unrecorded/{name}"; os.makedirs(out_dir, exist_ok=True)
    say(f"== automatic trial {name} (grasp attempt A{100 + n_att}); data -> {out_dir}")
    grasped = False
    try:
        if RESUME: grasped = True; bg = out_dir + "/pre"; s0 = time.strftime("sr%H%M%S"); run(f"snap {s0} --fs", trial); pts, _ = needle_points(out_dir, s0, bg); say(f"resumed with the needle held, snapshot {s0}")
        # 0. rig
        rc, rig = (0, "OK - ready") if RESUME else (None, None)
        if not RESUME: rc, rig = bash(f"cd {S1}/G_place; timeout 40 python3 check_rig.py 2>&1"); print("\n".join(rig.strip().splitlines()[-3:]), flush=True)
        if "OK - ready" not in rig:
            if "state checks failed" in rig: stop("ECM / setup joints moved since the calibration (or PSM1 not enabled) -> recalibrate first")
            if "not at the base" in rig:
                say("arm not at the base -> recover_to_base (jaw assumed empty)"); print(bash(f"timeout 150 python3 {S1}/code/recover_to_base_1109.py 2>&1 | tail -2")[1])
                rc, rig = bash(f"cd {S1}/G_place; timeout 40 python3 check_rig.py 2>&1")
            if "OK - ready" not in rig: stop("rig not ready")
        # 1. background snapshot (tool at the base, needle on the pad) + reach + grasp
        if not RESUME:
            bg = out_dir + "/pre"; bash(f"python3 {S1}/code/r6_local/r6_snap.py {bg} >/dev/null 2>&1"); fs(bg); say("background snapshot + FS done; reach + grasp ...")
            out = run(f"grasp {n_att}", trial, must_pass=False)
            if "GRASPED" not in out or "NOT GRASPED" in out: grasped = None; stop("not grasped")
            grasped = True
            # 2. needle shape -> grip -> plan
            s0, pts = needle_in_view(out_dir, trial, bg)
        out = run(f"needle {s0} " + " ".join(fmt(p) for p in pts), trial)
        if not os.path.exists(out_dir + "/rot_plan.json"): stop("no placing plan (best_reach)")
        total = json.load(open(out_dir + "/rot_plan.json"))["deg"]; uvd = np.array(json.load(open(out_dir + "/uvd.json"))); say(f"rotation needed {total:.1f} deg")
        if total > MAX_ROT: stop(f"planned rotation {total:.0f} deg > {MAX_ROT:.0f}: grip too far from the placing pose")
        # 3. rotations about the tip
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]; ib = int(np.argmin(np.abs(s - (s[-1] - 45))))
        tipR, body, Z = pts[-1], pts[ib], FB / uvd[-1, 2]; last, done, k, det = s0, 0.0, 1, None
        cap = ROT_STEP
        while total - done > 0.5:
            step = min(cap, total - done)
            for vk in range(3):                                      # tip near the image border (e.g. after a lift toward the camera) -> shift it toward the centre
                dx, dy = view_shift(*tipR)
                if not (dx or dy): break
                if vk == 2: stop("needle tip still at the image border after two shifts")
                say(f"needle tip at ({tipR[0]:.0f},{tipR[1]:.0f}), near the image border -> shift the tool {dx:+.0f} / {dy:+.0f} mm (ECM x / y)")
                last = f"v{k}{'ab'[vk]}"; run(f"shift {last} {dx} {dy} 0", trial); dl = tip_of(out_dir, last, bg, prior=tipR); tipR, body, Z = dl["tipR"], dl["bodyR"], FB / dl["disp"]
            for lift_k in range(4):                                  # keep the jaw >= ROT_CLEAR mm above the pad during the turn: lift along the pad normal first
                low, nrm = rot_clearance(out_dir, last, tipR, Z, step)
                if low >= ROT_CLEAR: break
                if lift_k == 3: stop(f"jaw would come to {low:.1f} mm above the pad in the next rotation, even after 3 lifts")
                up = min(8.0, ROT_CLEAR - low + 1.0) * nrm; say(f"jaw would come to {low:.1f} mm above the pad in the next {step:.0f} deg -> lift {np.linalg.norm(up):.1f} mm first")
                last = f"l{k}{'abc'[lift_k]}"; run(f"shift {last} {up[0]:.2f} {up[1]:.2f} {up[2]:.2f}", trial); dl = tip_of(out_dir, last, bg, prior=tipR); tipR, body, Z = dl["tipR"], dl["bodyR"], FB / dl["disp"]
            out = run(f"rot r{k} {step:.1f} {fmt(tipR)} {fmt(safe_px(body))} {Z:.1f} {last}", trial, must_pass=False)
            if "IK / margins" in out:                                # A178: the 3rd 25 deg step needed wrist pitch 78.7 deg (margin 1.3 < 3) -> smaller steps, then go on with what was reached
                if step > 6.0: cap = step / 2; say(f"rotation step of {step:.1f} deg leaves the joint margins -> try {cap:.1f} deg"); continue
                say(f"wrist limit reached {total - done:.1f} deg before the planned orientation -> continue with the orientation reached"); break
            if "done: wrist" not in out: stop("stage 'rot' stopped")
            last = f"r{k}"; det = tip_of(out_dir, last, bg, prior=tipR)
            jump = float(np.linalg.norm(np.array(det["tipR"]) - np.array(tipR)))
            if jump > 26 * step + 100: stop(f"tip moved {jump:.0f} px during a {step:.0f} deg rotation about the tip (> {26 * step + 100:.0f})")   # A173: 302 px in one 25 deg step with a correct detection (T01: ~150)
            tipR, body, Z = det["tipR"], det["bodyR"], FB / det["disp"]; done += step; k += 1
        if det is None: det = tip_of(out_dir, s0, bg, prior=tipR)
        # 4. down to entry + 15 / 8 / 3.5 mm
        for h in HEIGHTS:
            name_h = {15.0: "rc1", 8.0: "c8", 3.5: "c35"}[h]
            for att in range(3):                                     # A179: 'joints did not settle at step 2' on the last move (a single move_jp message can get lost) -> look again and retry
                out = run(f"down {name_h} {h} {fmt(det['tipR'])} {fmt(det['tipL'])}", trial, must_pass=False)
                if "did not settle" in out and att < 2:
                    say(f"move to {h} mm did not settle -> new snapshot, retry {att + 1}"); nm = f"{name_h}_retry{att + 1}"; run(f"snap {nm} --fs", trial); det = tip_of(out_dir, nm, bg, prior=det["tipR"]); continue
                if "STOPPED" in out or "did not pass" in out or "guard stop" in out or "command failed" in out: stop(f"stage 'down' to {h} mm stopped")
                break
            det = tip_of(out_dir, name_h, bg, prior=det["tipR"]); hh, lat = height_lateral(det["tip_cam"])
            if h in H_TOL and (abs(hh - h) > H_TOL[h] or lat > LAT_TOL): stop(f"after the move to {h} mm the tip is {hh:.1f} mm above / {lat:.1f} mm beside the entry (tolerance +-{H_TOL[h]} / {LAT_TOL})")
        # 5. final metric + machine verdict
        hh, lat = height_lateral(det["tip_cam"]); run(f"metric {fmt(det['tipR'])} {fmt(det['tipL'])}", trial)
        ok = lat <= 2.0 and 0.5 <= hh <= 6.5
        say(f"FINAL: tip {hh:.1f} mm above the entry, {lat:.1f} mm lateral -> {'SUCCESS' if ok else 'NOT a success'} (machine verdict)")
        json.dump(dict(height_mm=hh, lateral_mm=lat, success=bool(ok), tip=det), open(out_dir + "/result.json", "w"), indent=1)
        if ok:
            nums = [int(x[1:]) for x in os.listdir(HERE + "/trials") if re.fullmatch(r"T\d+", x)]; tn = f"T{max(nums + [0]) + 1:02d}"
            shutil.move(out_dir, f"{HERE}/trials/{tn}"); open(f"{HERE}/trials/{tn}/VERDICT.txt", "w").write(f"{name}: SUCCESS (machine: height {hh:.1f} mm, lateral {lat:.1f} mm)\n"); say("recorded as", tn); trial = tn
    except Stop as e:
        say("STOPPED:", e); open(out_dir + "/STOPPED.txt", "w").write(str(e) + "\n") if os.path.isdir(out_dir) else None
        if grasped is not False and not HOLD: say("release + back to the base"); run("release", trial, must_pass=False)
        return 1
    if not HOLD: say("release + back to the base"); run("release", trial, must_pass=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
