#!/usr/bin/env python3
"""One full trial, reach -> grasp -> place, run by an operator in a terminal (2026-11-09 setting, blue sticker = entry).

    cd ~/rl_live/full_run_1109 && python3 trial.py

The robot steps are the ones of trials/T01 (the first success), executed by run.py (tested code, unchanged); this script only
sequences them and collects the operator's clicks. Clicks are needed because there is no automatic held-needle detector:
  1. TAIL of the needle (the end the thread leaves) before the grasp
  2. needle points TAIL -> TIP after the grasp (>= 6 points along the needle, in order)
  3. before every placing move: the needle TIP in the right image, then in the left image (same row; a guide line is drawn),
     and for rotations also a needle point ~2 mm behind the tip.
Click windows: left click = point, u = undo, ENTER = done, f = back to the full image, ESC = abort the trial (arm stays where it is).
Every moving stage is dry-run first and only sent if the dry run passes (run.py); guard stops abort the trial.
At the end: verdict y/n -> trials/T<nn> (success) or dev_unrecorded/<name> (not counted); then optional release + back to base.
"""
import glob, json, os, re, shutil, subprocess, sys, time
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__)); S1 = os.path.expanduser("~/rl_live/sim2real_20261106/S1")
FB = 1860.724932 * 5.4476353950418295; DV = 2.2                    # stereo baseline x focal; left row = right row + 2.2 px
ROT_STEP = 25.0                                                     # deg per rotation about the tip (T01: 25 + 24.3)
HEIGHTS = (15.0, 8.0, 3.5)                                          # mm above the entry: re-centre, approach, final (T01)
ENVSH = f"source {S1}/trials/env.sh"
WIN_XY = (60, 40)
os.environ.setdefault("DISPLAY", ":1"); os.environ.setdefault("XAUTHORITY", "/run/user/%d/gdm/Xauthority" % os.getuid())


# ------------------------------------------------------------------ helpers
def run(stage_args, trial, must_pass=True):
    """one run.py stage; prints its output; returns it."""
    env = dict(os.environ, TRIAL=trial)
    p = subprocess.run(["bash", "-c", f"{ENVSH}; python3 {HERE}/run.py {stage_args}"], env=env, capture_output=True, text=True)
    out = "\n".join(l for l in (p.stdout + p.stderr).splitlines() if not l.startswith("$") and "autocast" not in l and "Warn" not in l)
    print(out.strip()[-1500:], flush=True)
    if must_pass and p.returncode != 0: sys.exit(f"stage '{stage_args.split()[0]}' stopped (exit {p.returncode}) -> trial aborted, arm left where it is; see {trial}/log.txt")
    return out


def ask(q): return input(q + " [y/n] ").strip().lower().startswith("y")


def clicker(img, title, n=None, scale=1.0, origin=(0, 0), guide_row=None, marks=()):
    """click points on img (shown at `scale`); returns full-res pixel list, 'FULL' (f pressed) or None (ESC)."""
    clicks = []
    def cb(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN: clicks.append((origin[0] + x / scale, origin[1] + y / scale))
    v0 = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE); cv2.moveWindow(title, *WIN_XY); cv2.setMouseCallback(title, cb)
    while True:
        v = v0.copy(); f = lambda p: (int((p[0] - origin[0]) * scale), int((p[1] - origin[1]) * scale))
        if guide_row is not None: y = int((guide_row - origin[1]) * scale); cv2.line(v, (0, y), (v.shape[1], y), (0, 255, 255), 1)
        for m in marks: cv2.drawMarker(v, f(m), (255, 255, 0), cv2.MARKER_CROSS, 18, 1)
        for i, p in enumerate(clicks):
            cv2.circle(v, f(p), 4, (0, 0, 255), -1)
            if i: cv2.line(v, f(clicks[i - 1]), f(p), (0, 0, 255), 1)
        cv2.putText(v, f"{title}: click, u=undo, ENTER=done, f=full image, ESC=abort", (8, 22), 0, 0.6, (0, 0, 0), 3)
        cv2.putText(v, f"{title}: click, u=undo, ENTER=done, f=full image, ESC=abort", (8, 22), 0, 0.6, (255, 255, 255), 1)
        cv2.imshow(title, v); k = cv2.waitKey(30) & 0xFF
        if n is not None and len(clicks) > n: clicks.pop()
        if k == 27: cv2.destroyAllWindows(); cv2.waitKey(1); return None
        if k == ord("u") and clicks: clicks.pop()
        if k == ord("f"): cv2.destroyAllWindows(); cv2.waitKey(1); return "FULL"
        if k in (10, 13) and clicks and (n is None or len(clicks) == n): cv2.destroyAllWindows(); cv2.waitKey(1); return clicks


def pick_zoom(img, title, approx, n=1, guide_row=None, zoom=3, half=90):
    """precise click(s) in a zoomed crop around approx; 'f' falls back to an approximate click on the full image first."""
    H, W = img.shape[:2]
    while True:
        if approx is None:
            r = clicker(img, title + " (approximate)", 1, scale=0.75)
            if r is None: return None
            if r == "FULL": continue
            approx = r[0]
        x0 = int(np.clip(approx[0] - half, 0, W - 2 * half)); y0 = int(np.clip(approx[1] - half, 0, H - 2 * half))
        r = clicker(img[y0:y0 + 2 * half, x0:x0 + 2 * half], title, n, scale=zoom, origin=(x0, y0), guide_row=guide_row)
        if r == "FULL": approx = None; continue
        return r


def read_tip(snap, title, pred_R=None, pred_disp=None, body=False):
    """operator reads the tip (right, then left on the same row) and optionally a needle point behind the tip (right)."""
    R = cv2.imread(snap + "/right_rect.png"); L = cv2.imread(snap + "/left_rect.png")
    r = pick_zoom(R, f"{title} - RIGHT image: click the needle TIP" + (" then a needle point ~2 mm behind it" if body else ""), pred_R, 2 if body else 1)
    if r is None: sys.exit("aborted by the operator (arm left where it is)")
    tipR = r[0]; d0 = pred_disp if pred_disp else 130.0
    l = pick_zoom(L, f"{title} - LEFT image: click the same TIP (yellow line = same row)", (tipR[0] - d0, tipR[1] + DV), 1, guide_row=tipR[1] + DV)
    if l is None: sys.exit("aborted by the operator (arm left where it is)")
    tipL = l[0]; d = tipR[0] - tipL[0]
    print(f"tip right ({tipR[0]:.0f},{tipR[1]:.0f}) left ({tipL[0]:.0f},{tipL[1]:.0f}) disparity {d:.1f} -> depth {FB / d:.1f} mm", flush=True)
    return tipR, tipL, (r[1] if body else None), d


def expected(out):
    """phase2 prints 'expected tip at right [u, v] left [u, v]' -> (uvR, disparity)."""
    m = re.search(r"expected tip at right \[([\d.\-]+), ([\d.\-]+)\] left \[([\d.\-]+), ([\d.\-]+)\]", out)
    return ((float(m[1]), float(m[2])), float(m[1]) - float(m[3])) if m else (None, None)


def fmt(p): return f"{p[0]:.0f},{p[1]:.0f}"


# ------------------------------------------------------------------ trial
def main():
    os.chdir(HERE)
    n_att = max([int(m[0]) for p in glob.glob(S1 + "/F_attempts/A1??_Prep") for m in [re.findall(r"A1(\d\d)_Prep", p)] if m] + [68]) + 1   # next free grasp number (A1nn)
    name = time.strftime("run_%m%d_%H%M%S"); trial = f"../dev_unrecorded/{name}"; out_dir = f"{HERE}/dev_unrecorded/{name}"
    print(f"== trial {name} (grasp attempt A{100 + n_att}); data -> {out_dir}", flush=True)

    # 0. rig: calibration valid, arm at the base with an empty jaw
    rig = subprocess.run(["bash", "-c", f"{ENVSH}; cd {S1}/G_place; timeout 40 python3 check_rig.py 2>&1"], capture_output=True, text=True).stdout
    print("\n".join(rig.strip().splitlines()[-3:]), flush=True)
    if "OK - ready" not in rig:
        if "state checks failed" in rig: sys.exit("ECM / setup joints moved since the calibration (or PSM1 not enabled) -> recalibrate / check the console first")
        if "not at the base" in rig and ask("arm is not at the base: move it back now (jaw must be empty)?"):
            print(subprocess.run(["bash", "-c", f"{ENVSH}; timeout 150 python3 {S1}/code/recover_to_base_1109.py 2>&1 | tail -2"], capture_output=True, text=True).stdout)
        else: sys.exit("rig not ready")

    # 1. tail click, then reach + grasp (RL Reach, frozen S1 code)
    snap = out_dir + "/pre"; os.makedirs(out_dir, exist_ok=True)
    subprocess.run(["bash", "-c", f"{ENVSH}; python3 {S1}/code/r6_local/r6_snap.py {snap} >/dev/null 2>&1"])
    t = pick_zoom(cv2.imread(snap + "/right_rect.png"), "needle TAIL (end where the thread leaves)", None, 1)
    if t is None: sys.exit("aborted")
    os.environ["NEEDLE_TAIL_HINT"] = fmt(t[0]); os.environ["NEEDLE_PX_SCALE"] = "2"
    out = run(f"grasp {n_att}", trial, must_pass=False)
    if "GRASPED" not in out or "NOT GRASPED" in out:
        print("not grasped -> trial not counted; run 'python3 run.py release' if the jaw holds anything, re-place the needle, try again"); return

    # 2. needle shape -> grip -> plan
    run("snap s0 --fs", trial)
    R = cv2.imread(out_dir + "/s0/right_rect.png")
    pts = clicker(R, "needle points: TAIL (at the jaw) -> TIP, >= 6 points, ENTER", None, scale=0.75)
    if pts in (None, "FULL") or len(pts) < 6: sys.exit("needle points needed (>= 6)")
    out = run("needle s0 " + " ".join(fmt(p) for p in pts), trial)
    plan = json.load(open(out_dir + "/rot_plan.json")); total = plan["deg"]; print(f"rotation needed {total:.1f} deg", flush=True)

    # 3. rotations about the tip (<= ROT_STEP each)
    last, done, k, disp = "s0", 0.0, 1, None
    tipR = pts[-1]
    while total - done > 0.5:
        step = min(ROT_STEP, total - done)
        tR, tL, body, disp = read_tip(f"{out_dir}/{last}", f"rotation {k} ({step:.1f} deg)", tipR, disp, body=True)
        run(f"rot r{k} {step:.1f} {fmt(tR)} {fmt(body)} {FB / disp:.1f} {last}", trial)
        done += step; last = f"r{k}"; tipR = tR; k += 1

    # 4. down to entry + 15 / 8 / 3.5 mm
    pred = tipR
    for h in HEIGHTS:
        tR, tL, _, disp = read_tip(f"{out_dir}/{last}", f"before the move to {h} mm", pred, disp)
        name_h = {15.0: "rc1", 8.0: "c8", 3.5: "c35"}.get(h, f"h{h}")
        o = run(f"down {name_h} {h} {fmt(tR)} {fmt(tL)}", trial)
        pred, d_exp = expected(o); disp = d_exp or disp; last = name_h

    # 5. final check, verdict, record, release
    tR, tL, _, _ = read_tip(f"{out_dir}/{last}", "FINAL tip", pred, disp)
    run(f"metric {fmt(tR)} {fmt(tL)}", trial)
    if ask("success (tip at the sticker entry)?"):
        nums = [int(x[1:]) for x in os.listdir(HERE + "/trials") if re.fullmatch(r"T\d+", x)]; tn = f"T{max(nums + [0]) + 1:02d}"
        shutil.move(out_dir, f"{HERE}/trials/{tn}"); open(f"{HERE}/trials/{tn}/VERDICT.txt", "w").write(f"{name}: SUCCESS (operator)\n"); print("recorded as", tn)
    else: print("not counted; data kept in", out_dir)
    if ask("release the needle here and go back to the base?"): run("release", trial)


if __name__ == "__main__":
    main()
