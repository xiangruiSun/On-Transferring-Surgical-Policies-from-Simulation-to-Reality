#!/usr/bin/env python3
"""Placing v3 (user 17:2x): needle tip 1-3 mm above the suturing site, ready for insertion; the needle stays in the jaw.

Task: tip at c + h n (h = --hover-mm, default 2; success window 1-3 mm along n), tip tangent = -n (<= 5 deg), lateral <= 1.5 mm.
  c, n = sticker site (YOLO + FS plane, plan_place.py);  tip / tip tangent in the tool frame from the grasp geometry (pre-grasp
  needle arc + the pose at closing, plan_place.py; checked on the hold image).  Camera <-> ECM = today's calibration.
Motion (rig.Mover, all of today's guards; the Mover 'tip' is set to the NEEDLE tip, so the pad test guards the needle tip):
  1. safe_goto to ABOVE: tip at c + --above-mm n with the final orientation (spin phi about n = the IK-feasible one with the largest
     smallest joint margin among phi in --phi-range);
  2. descend along -n: 1 mm steps to h + 2 mm, then 0.5 mm steps to h; after every step the insertion effort / tracking error
     vs the value after the first step (0.10 / 0.3 mm) -> contact = stop there (hardware-class);
  3. FINAL: frames + state; kinematic tip height / lateral / direction errors logged; the arm holds (no release).
    python3 place_v3.py --plan plan_A41_Prep.json            # dry: plan + path pre-check, nothing sent
    python3 place_v3.py --plan plan_A41_Prep.json --send
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.spatial.transform import Rotation
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code")
import cv2
from rig import RigIO, Mover, load_kin, Guard, TCP_MM, geodesic_deg, T_of
from calib import proj

ap = argparse.ArgumentParser(); ap.add_argument("--plan", required=True); ap.add_argument("--send", action="store_true")
ap.add_argument("--hover-mm", type=float, default=2.0); ap.add_argument("--above-mm", type=float, default=15.0)
ap.add_argument("--phi-range", type=float, nargs=2, default=[-30, 40]); ap.add_argument("--out", default=None)
a = ap.parse_args()
plan = json.load(open(a.plan if os.path.isabs(a.plan) else R0 + "/G_place/" + a.plan))
out = a.out or R0 + "/G_place/run_" + time.strftime("%H%M%S"); os.makedirs(out, exist_ok=True)
logf = open(out + "/place_log.jsonl", "a")
def log(r): logf.write(json.dumps(dict(t=time.time(), **r), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"); logf.flush()
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); pad = json.load(open(R0 + "/D_pad/pad.json"))
tip_tool = np.array(plan["tip_tool_mm"]) * 1e-3; tan_tool = np.array(plan["tan_tool"]); c = np.array(plan["site_ecm_mm"]) * 1e-3; n = np.array(plan["normal_ecm"])
CONTACT_INS, CONTACT_TRK_MM = 0.10, 0.3

io = RigIO(images=True); kin = load_kin(); time.sleep(2.5); mv = Mover(io, kin, log)
mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = np.array(cal["frozen"]["suj_psm1"]); mv.pad = (np.array(pad["normal"]), pad["d"])
mv.tip_off_tool = tip_tool - TCP_MM * 1e-3            # Mover.tip() = the needle tip (pad / near checks on the needle tip)
mv.needle = c; mv.cube = None
H = mv.setpoint(); js = io.get("js")[1][:6]; suj = io.get("suj")[1]
padmm = lambda p: mv.pad_dist_mm(p)
def tip_of_pose(T): return T[:3, :3] @ tip_tool + T[:3, 3]

def rot_between(x, y):
    w = np.cross(x, y); s = np.linalg.norm(w); return np.eye(3) if s < 1e-9 else Rotation.from_rotvec(w / s * np.arctan2(s, x @ y)).as_matrix()
want = -n; R_align = rot_between(H[:3, :3] @ tan_tool, want) @ H[:3, :3]
best = None
for phi in np.arange(a.phi_range[0], a.phi_range[1] + 0.1, 5.0):
    Rg = Rotation.from_rotvec(np.radians(phi) * want).as_matrix() @ R_align; ms = []
    for h in (a.hover_mm, a.above_mm):
        hinge = c + n * h * 1e-3 - Rg @ tip_tool; ok, q, why = kin.check_pose(hinge, Rg, js, suj, min_deg=3.0, min_mm=5.0)
        if not ok: ms = None; break
        ms.append(float(np.min(kin.margins(q)[[0, 1, 3, 4, 5]])))
    if ms and (best is None or min(ms) > best[0]): best = (min(ms), float(phi), Rg)
if best is None: raise SystemExit("no IK-feasible spin in the range")
_, phi, Rg = best
T_final = T_of(c + n * a.hover_mm * 1e-3 - Rg @ tip_tool, Rg); T_above = T_of(c + n * a.above_mm * 1e-3 - Rg @ tip_tool, Rg)
info = dict(phi=phi, rot_now_to_goal_deg=geodesic_deg(H[:3, :3], Rg), tip_now_to_site_mm=float(np.linalg.norm(tip_of_pose(H) - c) * 1e3),
            final=dict(hinge_pad=padmm(T_final[:3, 3]), jaw_tcp_pad=padmm(T_final[:3, 3] + Rg @ (TCP_MM * 1e-3)), tip_pad=padmm(tip_of_pose(T_final))),
            above=dict(hinge_pad=padmm(T_above[:3, 3]), tip_pad=padmm(tip_of_pose(T_above))))
print(json.dumps(info, indent=1)); log(dict(ev="plan", send=a.send, **info, T_final=T_final, T_above=T_above))
bad = mv.precheck_path(T_above, n=30)
print("pre-check of the direct path to ABOVE:", bad[:3] if bad else "ok (safe_goto lifts / rotates about the tip / translates, each sub-step checked)")

def snap(name, k=10):
    d = f"{out}/{name}"; os.makedirs(d, exist_ok=True); b = io.frames(k)
    R = np.mean([x[1] for x in b], 0).round().astype(np.uint8); L = np.mean([x[0] for x in b], 0).round().astype(np.uint8)
    cv2.imwrite(d + "/right_rect.png", R); cv2.imwrite(d + "/left_rect.png", L)
    T = mv.setpoint(); tp = tip_of_pose(T); uv = proj(np.r_[tp * 1e3, 1] @ M); cs = proj(np.r_[c * 1e3, 1] @ M)
    ov = R.copy(); cv2.circle(ov, tuple(np.round(uv).astype(int)), 8, (0, 0, 255), 2); cv2.circle(ov, tuple(np.round(cs).astype(int)), 8, (0, 255, 0), 2)
    cv2.putText(ov, f"{name}: red = predicted needle tip, green = site", (10, 30), 0, 0.8, (255, 255, 255), 2); cv2.imwrite(d + "/overlay.png", ov)
    json.dump(dict(sp=T.tolist(), js=io.get("js")[1].tolist(), jaw=io.get("jaw")[1].tolist(), t=time.time()), open(d + "/state.json", "w"))

def errors(T):
    tp = tip_of_pose(T); e = tp - c; h = float(e @ n); lat = float(np.linalg.norm(e - h * n) * 1e3)
    d = T[:3, :3] @ tan_tool; return dict(height_mm=h * 1e3, lateral_mm=lat, dir_err_deg=float(np.degrees(np.arccos(np.clip(d @ want, -1, 1)))))

status = "dry run: nothing sent"
try:
    snap("start", 6)
    if a.send:
        mv.safe_goto(T_above, tag="to_above"); snap("above"); log(dict(ev="above", **errors(mv.setpoint())))
        print("ABOVE:", errors(mv.setpoint()), flush=True)
        _, base, _ = io.samples(1.0); first = True
        while True:
            h = errors(mv.setpoint())["height_mm"]
            if h <= a.hover_mm + 0.05: break
            step = min(1.0 if h > a.hover_mm + 2.0 else 0.5, h - a.hover_mm)
            mv.translate(-n * step * 1e-3, tag="descend_n", lim=(1.0, 1.0))
            pos, eff, err = io.samples(0.3)
            if first: base, first = eff, False; continue
            de = eff - base; e = errors(mv.setpoint()); log(dict(ev="descend_probe", d_ins=float(de[2]), trk_ins_mm=float(err[2] * 1e3), **e))
            print(f"   h {e['height_mm']:.2f} mm  d_ins {de[2]:+.3f}  trk {err[2] * 1e3:.2f} mm", flush=True)
            if abs(de[2]) > CONTACT_INS or abs(err[2]) * 1e3 > CONTACT_TRK_MM: raise Guard(f"contact while descending to the site (d_ins {de[2]:+.3f}, trk {err[2] * 1e3:.2f} mm) at height {e['height_mm']:.2f} mm")
        snap("final"); e = errors(mv.setpoint())
        ok = 1.0 <= e["height_mm"] <= 3.0 and e["lateral_mm"] <= 1.5 and e["dir_err_deg"] <= 5.0
        status = f"FINAL (kinematic): height {e['height_mm']:.2f} mm, lateral {e['lateral_mm']:.2f} mm, dir {e['dir_err_deg']:.2f} deg -> {'within' if ok else 'OUTSIDE'} the task window; holding"
        log(dict(ev="final", ok=ok, **e))
except Guard as g:
    status = f"STOPPED by guard: {g}"; log(dict(ev="guard", why=str(g))); snap("stopped", 6)
except BaseException as ex:
    import traceback; traceback.print_exc(); status = f"STOPPED: {type(ex).__name__} {ex}"; log(dict(ev="error", why=str(ex)))
print("\n" + status + "\nout:", out, flush=True); log(dict(ev="end", status=status)); sys.stdout.flush(); os._exit(0)
