#!/usr/bin/env python3
"""Placing v4 (user 17:5x: whole chain again, approach -> grasp -> placing).  The needle is held (grasp_hold.py).

Task: needle tip 1-3 mm above the suturing site along its normal, tip tangent along -n, lateral <= 1.5 mm.
  site c, n: sticker (YOLO) + FS point-cloud plane within 4 mm (plan_place.py); measured needle tip p / tangent d: tipmeas
  (right-image arc, left NCC sub-pixel, quadratic fit over 8 mm of arc).
Sequence (lessons of run_1 / vs_1):
  A. lift: +--lift-mm along the pad normal (away from the tissue, no dragging);
  B. coarse (kinematic model from the grasp geometry): Mover.safe_goto to the PRE pose, tip at c + --h-pre n, aligned;
  C. vision at h_pre: rotate about the measured tip until dir <= --tol-deg (<= --rot-step per move), then translate to c + h n;
     h: h_pre -> --h1 -> --hover-mm (each reached within --tol-mm before the next); translation <= 1.5 / 1.0 / 0.5 mm per move;
     a rotation is allowed again whenever dir > tol (only at heights >= --h1).
  Guards: all Mover guards (IK, goal_reached, setpoint, tracking error = unexpected contact on every move), Mover.tip() = the
  measured tip; effort: baseline re-taken right before every move; translation |d ins| > 0.10 -> stop; rotation: logged only.
  End: holds.  Leaving the site later must be along +n.
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.spatial.transform import Rotation
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code"); sys.path.insert(0, R0 + "/G_place")
import cv2
from rig import RigIO, Mover, load_kin, Guard, TCP_MM, geodesic_deg, T_of
from calib import proj, cam2ecm_mm, rotation_from_affine
import tipmeas

ap = argparse.ArgumentParser(); ap.add_argument("--plan", required=True); ap.add_argument("--send", action="store_true"); ap.add_argument("--out", default=None)
ap.add_argument("--lift-mm", type=float, default=8.0); ap.add_argument("--h-pre", type=float, default=12.0); ap.add_argument("--h1", type=float, default=5.0)
ap.add_argument("--hover-mm", type=float, default=2.0); ap.add_argument("--tol-mm", type=float, default=0.5); ap.add_argument("--tol-deg", type=float, default=8.0)
ap.add_argument("--rot-step", type=float, default=5.0); ap.add_argument("--max-iters", type=int, default=45); ap.add_argument("--phi-range", type=float, nargs=2, default=[-30, 40])
a = ap.parse_args()
plan = json.load(open(a.plan if os.path.isabs(a.plan) else R0 + "/G_place/" + a.plan))
out = a.out or R0 + "/G_place/v4_" + time.strftime("%H%M%S"); os.makedirs(out, exist_ok=True); logf = open(out + "/v4_log.jsonl", "a")
def log(r): logf.write(json.dumps(dict(t=time.time(), **r), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"); logf.flush()
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); A3 = M[:3]; Rc = rotation_from_affine(M); pad = json.load(open(R0 + "/D_pad/pad.json"))
c, n = np.array(plan["site_cam"]["centre"]), np.array(plan["site_cam"]["normal"]); c_e, n_e = cam2ecm_mm(c, M) * 1e-3, (Rc.T @ n) / np.linalg.norm(Rc.T @ n)
tip_tool, tan_tool = np.array(plan["tip_tool_mm"]) * 1e-3, np.array(plan["tan_tool"])
dcam2ecm = lambda dX: np.asarray(dX) @ np.linalg.inv(A3) * 1e-3

io = RigIO(images=True); kin = load_kin(); time.sleep(2.5); mv = Mover(io, kin, log)
mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = np.array(cal["frozen"]["suj_psm1"]); mv.pad = (np.array(pad["normal"]), pad["d"]); mv.cube = None; mv.needle = c_e
mv.tip_off_tool = tip_tool - TCP_MM * 1e-3                       # kinematic needle tip until the first measurement

def rot_between(x, y):
    w = np.cross(x, y); s = np.linalg.norm(w); return np.eye(3) if s < 1e-9 else Rotation.from_rotvec(w / s * np.arctan2(s, x @ y)).as_matrix()

def pre_pose():
    H = mv.setpoint(); js = io.get("js")[1][:6]; suj = io.get("suj")[1]; want = -n_e; R_al = rot_between(H[:3, :3] @ tan_tool, want) @ H[:3, :3]; best = None
    for phi in np.arange(a.phi_range[0], a.phi_range[1] + 0.1, 5.0):
        Rg = Rotation.from_rotvec(np.radians(phi) * want).as_matrix() @ R_al; ms = []
        for h in (a.hover_mm, a.h_pre):
            hinge = c_e + n_e * h * 1e-3 - Rg @ tip_tool; ok, q, _ = kin.check_pose(hinge, Rg, js, suj, min_deg=3.0, min_mm=5.0)
            if not ok: ms = None; break
            ms.append(float(np.min(kin.margins(q)[[0, 1, 3, 4, 5]])))
        if ms and (best is None or min(ms) > best[0]): best = (min(ms), float(phi), Rg)
    if best is None: raise Guard("no IK-feasible spin for the placing poses")
    return T_of(c_e + n_e * a.h_pre * 1e-3 - best[2] @ tip_tool, best[2]), best[1]

def snap_measure(name):
    d = f"{out}/{name}"; os.makedirs(d, exist_ok=True); b = io.frames(10)
    R = np.mean([x[1] for x in b], 0).round().astype(np.uint8); L = np.mean([x[0] for x in b], 0).round().astype(np.uint8)
    cv2.imwrite(d + "/right_rect.png", R); cv2.imwrite(d + "/left_rect.png", L)
    T = mv.setpoint(); jaw_px = proj(np.r_[(T[:3, 3] + T[:3, :3] @ (TCP_MM * 1e-3)) * 1e3, 1] @ M)
    json.dump(dict(sp=T.tolist(), js=io.get("js")[1].tolist(), jaw=io.get("jaw")[1].tolist(), jaw_eff=io.get("jaw_eff")[1].tolist(), t=time.time()), open(d + "/state.json", "w"))
    m, why = tipmeas.measure(R, L, c[2], roi_uv=proj(c), roi_px=320, jaw_px=np.asarray(jaw_px))
    if m is None: log(dict(ev="measure_fail", name=name, why=why)); print(f"[{name}] no measurement: {why}", flush=True); return None
    e = m["tip_cam"] - c; h = float(e @ n); lat = float(np.linalg.norm(e - h * n)); ang = float(np.degrees(np.arccos(np.clip(m["dir_cam"] @ (-n), -1, 1))))
    ov = R.copy(); cv2.polylines(ov, [np.round(m["poly"]).astype(np.int32)], False, (0, 255, 255), 2)
    cv2.circle(ov, tuple(np.round(m["tip"]["uv"]).astype(int)), 8, (255, 0, 255), 2); cv2.circle(ov, tuple(np.round(proj(c)).astype(int)), 8, (0, 255, 0), 2)
    cv2.putText(ov, f"{name}: height {h:.2f} lateral {lat:.2f} mm dir {ang:.1f} deg (vision)", (10, 30), 0, 0.8, (255, 255, 255), 2); cv2.imwrite(d + "/overlay.png", ov)
    mv.tip_off_tool = T[:3, :3].T @ (cam2ecm_mm(m["tip_cam"], M) * 1e-3 - T[:3, 3]) - TCP_MM * 1e-3
    r = dict(name=name, p=m["tip_cam"], d=m["dir_cam"], height_mm=h, lateral_mm=lat, dir_err_deg=ang, ncc=m["tip"]["ncc"], jaw=float(io.get("jaw")[1][0]), jaw_eff=float(io.get("jaw_eff")[1][0]))
    log(dict(ev="measure", **r)); print(f"[{name}] vision: height {h:6.2f} mm  lateral {lat:5.2f} mm  dir {ang:5.1f} deg  | jaw {r['jaw']:.3f} eff {r['jaw_eff']:.3f}", flush=True)
    return r

def guarded(move, kind):
    _, base, _ = io.samples(0.5); move(); time.sleep(0.3); _, eff, err = io.samples(0.3); de = float((eff - base)[2])
    log(dict(ev="move", kind=kind, d_ins=de, trk_ins_mm=float(err[2] * 1e3)))
    if kind == "translate" and abs(de) > 0.10: raise Guard(f"contact? insertion effort {de:+.3f} after a translation")
    return de

status = "dry run: nothing sent"
try:
    T_pre, phi = pre_pose(); log(dict(ev="plan", phi=phi, T_pre=T_pre)); print(f"PRE pose: spin {phi:+.0f} deg, tip {a.h_pre:.0f} mm above the site (kinematic model)", flush=True)
    snap_measure("m00_hold")
    if a.send:
        guarded(lambda: mv.translate(mv.pad[0] * a.lift_mm * 1e-3, tag="v4_lift", lim=(1.5, 3.0)), "translate")
        mv.tip_off_tool = tip_tool - TCP_MM * 1e-3; T_pre, phi = pre_pose(); mv.safe_goto(T_pre, tag="v4_to_pre")
        stages = [a.h_pre, a.h1, a.hover_mm]; si = 0; miss = 0
        for it in range(1, a.max_iters + 1):
            r = snap_measure(f"m{it:02d}")
            if r is None:
                miss += 1
                if miss >= 3: status = "STOPPED: 3 failed tip measurements in a row"; break
                continue
            miss = 0; h_t = stages[si]
            if r["dir_err_deg"] > a.tol_deg and h_t >= a.h1:
                w = np.cross(r["d"], -n); s = np.linalg.norm(w); ang = min(np.arctan2(s, r["d"] @ -n), np.radians(a.rot_step))
                R_new = (Rc.T @ Rotation.from_rotvec(w / s * ang).as_matrix() @ Rc) @ mv.setpoint()[:3, :3]
                de = guarded(lambda: mv.rotate_about_tip(R_new, tag="v4_rotate", lim=(1.5, 3.0)), "rotate"); print(f"   rotate {np.degrees(ang):.1f} deg (d_ins {de:+.3f})", flush=True); continue
            e = (c + h_t * n) - np.array(r["p"]); en = float(np.linalg.norm(e))
            if en <= a.tol_mm:
                if si < len(stages) - 1: si += 1; print(f"   -> target height {stages[si]:.1f} mm", flush=True); continue
                status = f"DONE (vision): height {r['height_mm']:.2f} mm, lateral {r['lateral_mm']:.2f} mm, dir {r['dir_err_deg']:.1f} deg; holding"; break
            cap = (1.5, 1.0, 0.5)[si]; e = e if en <= cap else e * cap / en
            de = guarded(lambda: mv.translate(dcam2ecm(e), tag="v4_translate", lim=(1.5, 3.0)), "translate"); print(f"   translate {np.linalg.norm(e):.2f} mm toward h {h_t:.1f} (d_ins {de:+.3f})", flush=True)
        else:
            status = "max iterations"
except Guard as g:
    status = f"STOPPED by guard: {g}"; log(dict(ev="guard", why=str(g))); snap_measure("stopped")
except BaseException as ex:
    import traceback; traceback.print_exc(); status = f"STOPPED: {type(ex).__name__} {ex}"
print("\n" + status + "\nout:", out, flush=True); log(dict(ev="end", status=status)); sys.stdout.flush(); os._exit(0)
