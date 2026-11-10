#!/usr/bin/env python3
"""Placing by visual servoing on the measured needle tip (user 17:3x '可以再修正一下').

run_1 showed the kinematic needle model is several mm / tens of degrees off after the 74 deg reorientation (affine calibration far
from where it was fitted), so the error is measured in the camera, where the site (YOLO + FS plane: c, n) is measured too:
  tip p and tip direction d = tipmeas.measure (right image arc, left NCC with sub-pixel disparity, point 4 mm back for d).
  target: p = c + h n (stage 1 h = --h1 5 mm, stage 2 h = --hover-mm 2), d = -n.
Each iteration (10 averaged frames, arm still):
  dir error > --tol-deg: rotate about the measured tip by the camera-frame correction (orthonormal part of the calibration),
     <= --rot-step deg;   else position error > --tol-mm: translate by the camera error mapped with the calibration's linear part,
     <= 1.5 mm (stage 1) / 0.5 mm (stage 2);   else stage 1 -> 2, stage 2 -> done.
Mover guards as today; Mover.tip() = the MEASURED needle tip (updated every iteration), so the pad test guards the real tip.
Contact: insertion effort / tracking after every move vs the value after the first retreat step (0.10 / 0.3 mm) -> stop.
First move: retreat --retreat-mm along n (run_1 ended with the tip measured 1.5 mm below the site plane).  The arm holds at the end.
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.spatial.transform import Rotation
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code"); sys.path.insert(0, R0 + "/G_place")
import cv2
from rig import RigIO, Mover, load_kin, Guard, TCP_MM, geodesic_deg
from calib import proj, cam2ecm_mm, rotation_from_affine
import tipmeas

ap = argparse.ArgumentParser(); ap.add_argument("--plan", default="plan_A41_Prep.json"); ap.add_argument("--hover-mm", type=float, default=2.0)
ap.add_argument("--h1", type=float, default=5.0); ap.add_argument("--h-rot", dest="h_rot", type=float, default=8.0); ap.add_argument("--retreat-mm", type=float, default=4.0); ap.add_argument("--tol-mm", type=float, default=0.5)
ap.add_argument("--tol-deg", type=float, default=8.0); ap.add_argument("--rot-step", type=float, default=5.0); ap.add_argument("--max-iters", type=int, default=40)
ap.add_argument("--send", action="store_true"); ap.add_argument("--out", default=None)
a = ap.parse_args()
plan = json.load(open(R0 + "/G_place/" + a.plan)); c, n = np.array(plan["site_cam"]["centre"]), np.array(plan["site_cam"]["normal"])
out = a.out or R0 + "/G_place/vs_" + time.strftime("%H%M%S"); os.makedirs(out, exist_ok=True); logf = open(out + "/vs_log.jsonl", "a")
def log(r): logf.write(json.dumps(dict(t=time.time(), **r), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"); logf.flush()
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); A3 = M[:3]; Rc = rotation_from_affine(M); pad = json.load(open(R0 + "/D_pad/pad.json"))
dcam2ecm = lambda dX: np.asarray(dX) @ np.linalg.inv(A3) * 1e-3          # camera mm displacement -> ECM m (linear part)
n_e = (Rc.T @ n) / np.linalg.norm(Rc.T @ n)

io = RigIO(images=True); kin = load_kin(); time.sleep(2.5); mv = Mover(io, kin, log)
mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = np.array(cal["frozen"]["suj_psm1"]); mv.pad = (np.array(pad["normal"]), pad["d"]); mv.cube = None
mv.needle = cam2ecm_mm(c, M) * 1e-3

def snap_measure(name):
    d = f"{out}/{name}"; os.makedirs(d, exist_ok=True); b = io.frames(10)
    R = np.mean([x[1] for x in b], 0).round().astype(np.uint8); L = np.mean([x[0] for x in b], 0).round().astype(np.uint8)
    cv2.imwrite(d + "/right_rect.png", R); cv2.imwrite(d + "/left_rect.png", L)
    T = mv.setpoint(); jaw_px = proj(np.r_[(T[:3, 3] + T[:3, :3] @ (TCP_MM * 1e-3)) * 1e3, 1] @ M)    # only tells which arc end is the tip
    m, why = tipmeas.measure(R, L, c[2], roi_uv=proj(c), jaw_px=np.asarray(jaw_px))
    json.dump(dict(sp=T.tolist(), js=io.get("js")[1].tolist(), jaw=io.get("jaw")[1].tolist(), t=time.time()), open(d + "/state.json", "w"))
    if m is None: return None, why
    e = m["tip_cam"] - c; h = float(e @ n); lat = float(np.linalg.norm(e - h * n)); ang = float(np.degrees(np.arccos(np.clip(m["dir_cam"] @ (-n), -1, 1))))
    ov = R.copy(); cv2.polylines(ov, [np.round(m["poly"]).astype(np.int32)], False, (0, 255, 255), 2)
    cv2.circle(ov, tuple(np.round(m["tip"]["uv"]).astype(int)), 8, (255, 0, 255), 2); cv2.circle(ov, tuple(np.round(proj(c)).astype(int)), 8, (0, 255, 0), 2)
    cv2.putText(ov, f"{name}: height {h:.2f} mm lateral {lat:.2f} mm dir {ang:.1f} deg (vision)", (10, 30), 0, 0.8, (255, 255, 255), 2); cv2.imwrite(d + "/overlay.png", ov)
    # Mover.tip() := the measured tip (rigid in the tool frame at this pose)
    tip_e = cam2ecm_mm(m["tip_cam"], M) * 1e-3; mv.tip_off_tool = T[:3, :3].T @ (tip_e - T[:3, 3]) - TCP_MM * 1e-3
    r = dict(name=name, tip_cam=m["tip_cam"], dir_cam=m["dir_cam"], height_mm=h, lateral_mm=lat, dir_err_deg=ang, ncc=m["tip"]["ncc"], disp=m["tip"]["disp"])
    log(dict(ev="measure", **r)); print(f"[{name}] vision: height {h:.2f} mm  lateral {lat:.2f} mm  dir {ang:.1f} deg  (ncc {m['tip']['ncc']:.2f})", flush=True)
    return r, "ok"

status = "dry run: measured only"
try:
    r, why = snap_measure("m00_start")
    if r is None: raise SystemExit("no tip measurement at the start: " + why)
    if a.send:
        mv.translate(n_e * a.retreat_mm * 1e-3, tag="retreat_n", lim=(1.0, 1.0)); time.sleep(0.3)
        _, base, _ = io.samples(1.0); stage, h_t = 1, a.h1
        for it in range(1, a.max_iters + 1):
            r, why = snap_measure(f"m{it:02d}")
            if r is None: status = f"STOPPED: no tip measurement ({why})"; break
            p, dvec = np.array(r["tip_cam"]), np.array(r["dir_cam"])
            if r["dir_err_deg"] > a.tol_deg and stage == 1 and abs(r["height_mm"] - a.h_rot) > 1.5:
                e = (a.h_rot - r["height_mm"]) * n; e = e if np.linalg.norm(e) <= 1.5 else e * 1.5 / np.linalg.norm(e)
                mv.translate(dcam2ecm(e), tag="vs_to_rot_height", lim=(1.5, 3.0)); act = f"height to {a.h_rot:.0f} mm for rotating ({np.linalg.norm(e):.2f} mm)"
            elif r["dir_err_deg"] > a.tol_deg:
                w = np.cross(dvec, -n); s = np.linalg.norm(w); ang = min(np.arctan2(s, dvec @ -n), np.radians(a.rot_step))
                dR_cam = Rotation.from_rotvec(w / s * ang).as_matrix(); dR_e = Rc.T @ dR_cam @ Rc
                R_new = dR_e @ mv.setpoint()[:3, :3]; mv.rotate_about_tip(R_new, tag="vs_rotate", lim=(1.5, 3.0)); act = f"rotate {np.degrees(ang):.1f} deg"
            else:
                e = (c + h_t * n) - p; en = float(np.linalg.norm(e))
                if en <= a.tol_mm:
                    if stage == 1: stage, h_t = 2, a.hover_mm; log(dict(ev="stage2", it=it)); print("   -> stage 2 (h = %.1f mm)" % h_t, flush=True); continue
                    status = f"DONE (vision): height {r['height_mm']:.2f} mm, lateral {r['lateral_mm']:.2f} mm, dir {r['dir_err_deg']:.1f} deg; holding"; break
                cap = 1.5 if stage == 1 else 0.5; e = e if en <= cap else e * cap / en
                mv.translate(dcam2ecm(e), tag="vs_translate", lim=(1.5, 3.0)); act = f"translate {np.linalg.norm(e):.2f} mm (stage {stage})"
            time.sleep(0.3); _, eff, err = io.samples(0.3); de = eff - base
            log(dict(ev="move", it=it, act=act, d_ins=float(de[2]), trk_ins_mm=float(err[2] * 1e3))); print(f"   {act}; d_ins {de[2]:+.3f} trk {err[2] * 1e3:.2f} mm", flush=True)
            if abs(de[2]) > 0.10 or abs(err[2]) * 1e3 > 0.3: raise Guard(f"contact? d_ins {de[2]:+.3f} trk {err[2] * 1e3:.2f} mm")
        else:
            status = "max iterations"
        if not status.startswith("DONE"): snap_measure("m99_end")
except Guard as g:
    status = f"STOPPED by guard: {g}"; log(dict(ev="guard", why=str(g)))
except SystemExit as e:
    status = str(e)
except BaseException as ex:
    import traceback; traceback.print_exc(); status = f"STOPPED: {type(ex).__name__} {ex}"
print("\n" + status + "\nout:", out, flush=True); log(dict(ev="end", status=status)); sys.stdout.flush(); os._exit(0)
