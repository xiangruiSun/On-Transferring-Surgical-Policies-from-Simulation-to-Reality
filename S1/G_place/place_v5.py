#!/usr/bin/env python3
"""Placing v5 (2026-10-30, user: no wrist fix -> pure vision, small steps).  The needle is held (user put it in the jaw).

Why: today the PSM1 wrist readings disagree with the real jaw direction by ~70 deg (FK jaw vs image, base pre3), so neither the
kinematic jaw/needle pose nor the effect of a commanded rotation can be predicted.  Translations at fixed orientation are fine
(the calibration itself is translation-only).
Task: needle tip --hover-mm above the site along its normal n, tip tangent along -n.  Site c, n from plan_manual.py.
Loop (every move followed by a fresh measurement = median of --n-cap tipmeas captures):
  1. height: while the tip is below --h-pre over the site -> translate along +n only (<= 1.5 mm); then lateral to c + h_pre n.
  2. direction at h >= --h1 (rotations never below --h1): rotation about the measured tip, <= --rot-step deg per move; the
     commanded ECM rotation u is u = J^-1 w (w = camera rotation vector d -> -n), J starts at the calibration rotation and is
     Broyden-updated with the observed direction change (component perpendicular to d only; spin about d is unobservable).
     Error grew twice in a row -> J reset, step halved.
  3. descent by translation only: h_pre -> h1 -> hover, caps 1.5 / 1.0 / 0.5 mm, each within --tol-mm before the next.
     Below h1 a direction error > --tol-deg + 7 -> back up to h1.
Guards: Mover (IK, goal_reached, setpoint, joint tracking = unexpected contact, pad clearance with the MEASURED tip), insertion
effort jump > 0.10 after a translation, measured tip height < 0.3 mm, measured motion vs commanded translation > 2.5 mm
(perception jump) -> remeasure, 3 failed measurements -> stop.  End: holds; leaving later must be along +n.
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.spatial.transform import Rotation
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code"); sys.path.insert(0, R0 + "/G_place")
import cv2
from rig import RigIO, Mover, load_kin, Guard, TCP_MM
from calib import proj, cam2ecm_mm, rotation_from_affine
import tipmeas

ap = argparse.ArgumentParser(); ap.add_argument("--plan", required=True); ap.add_argument("--send", action="store_true"); ap.add_argument("--out", default=None)
ap.add_argument("--h-pre", type=float, default=12.0); ap.add_argument("--h1", type=float, default=5.0); ap.add_argument("--hover-mm", type=float, default=1.0)
ap.add_argument("--tol-mm", type=float, default=0.5); ap.add_argument("--tol-deg", type=float, default=8.0); ap.add_argument("--rot-step", type=float, default=3.0)
ap.add_argument("--max-iters", type=int, default=80); ap.add_argument("--n-cap", type=int, default=3)
a = ap.parse_args()
plan = json.load(open(a.plan if os.path.isabs(a.plan) else R0 + "/G_place/" + a.plan))
out = a.out or R0 + "/G_place/v5_" + time.strftime("%H%M%S"); os.makedirs(out, exist_ok=True); logf = open(out + "/v5_log.jsonl", "a")
def log(r): logf.write(json.dumps(dict(t=time.time(), **r), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"); logf.flush()
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); A3 = M[:3]; Rc = rotation_from_affine(M); pad = json.load(open(R0 + "/D_pad/pad.json"))
c, n = np.array(plan["site_cam"]["centre"]), np.array(plan["site_cam"]["normal"]); c_e = cam2ecm_mm(c, M) * 1e-3
dcam2ecm = lambda dX: np.asarray(dX) @ np.linalg.inv(A3) * 1e-3
log(dict(ev="start", args=vars(a), plan=a.plan, site_cam=c, n_cam=n))

io = RigIO(images=True); kin = load_kin(); time.sleep(2.5); mv = Mover(io, kin, log)
mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = np.array(cal["frozen"]["suj_psm1"]); mv.pad = (np.array(pad["normal"]), pad["d"]); mv.cube = None; mv.needle = c_e
mv.tip_off_tool = np.array(plan["tip_tool_mm"]) * 1e-3 - TCP_MM * 1e-3
st = dict(jaw_px=None)

def snap_measure(name):
    d = f"{out}/{name}"; os.makedirs(d, exist_ok=True); ps, ds, ms = [], [], []
    for k in range(a.n_cap):
        b = io.frames(10); R = np.mean([x[1] for x in b], 0).round().astype(np.uint8); L = np.mean([x[0] for x in b], 0).round().astype(np.uint8)
        if k == 0: cv2.imwrite(d + "/right_rect.png", R); cv2.imwrite(d + "/left_rect.png", L)
        jp = st["jaw_px"] if st["jaw_px"] is not None else np.asarray(proj(np.r_[(mv.setpoint()[:3, 3] + mv.setpoint()[:3, :3] @ (TCP_MM * 1e-3)) * 1e3, 1] @ M))
        m, why = tipmeas.measure(R, L, c[2], roi_uv=proj(c), roi_px=320, jaw_px=jp)
        if m is None: log(dict(ev="measure_fail", name=name, k=k, why=why)); continue
        ps.append(m["tip_cam"]); ds.append(m["dir_cam"]); ms.append(m)
    T = mv.setpoint()
    json.dump(dict(sp=T.tolist(), js=io.get("js")[1].tolist(), jaw=io.get("jaw")[1].tolist(), jaw_eff=io.get("jaw_eff")[1].tolist(), t=time.time(), n_ok=len(ps)), open(d + "/state.json", "w"))
    if len(ps) < max(2, a.n_cap - 1): print(f"[{name}] no measurement ({len(ps)}/{a.n_cap})", flush=True); return None
    p = np.median(ps, 0); dd = np.mean(ds, 0); dd /= np.linalg.norm(dd); m = ms[0]
    st["jaw_px"] = np.array(m["poly"][0])                                                  # jaw-side end of the arc (the kinematic jaw is wrong today)
    e = p - c; h = float(e @ n); lat = float(np.linalg.norm(e - h * n)); ang = float(np.degrees(np.arccos(np.clip(dd @ (-n), -1, 1))))
    spread = float(max(np.linalg.norm(np.array(ps) - p, axis=1)))
    ov = cv2.imread(d + "/right_rect.png"); cv2.polylines(ov, [np.round(m["poly"]).astype(np.int32)], False, (0, 255, 255), 2)
    cv2.circle(ov, tuple(np.round(proj(p)).astype(int)), 8, (255, 0, 255), 2); cv2.circle(ov, tuple(np.round(proj(c)).astype(int)), 8, (0, 255, 0), 2)
    cv2.line(ov, tuple(np.round(proj(c)).astype(int)), tuple(np.round(proj(c + 6 * n)).astype(int)), (0, 255, 0), 2)
    cv2.line(ov, tuple(np.round(proj(p)).astype(int)), tuple(np.round(proj(p + 6 * dd)).astype(int)), (255, 0, 255), 2)
    cv2.putText(ov, f"{name}: height {h:.2f} lateral {lat:.2f} mm dir {ang:.1f} deg (vision)", (10, 30), 0, 0.8, (255, 255, 255), 2); cv2.imwrite(d + "/overlay.png", ov)
    mv.tip_off_tool = T[:3, :3].T @ (cam2ecm_mm(p, M) * 1e-3 - T[:3, 3]) - TCP_MM * 1e-3   # pad guard / rotation pivot = the measured tip
    r = dict(name=name, p=p, d=dd, height_mm=h, lateral_mm=lat, dir_err_deg=ang, spread_mm=spread, n_ok=len(ps), jaw=float(io.get("jaw")[1][0]), jaw_eff=float(io.get("jaw_eff")[1][0]))
    log(dict(ev="measure", **r)); print(f"[{name}] vision: height {h:6.2f} mm  lateral {lat:5.2f} mm  dir {ang:5.1f} deg  spread {spread:.2f} | jaw {r['jaw']:.3f} eff {r['jaw_eff']:.3f}", flush=True)
    return r

DESC_EFF = 0.30          # run1 (11:4x): a 1.5 mm LIFT left -1.14 N on insertion at rest (user: nothing touching) = friction / cable
st["desc_eff"] = None    # hysteresis after a direction change -> effort is compared only within a run of descent moves (10-28 ADD_12 idea)
def guarded(move, kind):
    _, base, _ = io.samples(0.5); move(); time.sleep(0.3); _, eff, err = io.samples(0.3); de = float((eff - base)[2])
    ref = st["desc_eff"]; dd = None if (kind != "descend" or ref is None) else float(eff[2] - ref)
    log(dict(ev="move", kind=kind, d_ins=de, eff_ins=float(eff[2]), vs_prev_descent=dd, trk_ins_mm=float(err[2] * 1e3)))
    st["desc_eff"] = float(eff[2]) if kind == "descend" else None
    if dd is not None and abs(dd) > DESC_EFF: raise Guard(f"contact? insertion effort {dd:+.3f} vs the previous descent step")
    return de if dd is None else dd

def translate_cam(e, tag, mode=False):
    kind = "descend" if float(np.asarray(e) @ n) < -0.3 else "translate"      # toward the tissue along the site normal
    de = guarded(lambda: mv.translate(dcam2ecm(e), tag=tag, contact_ok=mode, lim=(1.5, 3.0)), kind); print(f"   translate {np.linalg.norm(e):.2f} mm ({tag}, d_ins {de:+.3f})", flush=True); return de

def rotvec_between(x, y):
    w = np.cross(x, y); s = np.linalg.norm(w); return np.zeros(3) if s < 1e-9 else w / s * np.arctan2(s, x @ y)

status = "dry run: nothing sent"
try:
    r = snap_measure("m00")
    if r is None: raise Guard("no initial needle measurement")
    print(f"start: tip {r['height_mm']:.1f} mm above the site plane, lateral {r['lateral_mm']:.1f} mm, direction error {r['dir_err_deg']:.0f} deg", flush=True)
    if a.send:
        J = Rc.copy(); step = np.radians(a.rot_step); worse = 0; stages = [a.h_pre, a.h1, a.hover_mm]; si = 0; miss = 0; prev = r; last_t = None
        for it in range(1, a.max_iters + 1):
            r = snap_measure(f"m{it:02d}")
            if r is None:
                miss += 1
                if miss >= 3: status = "STOPPED: 3 failed tip measurements in a row"; break
                continue
            miss = 0
            if last_t is not None and np.linalg.norm((np.array(r["p"]) - np.array(prev["p"])) - last_t) > 2.5:
                log(dict(ev="perception_jump", moved=np.array(r["p"]) - np.array(prev["p"]), cmd=last_t)); print("   measured motion != commanded translation -> remeasure", flush=True)
                last_t = None; continue
            h, p, d = r["height_mm"], np.array(r["p"]), np.array(r["d"])
            if si >= 1 and h < 0.3: raise Guard(f"measured tip height {h:.2f} mm < 0.3 above the site plane (descent)")
            # 1. climb before anything else
            if si == 0 and h < a.h_pre - 1.0:
                t = n * min(1.5, a.h_pre - h); translate_cam(t, "v5_up", mode="away"); prev, last_t = r, t; continue   # the tip starts on the tissue (below pad.json)
            # 2. direction (only at h >= h1)
            if r["dir_err_deg"] > a.tol_deg and (si == 0 or (si == 1 and h >= a.h1 - 0.5)):
                lat_e = (c + stages[si] * n) - p
                if si == 0 and np.linalg.norm(lat_e) > 3.0:                                   # keep the tip near the axis while turning
                    t = lat_e * min(1.0, 1.5 / np.linalg.norm(lat_e)); translate_cam(t, "v5_recentre"); prev, last_t = r, t; continue
                if prev is not None and prev.get("_u") is not None:                               # Broyden on the observable part
                    u = prev["_u"]; d0 = np.array(prev["d"]); obs = rotvec_between(d0, d); P = np.eye(3) - np.outer(d0, d0)
                    J = J + np.outer(P @ (obs - J @ u), u) / (u @ u)
                    if r["dir_err_deg"] > prev["dir_err_deg"] + 0.5: worse += 1
                    else: worse = 0
                    log(dict(ev="broyden", obs_deg=np.degrees(obs), pred_deg=np.degrees(J @ u), worse=worse))
                    if worse >= 2: J = Rc.copy(); step = max(np.radians(1.0), step / 2); worse = 0; print(f"   direction got worse twice -> J reset, step {np.degrees(step):.1f} deg", flush=True)
                w = rotvec_between(d, -n); u = np.linalg.lstsq(J, w, rcond=None)[0]; un = np.linalg.norm(u)
                if un > step: u = u * step / un
                R_new = Rotation.from_rotvec(u).as_matrix() @ mv.setpoint()[:3, :3]
                de = guarded(lambda: mv.rotate_about_tip(R_new, tag="v5_rotate", lim=(1.5, 3.0)), "rotate")
                print(f"   rotate {np.degrees(np.linalg.norm(u)):.1f} deg (d_ins {de:+.3f})", flush=True); r["_u"] = u; prev, last_t = r, None; continue
            if si == 2 and r["dir_err_deg"] > a.tol_deg + 7:
                si = 1; print(f"   direction {r['dir_err_deg']:.1f} deg near the site -> back to {stages[si]:.0f} mm", flush=True)
            # 3. translation toward the current stage target
            h_t = stages[si]; e = (c + h_t * n) - p; en = float(np.linalg.norm(e))
            if en <= a.tol_mm:
                if si < len(stages) - 1: si += 1; print(f"   -> target height {stages[si]:.1f} mm", flush=True); prev, last_t = r, None; continue
                status = f"DONE (vision): height {r['height_mm']:.2f} mm, lateral {r['lateral_mm']:.2f} mm, dir {r['dir_err_deg']:.1f} deg; holding"; break
            cap = (1.5, 1.0, 0.5)[si]; e = e if en <= cap else e * cap / en
            translate_cam(e, f"v5_to_h{h_t:g}"); prev, last_t = r, e
        else:
            status = "max iterations"
except Guard as g:
    status = f"STOPPED by guard: {g}"; log(dict(ev="guard", why=str(g)))
    try: snap_measure("stopped")
    except Exception: pass
except BaseException as ex:
    import traceback; traceback.print_exc(); status = f"STOPPED: {type(ex).__name__} {ex}"
print("\n" + status + "\nout:", out, flush=True); log(dict(ev="end", status=status)); sys.stdout.flush(); os._exit(0)
