#!/usr/bin/env python3
"""Placing v7 (2026-10-30): POSITION ONLY (no rotation; the wrist model is wrong today and the stereo tip direction is not reliable),
fused tip estimate.

run4: the single-point NCC tip depth jumps ~1 mm between captures (> the 0.75 mm steps), so v6's per-step gate stopped it.
Translations are known (kinematics, calibration A3).  Fused estimate: p_hat = mean_k(raw_k - cum_k) + cum_now over the last --k
accepted raw measurements (cum = summed commanded camera displacement).  A raw reading > --outlier-mm from the prediction is
dropped; 3 in a row -> STOP (the measurement does not follow the robot).
Stages: --stages (default 5 1) mm above the site along n; intermediate tolerance --tol-stage, final --tol-mm.
Every move: translation (camera) * --gain, cap per stage (1.0 / 0.5 mm), toward the tissue = 'descend' (effort vs the previous descent
step, > 0.30 -> STOP); Mover guards (pad clearance with the measured tip, joint tracking, goal_reached, IK).  Final: hold.
"""
import argparse, json, os, sys, time
import numpy as np
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code"); sys.path.insert(0, R0 + "/G_place")
import cv2
from rig import RigIO, Mover, load_kin, Guard, TCP_MM
from calib import proj, cam2ecm_mm
import tipmeas

ap = argparse.ArgumentParser(); ap.add_argument("--plan", required=True); ap.add_argument("--send", action="store_true"); ap.add_argument("--out", default=None)
ap.add_argument("--roi", type=float, nargs=2, required=True); ap.add_argument("--jaw", type=float, nargs=2, required=True); ap.add_argument("--roi-px", type=int, default=300)
ap.add_argument("--stages", type=float, nargs="+", default=[5.0, 1.0]); ap.add_argument("--tol-stage", type=float, default=1.0); ap.add_argument("--tol-mm", type=float, default=0.5)
ap.add_argument("--gain", type=float, default=0.75); ap.add_argument("--k", type=int, default=5); ap.add_argument("--outlier-mm", type=float, default=2.0)
ap.add_argument("--max-iters", type=int, default=60); ap.add_argument("--n-cap", type=int, default=3)
ap.add_argument("--init-p", type=float, nargs=3, default=None)   # v7_run1: seed = last fused tip (the arm has not moved since)
ap.add_argument("--blind-below", type=float, default=5.0)        # v7_run1: near the sticker the arc detection jumps onto it -> below this height a missing / outlier reading = use the prediction
a = ap.parse_args()
plan = json.load(open(a.plan if os.path.isabs(a.plan) else R0 + "/G_place/" + a.plan))
out = a.out or R0 + "/G_place/v7_" + time.strftime("%H%M%S"); os.makedirs(out, exist_ok=True); logf = open(out + "/v7_log.jsonl", "a")
def log(r): logf.write(json.dumps(dict(t=time.time(), **r), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"); logf.flush()
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); A3 = M[:3]; pad = json.load(open(R0 + "/D_pad/pad.json"))
c, n = np.array(plan["site_cam"]["centre"]), np.array(plan["site_cam"]["normal"]); c_e = cam2ecm_mm(c, M) * 1e-3
dcam2ecm = lambda dX: np.asarray(dX) @ np.linalg.inv(A3) * 1e-3
log(dict(ev="start", args=vars(a), site_cam=c, n_cam=n))
io = RigIO(images=True); kin = load_kin(); time.sleep(2.5); mv = Mover(io, kin, log)
mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = np.array(cal["frozen"]["suj_psm1"]); mv.pad = (np.array(pad["normal"]), pad["d"]); mv.cube = None; mv.needle = c_e
mv.tip_off_tool = np.array(plan["tip_tool_mm"]) * 1e-3 - TCP_MM * 1e-3
st = dict(roi=np.array(a.roi), jaw=np.array(a.jaw), cum=np.zeros(3), hist=[] if a.init_p is None else [(np.array(a.init_p), np.zeros(3))] * 3, desc_eff=None)

def raw_measure(name):
    d = f"{out}/{name}"; os.makedirs(d, exist_ok=True); ps, ms = [], []
    for k in range(a.n_cap):
        b = io.frames(10); R = np.mean([x[1] for x in b], 0).round().astype(np.uint8); L = np.mean([x[0] for x in b], 0).round().astype(np.uint8)
        if k == 0: cv2.imwrite(d + "/right_rect.png", R); cv2.imwrite(d + "/left_rect.png", L)
        m, why = tipmeas.measure(R, L, c[2], roi_uv=st["roi"], roi_px=a.roi_px, jaw_px=st["jaw"])
        if m is None: log(dict(ev="measure_fail", name=name, k=k, why=why)); continue
        ps.append(m["tip_cam"]); ms.append(m)
    json.dump(dict(sp=mv.setpoint().tolist(), jaw=io.get("jaw")[1].tolist(), jaw_eff=io.get("jaw_eff")[1].tolist(), t=time.time(), n_ok=len(ps)), open(d + "/state.json", "w"))
    if len(ps) < 2: return None, None
    return np.median(ps, 0), ms[0]

def report(name, p, raw, m, fused_n):
    e = p - c; h = float(e @ n); lat = float(np.linalg.norm(e - h * n))
    ov = cv2.imread(f"{out}/{name}/right_rect.png"); cv2.polylines(ov, [np.round(m["poly"]).astype(np.int32)], False, (0, 255, 255), 2)
    cv2.circle(ov, tuple(np.round(proj(p)).astype(int)), 8, (255, 0, 255), 2); cv2.circle(ov, tuple(np.round(proj(c)).astype(int)), 8, (0, 255, 0), 2)
    cv2.line(ov, tuple(np.round(proj(c)).astype(int)), tuple(np.round(proj(c + 6 * n)).astype(int)), (0, 255, 0), 2)
    cv2.putText(ov, f"{name}: tip {h:.2f} mm above the site, lateral {lat:.2f} mm (vision, fused {fused_n})", (10, 30), 0, 0.8, (255, 255, 255), 2)
    cv2.imwrite(f"{out}/{name}/overlay.png", ov)
    log(dict(ev="measure", name=name, p=p, raw=raw, height_mm=h, lateral_mm=lat, fused_n=fused_n, jaw=float(io.get("jaw")[1][0]), jaw_eff=float(io.get("jaw_eff")[1][0])))
    print(f"[{name}] tip {h:6.2f} mm above the site, lateral {lat:5.2f} mm (raw h {float((raw - c) @ n):6.2f}, fused {fused_n}) | jaw eff {float(io.get('jaw_eff')[1][0]):.3f}", flush=True)
    return h, lat

def measure(name):
    raw, m = raw_measure(name)
    if raw is None: return None
    if st["hist"]:
        pred = np.mean([r - cm for r, cm in st["hist"][-a.k:]], 0) + st["cum"]
        if np.linalg.norm(raw - pred) > a.outlier_mm:
            log(dict(ev="outlier", name=name, raw=raw, pred=pred)); print(f"[{name}] outlier: raw {np.linalg.norm(raw - pred):.2f} mm from the prediction", flush=True); return "outlier"
    st["hist"].append((raw, st["cum"].copy())); H = st["hist"][-a.k:]
    p = np.mean([r - cm for r, cm in H], 0) + st["cum"]
    st["roi"] = np.mean(np.array(m["poly"]), 0); st["jaw"] = np.array(m["poly"][0])
    mv.tip_off_tool = mv.setpoint()[:3, :3].T @ (cam2ecm_mm(p, M) * 1e-3 - mv.setpoint()[:3, 3]) - TCP_MM * 1e-3
    h, lat = report(name, p, raw, m, len(H)); return p, h, lat

def move(e, tag):
    e = a.gain * np.asarray(e, float); kind = "descend" if float(e @ n) < -0.3 else "translate"
    _, base, _ = io.samples(0.5); mv.translate(dcam2ecm(e), tag=tag, lim=(1.5, 3.0)); time.sleep(0.3); _, eff, err = io.samples(0.3)
    ref = st["desc_eff"]; dd = None if (kind != "descend" or ref is None) else float(eff[2] - ref)
    log(dict(ev="move", kind=kind, cmd_cam=e, d_ins=float((eff - base)[2]), eff_ins=float(eff[2]), vs_prev_descent=dd, trk_ins_mm=float(err[2] * 1e3)))
    st["desc_eff"] = float(eff[2]) if kind == "descend" else None
    st["cum"] = st["cum"] + e; st["roi"] = st["roi"] + (np.array(proj(c + e)) - np.array(proj(c)))
    print(f"   move {np.linalg.norm(e):.2f} mm ({tag}{'' if dd is None else f', descent effort change {dd:+.3f}'})", flush=True)
    if dd is not None and abs(dd) > 0.30: raise Guard(f"contact? insertion effort {dd:+.3f} vs the previous descent step")

status = "dry run: nothing sent"
try:
    r = None
    for _ in range(3):
        r = measure("m00")
        if isinstance(r, tuple): break
    if not isinstance(r, tuple) and a.init_p is not None:
        p0 = np.array(a.init_p); e0 = p0 - c; h0 = float(e0 @ n); r = (p0, h0, float(np.linalg.norm(e0 - h0 * n))); print(f"   seed used: tip {h0:.2f} mm above the site", flush=True)
    if not isinstance(r, tuple): raise Guard("no initial measurement")
    for k in range(1, 3): r2 = measure(f"m00_{k}"); r = r2 if isinstance(r2, tuple) else r       # fill the filter at rest
    if a.send:
        si, bad = 0, 0
        for it in range(1, a.max_iters + 1):
            p, h, lat = r; h_t = a.stages[si]
            if si == len(a.stages) - 1 and h < 0.3: raise Guard(f"tip {h:.2f} mm above the site plane < 0.3")
            e = (c + h_t * n) - p; en = float(np.linalg.norm(e)); last = si == len(a.stages) - 1
            if en <= (a.tol_mm if last else a.tol_stage):
                if not last: si += 1; print(f"   -> target height {a.stages[si]:.1f} mm", flush=True); continue
                status = f"DONE (vision, fused): tip {h:.2f} mm above the site along its normal, lateral {lat:.2f} mm; holding"; break
            cap = 0.5 if last else 1.0; e = e if en <= cap else e * cap / en
            move(e, f"v7_to_h{h_t:g}")
            r2 = measure(f"m{it:02d}")
            pp = np.mean([x - cm for x, cm in st["hist"][-a.k:]], 0) + st["cum"]; hp = float((pp - c) @ n)
            if not isinstance(r2, tuple) and hp <= a.blind_below:
                e0 = pp - c; r2 = (pp, hp, float(np.linalg.norm(e0 - hp * n))); log(dict(ev="predicted", name=f"m{it:02d}", p=pp, height_mm=hp))
                print(f"[m{it:02d}] predicted (no valid reading near the sticker): tip {hp:.2f} mm above the site, lateral {r2[2]:.2f} mm", flush=True)
            while not isinstance(r2, tuple):
                bad += 1
                if bad >= 3: raise Guard("3 measurements in a row failed / did not follow the robot")
                r2 = measure(f"m{it:02d}_{bad}")
            bad = 0; r = r2
        else:
            status = "max iterations"
except Guard as g:
    status = f"STOPPED by guard: {g}"; log(dict(ev="guard", why=str(g)))
except BaseException as ex:
    import traceback; traceback.print_exc(); status = f"STOPPED: {type(ex).__name__} {ex}"
print("\n" + status + "\nout:", out, flush=True); log(dict(ev="end", status=status)); sys.stdout.flush(); os._exit(0)
