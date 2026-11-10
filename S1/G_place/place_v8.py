#!/usr/bin/env python3
"""Placing v8 (revised definition 2026-10-30): needle plane = span(n, u) (vertical, through entry -> exit), tip lowest at the entry
pointing into the tissue (-n), arc rising toward the exit side.  Scene from scene_v8.py.

Orientation: today's wrist FK is wrong (~70 deg), so the WRIST JOINTS (roll, pitch, yaw) are servoed directly on 2D image features of
the held needle (held_needle arc, grasp end -> tip): tip-tangent angle, chord angle, signed bulge.  Targets = the same features of the
target needle arc projected into the right image.  Jacobian: finite-difference probes (+--probe deg per joint), then Broyden.
Position: tip kept / brought to e + h n by camera-frame translations (calibration A3, gain 0.75); final descent h_pre -> --hover-mm
(translation only).  Guards: Mover (translations), measured tip height >= --min-h during rotations (else lift along n), jaw effort logged.

    python3 place_v8.py --scene scene2/scene.json --roi 260 700 --jaw 120 900 [--send]
"""
import argparse, json, os, sys, time
import numpy as np
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code"); sys.path.insert(0, R0 + "/G_place")
import cv2
from sensor_msgs.msg import JointState
from rig import RigIO, Mover, load_kin, Guard, TCP_MM, FX, CX, CY
from calib import proj, cam2ecm_mm
import tipmeas

ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--send", action="store_true"); ap.add_argument("--out", default=None)
ap.add_argument("--roi", type=float, nargs=2, required=True); ap.add_argument("--jaw", type=float, nargs=2, required=True); ap.add_argument("--roi-px", type=int, default=320)
ap.add_argument("--h-pre", type=float, default=12.0); ap.add_argument("--hover-mm", type=float, default=1.5); ap.add_argument("--min-h", type=float, default=5.0)
ap.add_argument("--radius", type=float, default=9.7); ap.add_argument("--span", type=float, default=130.0); ap.add_argument("--probe", type=float, default=6.0)
ap.add_argument("--max-step", type=float, default=8.0); ap.add_argument("--tol", type=float, default=0.12); ap.add_argument("--iters", type=int, default=25)
ap.add_argument("--gain", type=float, default=0.75); ap.add_argument("--n-cap", type=int, default=2); ap.add_argument("--skip-orient", action="store_true"); ap.add_argument("--only-pre", action="store_true")
a = ap.parse_args()
G = R0 + "/G_place"; sc = json.load(open(a.scene if os.path.isabs(a.scene) else G + "/" + a.scene))
out = a.out or G + "/v8_" + time.strftime("%H%M%S"); os.makedirs(out, exist_ok=True); logf = open(out + "/v8_log.jsonl", "a")
def log(r): logf.write(json.dumps(dict(t=time.time(), **r), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"); logf.flush()
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); A3 = M[:3]; pad = json.load(open(R0 + "/D_pad/pad.json"))
e, n, u = np.array(sc["entry"]), np.array(sc["normal"]), np.array(sc["u"])
dcam2ecm = lambda dX: np.asarray(dX) @ np.linalg.inv(A3) * 1e-3
io = RigIO(images=True); kin = load_kin(); time.sleep(2.5); mv = Mover(io, kin, log)
mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = np.array(cal["frozen"]["suj_psm1"]); mv.pad = (np.array(pad["normal"]), pad["d"]); mv.cube = None; mv.needle = cam2ecm_mm(e, M) * 1e-3
mv.tip_off_tool = np.zeros(3)
pubj = io.node.create_publisher(JointState, '/PSM1/move_jp', 1)
st = dict(roi=np.array(a.roi), jaw=np.array(a.jaw))
log(dict(ev="start", args=vars(a), scene=sc))

def target_poly(h, n_pts=13):
    """target arc (grasp end -> tip) in the right image for tip height h."""
    c = e + h * n + a.radius * u; th = np.radians(np.linspace(a.span, 0, n_pts))
    P = c[None] + a.radius * (-np.cos(th)[:, None] * u[None] + np.sin(th)[:, None] * n[None])
    return np.array([proj(p) for p in P]), P

def feats(poly):
    poly = np.asarray(poly, float); t = poly[-1] - poly[-3]; ch = poly[-1] - poly[0]; L = np.linalg.norm(ch)
    nrm = np.array([-ch[1], ch[0]]) / L; bulge = float(np.max((poly - poly[0]) @ nrm) + np.min((poly - poly[0]) @ nrm)) / L
    return np.array([np.arctan2(t[1], t[0]), np.arctan2(ch[1], ch[0]), bulge])

def fdiff(f1, f0):
    d = f1 - f0; d[:2] = (d[:2] + np.pi) % (2 * np.pi) - np.pi; return d

def measure(name):
    d = f"{out}/{name}"; os.makedirs(d, exist_ok=True); ps, polys = [], []
    for k in range(a.n_cap):
        b = io.frames(8); R = np.mean([x[1] for x in b], 0).round().astype(np.uint8); L = np.mean([x[0] for x in b], 0).round().astype(np.uint8)
        if k == 0: cv2.imwrite(d + "/right_rect.png", R); cv2.imwrite(d + "/left_rect.png", L)
        m, why = tipmeas.measure(R, L, e[2], roi_uv=st["roi"], roi_px=a.roi_px, jaw_px=st["jaw"])
        if m is None: log(dict(ev="measure_fail", name=name, why=why)); continue
        ps.append(m["tip_cam"]); polys.append(np.array(m["poly"]))
    if not ps: print(f"[{name}] no needle", flush=True); return None
    p = np.median(ps, 0); poly = polys[0]; st["roi"] = poly.mean(0); st["jaw"] = poly[0]
    eh = p - e; h = float(eh @ n); lat = float(np.linalg.norm(eh - h * n)); f = feats(poly)
    tp, _ = target_poly(max(h, a.hover_mm)); ft = feats(tp); err = fdiff(ft, f)
    ov = cv2.imread(d + "/right_rect.png"); cv2.polylines(ov, [np.int32(poly)], False, (0, 255, 255), 2); cv2.polylines(ov, [np.int32(tp)], False, (0, 255, 0), 2)
    cv2.circle(ov, tuple(np.int32(proj(p))), 8, (255, 0, 255), 2); cv2.circle(ov, tuple(np.int32(proj(e))), 7, (0, 0, 255), 2)
    cv2.putText(ov, f"{name}: tip h {h:.1f} lat {lat:.1f} mm | feature err tip {np.degrees(err[0]):.0f} chord {np.degrees(err[1]):.0f} deg bulge {err[2]:+.2f}", (10, 30), 0, 0.75, (255, 255, 255), 2)
    cv2.imwrite(d + "/overlay.png", ov)
    r = dict(name=name, p=p, h=h, lat=lat, f=f, ft=ft, err=err, q=np.array(io.get("js")[1][:6]), jaw_eff=float(io.get("jaw_eff")[1][0]))
    log(dict(ev="measure", **r)); print(f"[{name}] tip h {h:5.1f} lat {lat:5.1f} mm | err tip {np.degrees(err[0]):6.1f} chord {np.degrees(err[1]):6.1f} deg bulge {err[2]:+.2f} | jaw eff {r['jaw_eff']:.3f}", flush=True)
    return r

def move_wrist(dq):
    q = np.array(io.get("spj")[1][:6]) if io.get("spj") else np.array(io.get("js")[1][:6]); q2 = q.copy(); q2[3:6] += dq
    m = JointState(); m.position = q2.tolist(); pubj.publish(m); time.sleep(1.5 + 2.0 * float(np.max(np.abs(dq)))); log(dict(ev="wrist", dq_deg=np.degrees(dq), q=q2))

def translate(dX, tag, mode=False):
    dX = a.gain * np.asarray(dX, float); mv.translate(dcam2ecm(dX), tag=tag, contact_ok=mode, lim=(1.5, 3.0)); st["roi"] = st["roi"] + (np.array(proj(e + dX)) - np.array(proj(e)))
    log(dict(ev="translate", dX=dX, tag=tag)); print(f"   translate {np.linalg.norm(dX):.1f} mm ({tag})", flush=True)

def to_height(r, h_t, cap=3.0, tag="pos"):
    dX = (e + h_t * n) - np.asarray(r["p"]); nn = np.linalg.norm(dX); dX = dX if nn <= cap else dX * cap / nn
    translate(dX, tag, mode="away" if (dX @ n) > 0 else False)

W = np.array([1.0, 1.0, 2.0])     # feature weights (rad, rad, bulge)
status = "dry run: nothing sent"
try:
    r = measure("m00")
    if r is None: raise Guard("no initial needle measurement")
    if a.send:
        k = 0
        for _ in range(8):                                                                # 1. tip to e + h_pre n
            if abs(r["h"] - a.h_pre) < 2.0 and r["lat"] < 3.0: break
            to_height(r, a.h_pre, tag="v8_pre"); k += 1; r = measure(f"p{k:02d}") or r
        if a.only_pre: raise Guard(f"only-pre: tip {r['h']:.1f} mm above the entry, lateral {r['lat']:.1f} mm")
        if not a.skip_orient:                                                             # 2. orientation by wrist joints
            J = np.zeros((3, 3)); base = r
            for j in range(3):
                dq = np.zeros(3); dq[j] = np.radians(a.probe); move_wrist(dq); rj = measure(f"probe{j}")
                move_wrist(-dq)
                if rj is None: raise Guard(f"probe {j}: needle lost")
                J[:, j] = fdiff(rj["f"], base["f"]) / np.radians(a.probe)
            r = measure("after_probe") or base; log(dict(ev="J", J=J)); print("J (feature per rad)\n", np.round(J, 3), flush=True)
            prev = None
            for it in range(1, a.iters + 1):
                e_w = W * r["err"]; en = float(np.linalg.norm(e_w))
                if en < a.tol: print(f"   orientation converged ({en:.3f})", flush=True); break
                dq = np.linalg.lstsq(W[:, None] * J, e_w, rcond=1e-3)[0]; mx = np.radians(a.max_step)
                if np.max(np.abs(dq)) > mx: dq *= mx / np.max(np.abs(dq))
                move_wrist(dq); r2 = measure(f"o{it:02d}")
                if r2 is None: move_wrist(-dq); raise Guard("needle lost after a wrist step (undone)")
                J = J + np.outer(fdiff(r2["f"], r["f"]) - J @ dq, dq) / (dq @ dq)
                print(f"   wrist {np.round(np.degrees(dq), 1).tolist()} deg -> |err| {np.linalg.norm(W * r2['err']):.3f}", flush=True)
                r = r2
                if r["h"] < a.min_h: to_height(r, a.h_pre, tag="v8_lift"); r = measure(f"o{it:02d}_l") or r
                elif r["lat"] > 4.0: to_height(r, a.h_pre, tag="v8_recentre"); r = measure(f"o{it:02d}_c") or r
        for st_h, cap in ((5.0, 1.0), (a.hover_mm, 0.5)):                               # 3. translation-only descent
            for k2 in range(20):
                dX = (e + st_h * n) - np.asarray(r["p"])
                if np.linalg.norm(dX) <= (0.6 if st_h == a.hover_mm else 1.0): break
                to_height(r, st_h, cap=cap, tag=f"v8_to_h{st_h:g}"); r2 = measure(f"d{st_h:g}_{k2:02d}")
                if r2 is None: print("   no reading near the site -> stop descending here", flush=True); break
                r = r2
        status = f"END: tip {r['h']:.2f} mm above the entry along n, lateral {r['lat']:.2f} mm; feature err tip {np.degrees(r['err'][0]):.0f} chord {np.degrees(r['err'][1]):.0f} deg bulge {r['err'][2]:+.2f}; holding"
except Guard as g:
    status = f"STOPPED by guard: {g}"; log(dict(ev="guard", why=str(g)))
except BaseException as ex:
    import traceback; traceback.print_exc(); status = f"STOPPED: {type(ex).__name__} {ex}"
print("\n" + status + "\nout:", out, flush=True); log(dict(ev="end", status=status)); sys.stdout.flush(); os._exit(0)
