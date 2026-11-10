#!/usr/bin/env python3
"""Placing plan for a needle the USER put into the jaw (2026-10-30: no grasp record -> no grasp geometry).  No robot command.

needle-in-jaw: measured now with tipmeas (right-image held-needle arc, left NCC, quadratic tangent) on --n-meas captures at the
  current (static) pose; tip / tangent mapped to ECM with the calibration and to the tool frame with the current setpoint; median.
site: as plan_place.py (YOLO sticker box -> place_vision.site_from_frame, then FS point-cloud plane within 4 mm, centre = ray x plane).
spins: as plan_place.py (tip at c + h n, tangent -n, IK with margins).  Output keys = plan_place.py's, so place_v4.py reads it.

    python3 plan_manual.py --name manual1 [--hover-mm 1]
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.spatial.transform import Rotation
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1")
sys.path.insert(0, R0 + "/code"); sys.path.insert(0, R0 + "/G_place"); sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261027/place/code"))
import cv2
from rig import RigIO, load_kin, TCP_MM, FX, CX, CY, FB
import calib as C
from calib import rotation_from_affine, cam2ecm_mm, proj
import place_vision as pv
import tipmeas

ap = argparse.ArgumentParser(); ap.add_argument("--name", default="manual1"); ap.add_argument("--hover-mm", type=float, default=1.0)
ap.add_argument("--above-mm", type=float, default=12.0); ap.add_argument("--n-meas", type=int, default=5)
a = ap.parse_args()
d = f"{R0}/G_place/{a.name}"; os.makedirs(d, exist_ok=True); out = f"{R0}/G_place/plan_{a.name}.json"
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); Rc = rotation_from_affine(M)
pad = json.load(open(R0 + "/D_pad/pad.json")); pn, pdd = np.array(pad["normal"]), pad["d"]; padmm = lambda p: float((pn @ p + pdd) * 1e3)
kin = load_kin(); suj = np.array(cal["frozen"]["suj_psm1"])
c2e = lambda X: cam2ecm_mm(X, M) * 1e-3
dir_c2e = lambda v: (Rc.T @ v) / np.linalg.norm(Rc.T @ v)

io = RigIO(images=True); time.sleep(2.5)
snap = d + "/snap0"; C.save_snap(io, snap); C.run_fs([snap])
H = np.array(io.get("sp")[1]); js = np.array(io.get("js")[1])[:6]
img = cv2.imread(snap + "/right_rect.png"); disp = np.load(snap + "/fs_disp.npy").astype(np.float32)

# --- site (plan_place.py)
from remote import Remote, deploy
deploy(); yo = Remote("yolo", d); ym, _ = yo.call({"imgsz": 960, "conf": 0.05}, {"image": img}); yo.close(); boxes = ym["boxes"]
json.dump(dict(boxes=boxes), open(d + "/yolo.json", "w"))
print("YOLO boxes (x0,y0,x1,y1,conf):", [[round(x, 2) for x in b[:5]] for b in boxes[:3]])
site, why = pv.site_from_frame(img, disp, boxes[0])
if site is None: raise SystemExit("site: " + why)
c_st, n_st = np.array(site["centre"]), np.array(site["normal"])
uv0 = proj(c_st); rp = int(4.0 * FX / c_st[2]) + 3; u0, v0 = int(uv0[0]), int(uv0[1])
sub = disp[v0 - rp:v0 + rp + 1, u0 - rp:u0 + rp + 1]; yy, xx = np.mgrid[v0 - rp:v0 + rp + 1, u0 - rp:u0 + rp + 1]; Zs = FB / np.clip(sub, 1, None)
Pp = np.dstack([(xx - CX) * Zs / FX, (yy - CY) * Zs / FX, Zs]).reshape(-1, 3); Pp = Pp[(np.linalg.norm(Pp - c_st, axis=1) <= 4.0) & (sub.reshape(-1) > 1)]
kp = np.ones(len(Pp), bool)
for _ in range(5):
    q0 = Pp[kp].mean(0); n_c = np.linalg.svd(Pp[kp] - q0)[2][2]; rs = (Pp - q0) @ n_c; kp = np.abs(rs) < 2.5 * np.std(rs[kp]) + 1e-6
n_c = n_c if n_c[2] < 0 else -n_c
ray = np.array([(uv0[0] - CX) / FX, (uv0[1] - CY) / FX, 1.0]); c_c = ray * ((q0 @ n_c) / (ray @ n_c))
site.update(centre=c_c, normal=n_c, plane_rms_mm=float(np.std(rs[kp])), n_points=int(kp.sum()), sticker_centre=c_st.tolist(), sticker_normal=n_st.tolist(),
            method="FS point cloud, robust plane within 4 mm (user 10-28 17:4x)")
print(f"FS-normal vs sticker-normal {np.degrees(np.arccos(np.clip(n_c @ n_st, -1, 1))):.1f} deg, centre shift {np.linalg.norm(c_c - c_st):.2f} mm")
print(f"site centre (cam) {np.round(c_c, 1).tolist()} mm, normal {np.round(n_c, 3).tolist()}, plane rms {site['plane_rms_mm']:.2f} mm, {site['n_points']} pts")
c_e = c2e(c_c); n_e = dir_c2e(n_c)
print(f"site in ECM {np.round(c_e * 1e3, 1).tolist()} mm; height above pad.json {padmm(c_e):.1f} mm; normal vs pad normal {np.degrees(np.arccos(abs(n_e @ pn))):.0f} deg")

# --- needle in the jaw, measured now (arm static)
jaw_px = np.asarray(proj(np.r_[(H[:3, 3] + H[:3, :3] @ (TCP_MM * 1e-3)) * 1e3, 1] @ M))
tips, tans, rows = [], [], []
for k in range(a.n_meas):
    b = io.frames(10); R = np.mean([x[1] for x in b], 0).round().astype(np.uint8); L = np.mean([x[0] for x in b], 0).round().astype(np.uint8)
    m, why = tipmeas.measure(R, L, c_c[2], roi_uv=proj(c_c), roi_px=320, jaw_px=jaw_px)
    if m is None: print(f"  meas {k}: {why}"); rows.append(dict(k=k, why=why)); continue
    tip_e = c2e(m["tip_cam"]); tan_e = dir_c2e(m["dir_cam"])
    tips.append(H[:3, :3].T @ (tip_e - H[:3, 3])); tans.append(H[:3, :3].T @ tan_e)
    rows.append(dict(k=k, tip_cam=m["tip_cam"], dir_cam=m["dir_cam"], ncc=m["tip"]["ncc"], fit=m["fit"]))
    print(f"  meas {k}: tip cam {np.round(m['tip_cam'], 2).tolist()} dir {np.round(m['dir_cam'], 3).tolist()} ncc {m['tip']['ncc']:.2f} rms {m['fit']['rms_mm']:.2f}")
    if k == 0:
        ov = R.copy(); cv2.polylines(ov, [np.round(m["poly"]).astype(np.int32)], False, (0, 255, 255), 2)
        cv2.circle(ov, tuple(np.round(m["tip"]["uv"]).astype(int)), 8, (255, 0, 255), 2); cv2.circle(ov, tuple(np.round(proj(c_c)).astype(int)), 8, (0, 255, 0), 2)
        cv2.circle(ov, tuple(np.round(jaw_px).astype(int)), 8, (255, 0, 0), 2); cv2.imwrite(d + "/needle_meas.png", ov)
if len(tips) < 3: raise SystemExit(f"needle tip measured in only {len(tips)}/{a.n_meas} captures")
tip_tool = np.median(tips, 0); tan_tool = np.median(tans, 0); tan_tool /= np.linalg.norm(tan_tool)
spread_mm = float(np.max(np.linalg.norm(np.array(tips) - tip_tool, axis=1)) * 1e3)
spread_deg = float(np.max(np.degrees(np.arccos(np.clip(np.array(tans) @ tan_tool, -1, 1)))))
tip_now = H[:3, :3] @ tip_tool + H[:3, 3]
print(f"needle tip in tool frame {np.round(tip_tool * 1e3, 2).tolist()} mm (spread {spread_mm:.2f} mm), tangent {np.round(tan_tool, 3).tolist()} (spread {spread_deg:.1f} deg)")
print(f"needle tip now {np.round(tip_now * 1e3, 1).tolist()} mm ECM; distance to the site {np.linalg.norm(tip_now - c_e) * 1e3:.1f} mm; "
      f"tip direction vs -n {np.degrees(np.arccos(np.clip((H[:3, :3] @ tan_tool) @ -n_e, -1, 1))):.0f} deg")

# --- spins (plan_place.py)
def rot_between(x, y):
    w = np.cross(x, y); s, cth = np.linalg.norm(w), x @ y
    return np.eye(3) if s < 1e-9 else Rotation.from_rotvec(w / s * np.arctan2(s, cth)).as_matrix()
want = -n_e; R_align = rot_between(H[:3, :3] @ tan_tool, want) @ H[:3, :3]
spins = []
for phi in range(-180, 180, 10):
    Rg = Rotation.from_rotvec(np.radians(phi) * want).as_matrix() @ R_align; res = {}
    for name, h in (("final", a.hover_mm), ("above", a.above_mm)):
        tipg = c_e + n_e * h * 1e-3; hinge = tipg - Rg @ tip_tool
        ok, q, why = kin.check_pose(hinge, Rg, js, suj, min_deg=3.0, min_mm=5.0); jaw_tip = hinge + Rg @ (TCP_MM * 1e-3)
        res[name] = dict(ok=bool(ok), why=why, hinge_pad_mm=padmm(hinge), jaw_tcp_pad_mm=padmm(jaw_tip), margins=np.round(kin.margins(q), 1).tolist() if ok else None)
    spins.append(dict(phi=phi, **res))
feas = [r for r in spins if r["final"]["ok"] and r["above"]["ok"]]
print(f"\nspins with IK-feasible final ({a.hover_mm} mm) AND above ({a.above_mm} mm): {len(feas)}/{len(spins)}")
for r in feas: print(f"  phi {r['phi']:+4d}: final hinge {r['final']['hinge_pad_mm']:6.1f} / jaw TCP {r['final']['jaw_tcp_pad_mm']:6.1f} mm above pad.json; margins {r['final']['margins']}")
json.dump(dict(attempt=a.name, source="manual grasp by the user, needle measured in the jaw (tipmeas)", hover_mm=a.hover_mm, tip_tool_mm=(tip_tool * 1e3).tolist(),
               tan_tool=tan_tool.tolist(), needle_meas=dict(n=len(tips), spread_mm=spread_mm, spread_deg=spread_deg, rows=rows, sp=H.tolist(), js=js.tolist()),
               site_cam=dict(centre=c_c.tolist(), normal=n_c.tolist(), radius_mm=site["radius_mm"], rms=site["plane_rms_mm"], box=site["box"], method=site["method"],
                             sticker_normal=site["sticker_normal"]), site_ecm_mm=(c_e * 1e3).tolist(), normal_ecm=n_e.tolist(), spins=spins),
          open(out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("->", out); sys.stdout.flush(); os._exit(0)
