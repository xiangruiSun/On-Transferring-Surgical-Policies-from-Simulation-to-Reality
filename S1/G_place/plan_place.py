#!/usr/bin/env python3
"""Offline placing plan (no robot command): needle-in-jaw from the grasp geometry, site from YOLO + FS, IK of the target.

needle-in-jaw: the needle arc fitted before the grasp (perception.json, camera mm, snap0) is rigid with the jaw from the close on:
  p_tool = T_close^-1 p_ecm for the tip and the tip tangent (T_close = setpoint at 'closed', hinge pose in ECM).
site: YOLO sticker box on snap0 (3090) -> place_vision.site_from_frame on the snap0 FS disparity -> centre c, normal n (camera).
camera <-> ECM: today's calibration (C_r6cal M, affine; directions with its orthonormal part).
target (17:2x user): tip at c + h n (h = 2 mm), tip tangent = -n; spin about n free -> IK-feasible spins listed.

    python3 plan_place.py --attempt A41_Prep [--hover-mm 2]
"""
import argparse, json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1")
sys.path.insert(0, R0 + "/code"); sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261027/place/code"))
import cv2
from rig import load_kin, tip_of, TCP_MM
from calib import rotation_from_affine, cam2ecm_mm, proj
import place_vision as pv

ap = argparse.ArgumentParser(); ap.add_argument("--attempt", default="A41_Prep"); ap.add_argument("--hover-mm", type=float, default=2.0)
ap.add_argument("--above-mm", type=float, default=15.0); ap.add_argument("--yolo-json", default=None)
a = ap.parse_args()
d = f"{R0}/F_attempts/{a.attempt}/"; out = f"{R0}/G_place/plan_{a.attempt}.json"
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); Rc = rotation_from_affine(M)
pad = json.load(open(R0 + "/D_pad/pad.json")); pn, pdd = np.array(pad["normal"]), pad["d"]; padmm = lambda p: float((pn @ p + pdd) * 1e3)
kin = load_kin(); suj = np.array(cal["frozen"]["suj_psm1"])
c2e = lambda X: cam2ecm_mm(X, M) * 1e-3                      # camera mm -> ECM m
dir_c2e = lambda v: (Rc.T @ v) / np.linalg.norm(Rc.T @ v)     # camera direction -> ECM direction (orthonormal part)

# --- needle geometry at snap0 (camera mm)
P = json.load(open(d + "perception.json")); f = P["frames"][-1]; fit = f["fit"]
C0, R_n, u, v = np.array(fit["centre_mm"]), fit["R_mm"], np.array(fit["_u"]), np.array(fit["_v"])
tip_c = np.array(f["tip_mm"]); tail_c = np.array(fit["end_a_mm"] if f["tail_is_end_a"] else fit["end_b_mm"])
ang = lambda X: np.arctan2((X - C0) @ v, (X - C0) @ u)
pt = lambda t: C0 + R_n * (np.cos(t) * u + np.sin(t) * v)
a_tip, a_tail = ang(tip_c), ang(tail_c)
sgn = 1.0 if np.linalg.norm(pt(a_tip - 0.05) - pt(a_tail)) < np.linalg.norm(pt(a_tip + 0.05) - pt(a_tail)) else -1.0   # sign of 'past the tip'
tan_c = (pt(a_tip + sgn * 0.01) - pt(a_tip)); tan_c /= np.linalg.norm(tan_c)
T_close = np.array(json.load(open(d + "closed/state.json"))["sp"])
tip_e = c2e(tip_c); tan_e = dir_c2e(tan_c)
tip_tool = T_close[:3, :3].T @ (tip_e - T_close[:3, 3]); tan_tool = T_close[:3, :3].T @ tan_e
print(f"needle R {R_n:.2f} mm arc {fit['arc_deg']:.0f} deg; tip in tool frame {np.round(tip_tool * 1e3, 2).tolist()} mm, tip tangent (tool) {np.round(tan_tool, 3).tolist()}")

# --- site from YOLO + FS on snap0
img = cv2.imread(d + "snap0/right_rect.png"); disp = np.load(d + "snap0/fs_disp.npy").astype(np.float32)
if a.yolo_json: boxes = json.load(open(a.yolo_json))["boxes"]
else:
    from remote import Remote, deploy
    deploy(); yo = Remote("yolo", R0 + "/G_place"); ym, _ = yo.call({"imgsz": 960, "conf": 0.05}, {"image": img}); yo.close(); boxes = ym["boxes"]
    json.dump(dict(boxes=boxes), open(R0 + f"/G_place/yolo_{a.attempt}.json", "w"))
print("YOLO boxes (x0,y0,x1,y1,conf):", [[round(x, 2) for x in b[:5]] for b in boxes[:3]])
site, why = pv.site_from_frame(img, disp, boxes[0])
if site is None: raise SystemExit("site: " + why)
c_st, n_st = np.array(site["centre"]), np.array(site["normal"])
# 17:4x user: site normal from the FS point cloud (robust plane within 4 mm of the sticker centre), centre = sticker-centre ray x plane
from rig import FX as _FX, CX as _CX, CY as _CY, FB as _FB
uv0 = proj(c_st); rp = int(4.0 * _FX / c_st[2]) + 3; u0, v0 = int(uv0[0]), int(uv0[1])
sub = disp[v0 - rp:v0 + rp + 1, u0 - rp:u0 + rp + 1]; yy, xx = np.mgrid[v0 - rp:v0 + rp + 1, u0 - rp:u0 + rp + 1]; Zs = _FB / np.clip(sub, 1, None)
Pp = np.dstack([(xx - _CX) * Zs / _FX, (yy - _CY) * Zs / _FX, Zs]).reshape(-1, 3); Pp = Pp[(np.linalg.norm(Pp - c_st, axis=1) <= 4.0) & (sub.reshape(-1) > 1)]
kp = np.ones(len(Pp), bool)
for _ in range(5):
    q0 = Pp[kp].mean(0); n_c = np.linalg.svd(Pp[kp] - q0)[2][2]; rs = (Pp - q0) @ n_c; kp = np.abs(rs) < 2.5 * np.std(rs[kp]) + 1e-6
n_c = n_c if n_c[2] < 0 else -n_c
ray = np.array([(uv0[0] - _CX) / _FX, (uv0[1] - _CY) / _FX, 1.0]); c_c = ray * ((q0 @ n_c) / (ray @ n_c))
site.update(centre=c_c, normal=n_c, plane_rms_mm=float(np.std(rs[kp])), n_points=int(kp.sum()), sticker_centre=c_st.tolist(), sticker_normal=n_st.tolist(),
            method="FS point cloud, robust plane within 4 mm (user 17:4x)")
print(f"FS-normal vs sticker-normal {np.degrees(np.arccos(np.clip(n_c @ n_st, -1, 1))):.1f} deg, centre shift {np.linalg.norm(c_c - c_st):.2f} mm")
print(f"site centre (cam) {np.round(c_c, 1).tolist()} mm, normal {np.round(n_c, 3).tolist()}, sticker r {site['radius_mm']:.1f} mm, plane rms {site['plane_rms_mm']:.2f} mm, {site['n_points']} pts")
c_e = c2e(c_c); n_e = dir_c2e(n_c)
print(f"site in ECM {np.round(c_e * 1e3, 1).tolist()} mm; height above pad.json {padmm(c_e):.1f} mm; normal vs pad normal {np.degrees(np.arccos(abs(n_e @ pn))):.0f} deg")
# current hold pose and the needle now (prediction)
H = np.array(json.load(open(d + "hold/state.json"))["sp"]); js = np.array(json.load(open(d + "hold/state.json"))["js"])[:6]
tip_now = H[:3, :3] @ tip_tool + H[:3, 3]
print(f"needle tip now (pred) {np.round(tip_now * 1e3, 1).tolist()} mm; distance to the site {np.linalg.norm(tip_now - c_e) * 1e3:.1f} mm")

# --- goals: tip at c + h n with tangent -n; spin phi about n
def rot_between(x, y):
    w = np.cross(x, y); s, cth = np.linalg.norm(w), x @ y
    return np.eye(3) if s < 1e-9 else Rotation.from_rotvec(w / s * np.arctan2(s, cth)).as_matrix()
want = -n_e; R_align = rot_between(H[:3, :3] @ tan_tool, want) @ H[:3, :3]
rows = []
for phi in range(-180, 180, 10):
    Rg = Rotation.from_rotvec(np.radians(phi) * want).as_matrix() @ R_align
    res = {}
    for name, h in (("final", a.hover_mm), ("above", a.above_mm)):
        tipg = c_e + n_e * h * 1e-3; hinge = tipg - Rg @ tip_tool
        ok, q, why = kin.check_pose(hinge, Rg, js, suj, min_deg=3.0, min_mm=5.0)
        jaw_tip = hinge + Rg @ (TCP_MM * 1e-3)
        res[name] = dict(ok=bool(ok), why=why, hinge_pad_mm=padmm(hinge), jaw_tcp_pad_mm=padmm(jaw_tip), hinge=hinge.tolist(), R=Rg.tolist(),
                         margins=np.round(kin.margins(q), 1).tolist() if ok and hasattr(kin, "margins") else None)
    rows.append(dict(phi=phi, **res))
feas = [r for r in rows if r["final"]["ok"] and r["above"]["ok"]]
print(f"\nspins with IK-feasible final ({a.hover_mm} mm) AND above ({a.above_mm} mm) poses: {len(feas)}/{len(rows)}")
for r in feas[:36]:
    print(f"  phi {r['phi']:+4d}: final hinge {r['final']['hinge_pad_mm']:6.1f} / jaw TCP {r['final']['jaw_tcp_pad_mm']:6.1f} mm above pad.json; margins {r['final']['margins']}")
if not feas:
    for r in rows[::6]: print(f"  phi {r['phi']:+4d}: final {r['final']['why']} | above {r['above']['why']}")
json.dump(dict(attempt=a.attempt, hover_mm=a.hover_mm, tip_tool_mm=(tip_tool * 1e3).tolist(), tan_tool=tan_tool.tolist(), site_cam=dict(centre=c_c.tolist(), normal=n_c.tolist(),
          radius_mm=site["radius_mm"], rms=site["plane_rms_mm"], box=site["box"], method=site["method"], sticker_normal=site["sticker_normal"]), site_ecm_mm=(c_e * 1e3).tolist(), normal_ecm=n_e.tolist(), spins=rows),
          open(out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("->", out)
