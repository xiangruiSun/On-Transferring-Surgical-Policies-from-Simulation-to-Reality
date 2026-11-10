#!/usr/bin/env python3
"""Placing scene (revised definition, 2026-10-30), no robot command.  Snapshot + FS + YOLO stickers ->
entry e (sticker nearer the needle) / exit x, tissue normal n at e (FS plane within 4 mm), bite direction u (x - e projected on the
tangent plane); pad.json re-made from this FS (phantom moved); target needle circle for tip height h:
tip = e + h n, tip tangent -n, centre = tip + R u, needle plane normal m = n x u (arc rises from the tip toward the exit side).

    python3 scene_v8.py --name scene2 --needle-uv 260 700
"""
import argparse, json, os, sys, time
import numpy as np
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code"); sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261027/place/code"))
import cv2
from rig import RigIO, FX, CX, CY, FB, tip_of
import calib as C
from calib import cam2ecm_mm, rotation_from_affine, proj
import place_vision as pv

ap = argparse.ArgumentParser(); ap.add_argument("--name", default="scene2"); ap.add_argument("--needle-uv", type=float, nargs=2, required=True)
ap.add_argument("--snap", default=None); a = ap.parse_args()
d = f"{R0}/G_place/{a.name}"; os.makedirs(d, exist_ok=True)
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); Rc = rotation_from_affine(M)
if a.snap is None:
    io = RigIO(images=True); time.sleep(2.5); C.save_snap(io, d + "/snap"); C.run_fs([d + "/snap"]); snap = d + "/snap"
else: snap = a.snap
img = cv2.imread(snap + "/right_rect.png"); disp = np.load(snap + "/fs_disp.npy").astype(np.float32); Zm = FB / np.clip(disp, 1, None)
P = np.dstack([(np.mgrid[0:disp.shape[0], 0:disp.shape[1]][1] - CX) * Zm / FX, (np.mgrid[0:disp.shape[0], 0:disp.shape[1]][0] - CY) * Zm / FX, Zm])
from remote import Remote, deploy
deploy(); yo = Remote("yolo", d); ym, _ = yo.call({"imgsz": 960, "conf": 0.05}, {"image": img}); yo.close()
# 13:3x: YOLO missed the small tilted sticker -> stickers = the two largest blue blobs (tape), YOLO logged only
labb = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); blue = (labb[..., 2] < 120).astype(np.uint8) * 255   # LAB b < 120: the tilted sticker is washed out (S 33)
blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); nb, lb, stb, _ = cv2.connectedComponentsWithStats(blue)
order = sorted(range(1, nb), key=lambda k: -stb[k, 4])[:2]
boxes = [[float(stb[k, 0]), float(stb[k, 1]), float(stb[k, 0] + stb[k, 2]), float(stb[k, 1] + stb[k, 3]), 1.0] for k in order if stb[k, 4] > 1500]
print("YOLO", [[round(x, 2) for x in bb[:5]] for bb in ym["boxes"][:3]], "| blue blobs", [[round(x) for x in bb[:4]] + [int(stb[k, 4])] for bb, k in zip(boxes, order)])
yb = [bb for bb in ym["boxes"] if bb[4] >= 0.3][:2]
if len(yb) == 2: boxes = yb; print("using the YOLO boxes")                                   # 13:4x: a blue object at the image edge beat the tilted sticker
if len(boxes) < 2: raise SystemExit("need 2 blue stickers")
def site(b):
    x0, y0, x1, y1 = [int(v) for v in b[:4]]; sub = labb[y0:y1, x0:x1, 2] < 128                            # 13:4x: blue pixels inside the box (site_from_frame -> NaN on the thin tilted sticker)
    ys, xs = np.nonzero(sub); uv0 = (x0 + float(xs.mean()), y0 + float(ys.mean())) if len(xs) > 50 else ((x0 + x1) / 2, (y0 + y1) / 2)
    c_st = np.median(P[int(uv0[1]) - 4:int(uv0[1]) + 5, int(uv0[0]) - 4:int(uv0[0]) + 5].reshape(-1, 3), 0); rp = int(4.0 * FX / c_st[2]) + 3; u0, v0 = int(uv0[0]), int(uv0[1])
    Pp = P[v0 - rp:v0 + rp + 1, u0 - rp:u0 + rp + 1].reshape(-1, 3); Pp = Pp[np.linalg.norm(Pp - c_st, axis=1) <= 4.0]; kp = np.ones(len(Pp), bool)
    for _ in range(5):
        q0 = Pp[kp].mean(0); nc = np.linalg.svd(Pp[kp] - q0)[2][2]; rs = (Pp - q0) @ nc; kp = np.abs(rs) < 2.5 * np.std(rs[kp]) + 1e-6
    nc = nc if nc[2] < 0 else -nc; ray = np.array([(uv0[0] - CX) / FX, (uv0[1] - CY) / FX, 1.0]); cc = ray * ((q0 @ nc) / (ray @ nc))
    return dict(centre=cc, normal=nc, box=b, rms=float(np.std(rs[kp])))
S = [site(b) for b in boxes]
nuv = np.array(a.needle_uv); dist = [np.linalg.norm(np.array(proj(s["centre"])) - nuv) for s in S]
ent, ex = (S[0], S[1]) if dist[0] <= dist[1] else (S[1], S[0])
e, n, x = ent["centre"], ent["normal"], ex["centre"]
u = (x - e) - ((x - e) @ n) * n; bite = float(np.linalg.norm(u)); u /= bite; m = np.cross(n, u)
print(f"entry {np.round(e, 1).tolist()} exit {np.round(x, 1).tolist()} bite {bite:.1f} mm; n {np.round(n, 3).tolist()} ({np.degrees(np.arccos(-n[2])):.0f} deg from the view axis); "
      f"needle plane normal m {np.round(m, 3).tolist()} ({np.degrees(np.arccos(abs(m[2]))):.0f} deg from the view axis: 90 = edge-on)")
# pad.json from this FS (as session_driver.fs_pad, but on the fresh snapshot): RANSAC plane on pad pixels >= 40 px from anything else
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32); pink = (lab[..., 1] > 140) & (lab[..., 0] > 90)
clear = cv2.distanceTransform(pink.astype(np.uint8), cv2.DIST_L2, 5) >= 40
Q = P[clear][::25]; rng = np.random.default_rng(0); best = None
for _ in range(500):
    s3 = Q[rng.choice(len(Q), 3, replace=False)]; nn = np.cross(s3[1] - s3[0], s3[2] - s3[0])
    if np.linalg.norm(nn) < 1e-9: continue
    nn /= np.linalg.norm(nn); k = int((np.abs((Q - s3[0]) @ nn) < 0.5).sum())
    if best is None or k > best[0]: best = (k, nn, s3[0])
I = Q[np.abs((Q - best[2]) @ best[1]) < 0.5]; E = np.array([cam2ecm_mm(p, M) for p in I]) * 1e-3; c0 = E.mean(0); nE = np.linalg.svd(E - c0)[2][2]
tip0 = tip_of(np.array(cal["base_T"]))
if (tip0 - c0) @ nE < 0: nE = -nE
resid = np.abs((E - c0) @ nE) * 1e3; dd = -float(nE @ c0) - 2.0e-3
pad = dict(normal=nE.tolist(), d=dd, n_points=int(len(E)), inlier_frac=float(len(I) / len(Q)), rms_mm=float(np.sqrt((resid ** 2).mean())), margin_mm=2.0,
           source=f"FS plane of {snap} (phantom moved 2026-10-30 ~13:3x) mapped with C_r6cal; margin 2 mm")
pp = R0 + "/D_pad/pad.json"
if os.path.exists(pp): os.rename(pp, pp.replace(".json", time.strftime("_superseded_%H%M%S.json")))
json.dump(pad, open(pp, "w"), indent=1); print(f"pad.json: {len(E)} pts, rms {pad['rms_mm']:.2f} mm, inlier frac {pad['inlier_frac']:.2f}")
out = dict(entry=e.tolist(), exit=x.tolist(), normal=n.tolist(), u=u.tolist(), m=m.tolist(), bite_mm=bite, entry_rms=ent["rms"], exit_rms=ex["rms"],
           entry_box=ent["box"], exit_box=ex["box"], snap=snap, needle_uv=a.needle_uv)
json.dump(out, open(d + "/scene.json", "w"), indent=1)
ov = img.copy()
for s, col in ((ent, (0, 255, 0)), (ex, (0, 0, 255))): cv2.circle(ov, tuple(np.int32(proj(s["centre"]))), 9, col, 2)
cv2.line(ov, tuple(np.int32(proj(e))), tuple(np.int32(proj(e + 6 * n))), (0, 255, 0), 2); cv2.line(ov, tuple(np.int32(proj(e))), tuple(np.int32(proj(e + 6 * u))), (255, 0, 0), 2)
cv2.imwrite(d + "/scene.png", ov); print("->", d + "/scene.json", flush=True); sys.stdout.flush(); os._exit(0)
