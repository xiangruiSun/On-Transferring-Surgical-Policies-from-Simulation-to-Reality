#!/usr/bin/env python3
"""pad.json only (no robot command): snapshot + FS -> RANSAC plane on pad pixels >= 40 px from anything else, mapped with C_r6cal,
margin 2 mm toward the base tip.  Same pad code as scene_v8.py, without the sticker part (phantom pushed 2026-10-30 ~15:3x).

    python3 pad_from_snap.py --name pad_1530
"""
import argparse, json, os, sys, time
import numpy as np
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code")
import cv2
from rig import RigIO, FX, CX, CY, FB, tip_of
import calib as C
from calib import cam2ecm_mm

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--snap", default=None); a = ap.parse_args()
d = f"{R0}/G_place/{a.name}"; os.makedirs(d, exist_ok=True)
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"])
if a.snap is None:
    io = RigIO(images=True); time.sleep(2.5); C.save_snap(io, d + "/snap"); C.run_fs([d + "/snap"]); snap = d + "/snap"
else: snap = a.snap
img = cv2.imread(snap + "/right_rect.png"); disp = np.load(snap + "/fs_disp.npy").astype(np.float32); Zm = FB / np.clip(disp, 1, None)
P = np.dstack([(np.mgrid[0:disp.shape[0], 0:disp.shape[1]][1] - CX) * Zm / FX, (np.mgrid[0:disp.shape[0], 0:disp.shape[1]][0] - CY) * Zm / FX, Zm])
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
           base_tip_height_mm=float((nE @ tip0 + dd) * 1e3), source=f"FS plane of {snap} (phantom pushed 2026-10-30 ~15:3x) mapped with C_r6cal; margin 2 mm")
mask = np.zeros(img.shape[:2], np.uint8); uv = np.argwhere(clear)[::25][np.abs((Q - best[2]) @ best[1]) < 0.5]
for v, u in uv: cv2.circle(img, (int(u), int(v)), 2, (0, 255, 0), -1)
cv2.imwrite(d + "/pad_inliers.png", img)
pp = R0 + "/D_pad/pad.json"
if os.path.exists(pp): os.rename(pp, pp.replace(".json", time.strftime("_superseded_%H%M%S.json")))
json.dump(pad, open(pp, "w"), indent=1); json.dump(pad, open(d + "/pad.json", "w"), indent=1)
print(f"pad.json: {len(E)} pts, rms {pad['rms_mm']:.2f} mm, inlier frac {pad['inlier_frac']:.2f}, base tip height {pad['base_tip_height_mm']:.1f} mm", flush=True)
os._exit(0)
