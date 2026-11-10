#!/usr/bin/env python3
"""2026-11-09: session_driver.fs_pad() on a NEW base snapshot (phantom moved after the calibration). Same method/constants;
only the snapshot dir is an argument. --write replaces D_pad/pad.json (old one renamed _superseded_HHMMSS)."""
import sys, os, json, time, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_driver as SD
from calib import cam_points, cam2ecm_mm
from rig import tip_of
snap = sys.argv[1]; write = '--write' in sys.argv; root = SD.ROOT; margin_mm = 2.0
cal = SD.load_cal(root); M = np.array(cal["M"])
R0 = cv2.imread(snap + "/right_rect.png"); P = cam_points(np.load(snap + "/fs_disp.npy"))
lab = cv2.cvtColor(R0, cv2.COLOR_BGR2LAB).astype(np.float32); pink = (lab[..., 1] > 140) & (lab[..., 0] > 90)
clear = cv2.distanceTransform(pink.astype(np.uint8), cv2.DIST_L2, 5) >= 40
Q = P[clear][::25]; rng = np.random.default_rng(0); best = None
for _ in range(500):
    s3 = Q[rng.choice(len(Q), 3, replace=False)]; n = np.cross(s3[1] - s3[0], s3[2] - s3[0])
    if np.linalg.norm(n) < 1e-9: continue
    n /= np.linalg.norm(n); k = int((np.abs((Q - s3[0]) @ n) < 0.5).sum())
    if best is None or k > best[0]: best = (k, n, s3[0])
inl = np.abs((Q - best[2]) @ best[1]) < 0.5; I = Q[inl]
E = np.array([cam2ecm_mm(x, M) for x in I]) * 1e-3; c = E.mean(0); _, S, Vt = np.linalg.svd(E - c); n = Vt[2]
tip0 = tip_of(np.array(cal["base_T"]))
if (tip0 - c) @ n < 0: n = -n
resid = np.abs((E - c) @ n) * 1e3; dd = -float(n @ c) - margin_mm * 1e-3
pad = dict(normal=n.tolist(), d=dd, n_points=int(len(E)), inlier_frac=float(len(I) / len(Q)), max_resid_mm=float(np.percentile(resid, 95)), rms_mm=float(np.sqrt((resid ** 2).mean())),
           margin_mm=margin_mm, base_tip_height_mm=float((n @ tip0 + dd) * 1e3),
           source=f"FS plane ({snap}, phantom moved after calibration) mapped with calibration_tcp.json (pin-corrected); shifted margin_mm toward free space; code/pad_from_snap_1109.py")
A = M[:3]; ncam = n @ A / np.linalg.norm(n @ A)
print(json.dumps({k: v for k, v in pad.items() if k != 'source'}, indent=0)); print('normal -> cam', np.round(ncam, 3))
# inlier map for a visual check
vis = R0.copy(); ys, xs = np.nonzero(clear); sel = np.arange(len(ys))[::25][inl]; vis[ys[sel], xs[sel]] = (0, 255, 0)
cv2.imwrite(snap + "/pad_inliers.png", vis)
if write:
    p = root + "/D_pad/pad.json"
    if os.path.exists(p): os.rename(p, p.replace(".json", time.strftime("_superseded_%H%M%S.json")))
    json.dump(pad, open(p, "w"), indent=1)
    SD.status_line(root + "/D_pad", f"D: FS pad plane from {os.path.basename(snap)} (phantom moved), margin {margin_mm} mm, {len(E)} pts, inlier frac {pad['inlier_frac']:.2f}, rms {pad['rms_mm']:.2f} mm, base tip height {pad['base_tip_height_mm']:.1f} mm", [p])
    print('written', p)
