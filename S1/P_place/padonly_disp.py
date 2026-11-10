"""READ-ONLY offline (2026-11-05): pad-only disparity for place_repro --fssnap.
Tool pixels come from kinematics: hinge positions FK(q with insertion reduced by s, s = 0..80 mm) = shaft axis, plus hinge -> TCP (jaw),
projected with the calibration M; disc radius R_MM (shaft radius ~4.2 mm + margin). Those pixels get the disparity of a RANSAC plane fitted
to the remaining pixels; everything else (tissue, ridges, needle) is kept as measured."""
import sys, os, json, numpy as np, cv2
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K
from rig import TCP_MM
FX, CX, CY = 1860.724932, 690.538345, 576.803638; FB = FX * 5.4476353950418295; R_MM = 6.0
src, out = sys.argv[1], sys.argv[2]; d = np.load(src + "/fs_disp.npy").astype(np.float32); st = json.load(open(src + "/state.json"))
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); SUJ = np.array(cal["frozen"]["suj_psm1"]); q = np.array(st["js"][:6])
pts = []
for s in np.linspace(0, 0.08, 81):
    qq = q.copy(); qq[2] -= s; pts.append(K.fk_ecm(qq, SUJ)[:3, 3])
T = K.fk_ecm(q, SUJ); pts += [T[:3, 3] + T[:3, :3] @ (TCP_MM * 1e-3 * f) for f in np.linspace(0, 1.15, 24)]
mask = np.zeros(d.shape, np.uint8); proj = []
for p in pts:
    X = np.r_[p * 1e3, 1] @ M
    if X[2] < 25: proj.append((np.nan, np.nan, X[2])); continue      # shaft runs nearly along the view: points this close are outside the image
    u, v = FX * X[0] / X[2] + CX, FX * X[1] / X[2] + CY; proj.append((u, v, X[2]))
    cv2.circle(mask, (int(round(u)), int(round(v))), int(np.ceil(R_MM * FX / X[2])), 255, -1)
tool = mask > 0
Z = FB / np.clip(d, 1, None); vv, uu = np.mgrid[0:d.shape[0], 0:d.shape[1]]; P = np.dstack([(uu - CX) * Z / FX, (vv - CY) * Z / FX, Z])
Q = P[~tool][::25]; rng = np.random.default_rng(0); best = None
for _ in range(500):
    s3 = Q[rng.choice(len(Q), 3, replace=False)]; n = np.cross(s3[1] - s3[0], s3[2] - s3[0])
    if np.linalg.norm(n) < 1e-9: continue
    n /= np.linalg.norm(n); k = int((np.abs((Q - s3[0]) @ n) < 1.0).sum())
    if best is None or k > best[0]: best = (k, n, s3[0])
I = Q[np.abs((Q - best[2]) @ best[1]) < 1.0]; c = I.mean(0); n = np.linalg.svd(I - c)[2][2]; n = n if n[2] < 0 else -n
ray = np.dstack([(uu - CX) / FX, (vv - CY) / FX, np.ones_like(Z)]); zp = (c @ n) / (ray @ n)
d2 = d.copy(); d2[tool] = FB / zp[tool]; np.save(out + "/fs_disp.npy", d2.astype(np.float32)); cv2.imwrite(out + "/replaced_mask.png", mask)
print("plane n %s rms %.2f mm; tool mask %d px (%.1f%%); jaw hinge uv %s, shaft end uv %s" % (np.round(n, 3).tolist(), np.sqrt(np.mean(((I - c) @ n) ** 2)), tool.sum(), 100 * tool.mean(), np.round(proj[0][:2]).tolist(), [np.round(x[:2]).tolist() for x in proj[:81:20]]))
