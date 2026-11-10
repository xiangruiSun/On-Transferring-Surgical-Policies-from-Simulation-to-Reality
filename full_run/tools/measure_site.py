#!/usr/bin/env python3
"""READ-ONLY. Placing site from a stereo snapshot with FS disparity: entry = the blue sticker (largest blue blob, centroid in camera mm,
local plane normal n25), exit = VIRTUAL point EXIT_MM along the pad plane in the image +x direction (across the slit), stored as a second
'sticker' with a larger area so P_place/phase2.py / rot_tip.py / best_reach.py (entry = smaller area) pick the real sticker as the entry.
usage: measure_site.py <snap dir with right_rect.png + fs_disp.npy> <out dir> [exit_mm=13.7] [image direction ux,uy = 1,0]"""
import json, os, sys, numpy as np, cv2
FX, CX, CY = 1860.724932, 690.538345, 576.803638; FB = FX * 5.4476353950418295
snap, out = sys.argv[1], sys.argv[2]; EXIT_MM = float(sys.argv[3]) if len(sys.argv) > 3 else 13.7
R = cv2.imread(snap + "/right_rect.png"); disp = np.load(snap + "/fs_disp.npy").astype(np.float32)
Z = FB / np.clip(disp, 1, None); vv, uu = np.mgrid[0:disp.shape[0], 0:disp.shape[1]]; P = np.dstack([(uu - CX) * Z / FX, (vv - CY) * Z / FX, Z])
lab = cv2.cvtColor(R, cv2.COLOR_BGR2LAB); blue = cv2.morphologyEx((lab[..., 2] < 120).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
nb, lb, st, ct = cv2.connectedComponentsWithStats(blue); k = max(range(1, nb), key=lambda j: st[j, 4]); mask = lb == k
e = np.median(P[mask], 0)
def plane(c, rad):
    d = np.linalg.norm(P - c, axis=2) < rad; Q = P[d][::3]; keep = np.ones(len(Q), bool)
    for _ in range(5):
        q0 = Q[keep].mean(0); nn = np.linalg.svd(Q[keep] - q0, full_matrices=False)[2][2]; r = (Q - q0) @ nn; keep = np.abs(r) < 2.5 * np.std(r[keep]) + 1e-6
    return (nn if nn[2] < 0 else -nn), float(np.std(r[keep]))
N = {r: plane(e, r)[0].tolist() for r in (4, 8, 15)}; n25, rms = plane(e, 25)
DIR = [float(t) for t in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1.0, 0.0]   # 2026-11-10: exit direction in the image (default +x; "0,-1" = up)
x_img = np.array([DIR[0], DIR[1], 0.0]) / np.hypot(*DIR); u = x_img - (x_img @ n25) * n25; u /= np.linalg.norm(u); x = e + EXIT_MM * u
uv_x = (FX * x[0] / x[2] + CX, FX * x[1] / x[2] + CY)
entry = dict(area=int(st[k, 4]), uv=ct[k].tolist(), cam=e.tolist(), n25=n25.tolist(), n4=N[4], n8=N[8], n15=N[15], n_rms=rms, disp=float(np.median(disp[mask])))
exit_ = dict(area=10 ** 9, uv=list(uv_x), cam=x.tolist(), n25=n25.tolist(), n4=n25.tolist(), n8=n25.tolist(), n15=n25.tolist(), virtual=True, note=f"virtual exit {EXIT_MM} mm along image direction {DIR} projected on the pad plane")
os.makedirs(out, exist_ok=True); json.dump(dict(stickers=[exit_, entry], note="entry = blue sticker (smaller area); exit virtual"), open(out + "/measure.json", "w"), indent=1)
ov = R.copy(); cv2.drawContours(ov, cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 255), 2)
cv2.circle(ov, tuple(np.int32(ct[k])), 6, (0, 255, 0), 2); cv2.arrowedLine(ov, tuple(np.int32(ct[k])), tuple(np.int32(uv_x)), (0, 0, 255), 2); cv2.imwrite(out + "/measure.png", ov)
print("entry uv %s cam %s mm, normal %s (rms %.2f mm, tilt from view %.0f deg); virtual exit uv %s" % (np.round(ct[k], 1).tolist(), np.round(e, 2).tolist(), np.round(n25, 3).tolist(), rms, np.degrees(np.arccos(-n25[2])), np.round(uv_x, 1).tolist()))
