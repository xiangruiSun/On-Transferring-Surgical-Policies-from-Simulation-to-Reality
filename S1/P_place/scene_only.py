#!/usr/bin/env python3
"""READ-ONLY. Stickers + surface planes from one snapshot, no needle.  Output usable as --measure for place_repro.py."""
import json, os, sys
import numpy as np, cv2
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code"]
from rig import FX, CX, CY, FB
snap, out = sys.argv[1], sys.argv[2]; thr = int(sys.argv[3]) if len(sys.argv) > 3 else 120; os.makedirs(out, exist_ok=True); R = cv2.imread(snap + "/right_rect.png"); disp = np.load(snap + "/fs_disp.npy").astype(np.float32); Z = FB / np.clip(disp, 1, None); vv, uu = np.mgrid[0:disp.shape[0], 0:disp.shape[1]]; P = np.dstack([(uu - CX) * Z / FX, (vv - CY) * Z / FX, Z])
lab = cv2.cvtColor(R, cv2.COLOR_BGR2LAB); blue = cv2.morphologyEx((lab[..., 2] < thr).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); nb, lb, st, ct = cv2.connectedComponentsWithStats(blue); order = sorted(range(1, nb), key=lambda k: -st[k, 4])[:4]; S = []; ov = R.copy()
def plane(c, rad):
    d = np.linalg.norm(P - c, axis=2) < rad; Q = P[d][::3]; keep = np.ones(len(Q), bool)
    for _ in range(5):
        q0 = Q[keep].mean(0); nn = np.linalg.svd(Q[keep] - q0, full_matrices=False)[2][2]; r = (Q - q0) @ nn; keep = np.abs(r) < 2.5 * np.std(r[keep]) + 1e-6
    return (nn if nn[2] < 0 else -nn), float(np.std(r[keep]))
for k in order:
    if st[k, 4] < 300: continue
    mask = lb == k; c = np.median(P[mask], 0); s = dict(area=int(st[k, 4]), uv=ct[k].tolist(), bbox=st[k, :4].tolist(), cam=c.tolist(), disp=float(np.median(disp[mask])))
    for rad in (4, 8, 15, 25): n, rms = plane(c, rad); s[f"n{rad}"] = n.tolist(); s[f"rms{rad}"] = rms
    S.append(s); print("blob area", s["area"], "uv", np.round(s["uv"]).tolist(), "bbox", s["bbox"], "disp %.1f Z %.1f" % (s["disp"], c[2]), "cam", np.round(c, 1).tolist(), "n25", np.round(s["n25"], 3).tolist(), "tilt from view %.0f" % np.degrees(np.arccos(-s["n25"][2])))
    cv2.drawContours(ov, cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 255), 1); cv2.putText(ov, str(len(S) - 1), tuple(np.int32(ct[k])), 0, .8, (0, 255, 0), 2)
cv2.imwrite(out + "/scene.png", ov); json.dump(dict(stickers=S, snap=snap), open(out + "/measure.json", "w"), indent=1); print("median FS Z over the image %.1f; ->" % float(np.median(Z)), out)
