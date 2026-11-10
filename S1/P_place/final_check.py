#!/usr/bin/env python3
"""READ-ONLY. Final pose: needle points by NCC stereo -> plane, heading of the bite vs entry->exit, tip direction vs surface normal."""
import json, sys
import numpy as np, cv2
sys.path.insert(0, ".")
from ncc_disp import disp, FB
FX, CX, CY = 1860.724932, 690.538345, 576.803638
d = sys.argv[1]; R = cv2.imread(d + "/right_rect.png"); L = cv2.imread(d + "/left_rect.png"); gray = cv2.cvtColor(R, 6).astype(np.float32)
st = json.load(open("m0/measure.json"))["stickers"]; en, ex = (st[1], st[0]) if st[1]["area"] < st[0]["area"] else (st[0], st[1]); e, x = np.array(en["cam"]), np.array(ex["cam"]); n = np.array(en["n25"]); n = n if n[2] < 0 else -n; u = (x - e) - ((x - e) @ n) * n; u /= np.linalg.norm(u)
# arc pixels: hand-seeded along the visible needle, each snapped to the darkest pixel across a small window
seeds = [(741, 566), (725, 572), (714, 590), (714, 612), (722, 632), (735, 650), (755, 668), (780, 684), (810, 698), (840, 708), (870, 720)]
P = []
for (a, b) in seeds:
    r = disp(R, L, a, b, d0=(FB / 100), rng=25); Z = FB / r[1]; X = np.array([(a - CX) * Z / FX, (b - CY) * Z / FX, Z]); print((a, b), "ncc %.2f d %.1f Z %.1f" % (r[0], r[1], Z))
    if r[0] > 0.75: P.append(X)
P = np.array(P); c = P.mean(0); Vt = np.linalg.svd(P - c, full_matrices=False)[2]; m = Vt[2]; rms = float(np.sqrt(np.mean(((P - c) @ m) ** 2)))
print("points %d, plane rms %.2f mm, plane normal %s" % (len(P), rms, np.round(m, 3).tolist()))
print("plane vs surface: needle plane contains n within %.1f deg (0 = vertical)" % np.degrees(np.arcsin(abs(m @ n))))
th = np.cross(n, m); th /= np.linalg.norm(th)                       # horizontal line in the needle plane
tail = P[-1] - P[0]; hz = tail - (tail @ n) * n; hz /= np.linalg.norm(hz)
print("heading tip -> jaw (horizontal) vs entry->exit: %.0f deg (180 = jaw trails directly away from the exit, 0 = jaw toward the exit)" % np.degrees(np.arccos(np.clip(hz @ u, -1, 1))))
print("angle of the plane's horizontal line vs entry->exit: %.0f deg (0 = plane through both stickers)" % np.degrees(np.arccos(abs(th @ u))))
t = P[0] - P[3]; t /= np.linalg.norm(t); print("near-tip direction vs -n: %.0f deg" % np.degrees(np.arccos(np.clip(t @ -n, -1, 1))), "; component along entry->exit %.2f" % (t @ u))
print("height of the point near the tip above entry level %.1f mm" % ((P[0] - e) @ n))
