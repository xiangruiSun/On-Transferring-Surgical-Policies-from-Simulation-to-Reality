#!/usr/bin/env python3
"""READ-ONLY. Sub-pixel disparity of the needle by NCC: patches centred on the needle in the right image, searched along the same (row-corrected) row in the left."""
import sys, json
import numpy as np, cv2
FB = 1860.724932 * 5.4476353950418295; DV = 2.2
def disp(R, L, u, v, hw=9, d0=100, rng=22, dvs=(1, 2, 3)):
    gR, gL = cv2.cvtColor(R, 6).astype(np.float32), cv2.cvtColor(L, 6).astype(np.float32); u, v = int(round(u)), int(round(v)); T = gR[v - hw:v + hw + 1, u - hw:u + hw + 1]; best = None
    for dv in dvs:
        x0 = int(u - d0 - rng - hw); S = gL[v + dv - hw:v + dv + hw + 1, x0:x0 + 2 * rng + 2 * hw + 2]; r = cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED)[0]; k = int(np.argmax(r))
        sub = 0.0
        if 0 < k < len(r) - 1:
            den = r[k - 1] - 2 * r[k] + r[k + 1]; sub = 0.5 * (r[k - 1] - r[k + 1]) / den if den < 0 else 0
        d = u - (x0 + hw + k + sub)
        if best is None or r[k] > best[0]: best = (float(r[k]), float(d), dv)
    return best
if __name__ == "__main__":
    for spec in sys.argv[1:]:
        d, pts = spec.split("=", 1); R = cv2.imread(d + "/right_rect.png"); L = cv2.imread(d + "/left_rect.png"); out = []
        for p in pts.split(";"):
            u, v = map(float, p.split(",")); b = disp(R, L, u, v, d0=FB / 100); out.append("(%d,%d) ncc %.2f d %.1f Z %.1f" % (u, v, b[0], b[1], FB / b[1]))
        print(d, " | ".join(out))
