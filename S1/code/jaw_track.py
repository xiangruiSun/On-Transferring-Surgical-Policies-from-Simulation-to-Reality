"""read-only offline: locate a template (from ref image, box u0 v0 half) in other images (left raw snaps); prints pixel + score."""
import sys, cv2, numpy as np
ref, u0, v0, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); R = cv2.imread(ref, 0); T = R[v0 - h:v0 + h, u0 - h:u0 + h]
for f in sys.argv[5:]:
    I = cv2.imread(f, 0); r = cv2.matchTemplate(I, T, cv2.TM_CCOEFF_NORMED); _, s, _, loc = cv2.minMaxLoc(r); print(f.split('/')[-1], 'uv', loc[0] + h, loc[1] + h, 'score %.2f' % s)
