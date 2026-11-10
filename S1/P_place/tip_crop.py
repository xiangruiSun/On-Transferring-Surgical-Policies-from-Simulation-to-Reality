"""READ-ONLY (2026-11-05): zoomed right/left crops with a 10-px grid around a right-image pixel, for reading the needle tip by eye,
plus NCC disparity (ncc_disp.disp) at that pixel and at a point a few px back along the needle.  usage: tip_crop.py <snapdir> u v [u_back v_back] out.png"""
import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from ncc_disp import disp, FB, DV
d, u, v = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); back = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 6 else None; out = sys.argv[-1]
R = cv2.imread(d + "/right_rect.png"); L = cv2.imread(d + "/left_rect.png"); H = 60; S = 6
res = [("tip", (u, v))] + ([("back", back)] if back else []); dd = {}
for nm, (a, b) in res:
    best = disp(R, L, a, b, hw=7, d0=105, rng=40); dd[nm] = best; print("%s right (%d,%d): ncc %.2f disp %.1f Z %.1f mm -> left (%.1f,%.1f)" % (nm, a, b, best[0], best[1], FB / best[1], a - best[1], b + best[2]))
dl = dd["back"][1] if back else dd["tip"][1]
def crop(I, cu, cv_):
    c = I[cv_ - H:cv_ + H, cu - H:cu + H].copy(); c = cv2.resize(c, None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)
    for k in range(0, 2 * H + 1, 10):
        cv2.line(c, (k * S, 0), (k * S, 2 * H * S), (0, 255, 0) if k == H else (90, 90, 90), 1); cv2.line(c, (0, k * S), (2 * H * S, k * S), (0, 255, 0) if k == H else (90, 90, 90), 1)
    return c
cv2.imwrite(out, np.hstack([crop(R, u, v), crop(L, int(round(u - dl)), v + int(round(DV)))]))
print("crops: right centred (%d,%d), left centred (%d,%d); grid 10 px, green = centre" % (u, v, int(round(u - dl)), v + int(round(DV))))
