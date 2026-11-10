#!/usr/bin/env python3
"""READ-ONLY. Held-needle 3D circle from one stereo pair: measured row offset, right arc, Viterbi disparity along the arc on the left thin-structure
response, robust plane + circle fit (radius free and fixed).  Prints the depth profile so the tilt branch is visible, not assumed."""
import json, os, sys
import numpy as np, cv2
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", R0 + "/G_place", os.path.expanduser("~/rl_live/sim2real_20261027/place/code")]
import tipmeas, needle_circle as NC, held_needle as hn
from rig import FX, CX, CY, FB

def row_offset(R, L, d0):
    gR, gL = cv2.cvtColor(R, 6).astype(np.float32), cv2.cvtColor(L, 6).astype(np.float32); out = []
    p = cv2.goodFeaturesToTrack(gR.astype(np.uint8), 400, 0.02, 25, blockSize=7)
    for (u, v) in p[:, 0]:
        u, v = int(u), int(v)
        if not (40 < v < 980 and 200 < u < 1250): continue
        T = gR[v - 12:v + 13, u - 12:u + 13]; x0 = int(u - d0)
        S = gL[v - 22:v + 23, max(0, x0 - 40):x0 + 41]
        if S.shape[1] < 60 or T.std() < 4: continue
        r = cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED); _, mx, _, loc = cv2.minMaxLoc(r)
        if mx > 0.9 and 0 < loc[1] < r.shape[0] - 1:
            a, b, c = r[loc[1] - 1, loc[0]], r[loc[1], loc[0]], r[loc[1] + 1, loc[0]]; out.append(loc[1] - 10 + 0.5 * (a - c) / (a - 2 * b + c))
    return float(np.median(out)), len(out), float(np.std(out))

def measure(R, L, roi, jaw, z_hint=115.0, drange=(70, 112), Rknown=None):
    d0 = FB / z_hint; dv, nn, sd = row_offset(R, L, d0)
    rR = tipmeas.detect_roi(R, z_hint, roi, 300, jaw)
    if rR is None: return None, "no arc in the right image"
    cR = NC.dense_arc(rR, 240); _, resp = hn.thin_mask(L); resp = cv2.GaussianBlur(resp.astype(np.float32), (5, 5), 1.0)
    ds = np.arange(drange[0], drange[1] + 0.25, 0.5); H, W = resp.shape; seg = np.gradient(cR, axis=0); slope = np.abs(seg[:, 1]) / (np.linalg.norm(seg, axis=1) + 1e-9)
    C = np.zeros((len(cR), len(ds)))
    for i, (u, v) in enumerate(cR):
        xs = np.clip(np.round(u - ds).astype(int), 0, W - 1); y = int(np.clip(round(v + dv), 0, H - 1)); C[i] = resp[y, xs]
    C = C / (np.percentile(C, 99) + 1e-6)
    dp = np.zeros_like(C); bp = np.zeros_like(C, int); dp[0] = C[0]; band = 2
    for i in range(1, len(cR)):
        for j in range(len(ds)):
            lo, hi = max(0, j - band), min(len(ds), j + band + 1); k = lo + int(np.argmax(dp[i - 1, lo:hi])); dp[i, j] = C[i, j] + dp[i - 1, k]; bp[i, j] = k
    j = int(np.argmax(dp[-1])); path = [j]
    for i in range(len(cR) - 1, 0, -1): j = bp[i, j]; path.append(j)
    path = np.array(path[::-1]); dd = ds[path]; score = C[np.arange(len(cR)), path]
    use = (slope > 0.25) & (score > 0.15)
    Zs = FB / dd; P = np.c_[(cR[:, 0] - CX) * Zs / FX, (cR[:, 1] - CY) * Zs / FX, Zs]; idx = np.nonzero(use)[0]; Q0 = P[idx]
    if len(Q0) < 30: return None, f"only {len(Q0)} points"
    keep = np.ones(len(Q0), bool)
    for _ in range(5):
        c0 = Q0[keep].mean(0); Vt = np.linalg.svd(Q0[keep] - c0, full_matrices=False)[2]; nrm, e1, e2 = Vt[2], Vt[0], Vt[1]
        q = np.c_[(Q0 - c0) @ e1, (Q0 - c0) @ e2]; sol = np.linalg.lstsq(np.c_[2 * q[keep], np.ones(keep.sum())], (q[keep] ** 2).sum(1), rcond=None)[0]; cc = sol[:2]; rad = float(np.sqrt(sol[2] + cc @ cc))
        res = np.sqrt((np.linalg.norm(q - cc, axis=1) - rad) ** 2 + ((Q0 - c0) @ nrm) ** 2); keep = res < max(0.4, 2.5 * np.median(res[keep]))
    Cc = c0 + cc[0] * e1 + cc[1] * e2
    return dict(centre=Cc, normal=nrm, radius_mm=rad, n=int(keep.sum()), n_all=len(Q0), rms_mm=float(np.sqrt(np.mean(res[keep] ** 2))), row_offset=dv, row_n=nn, row_sd=sd,
                cR=cR, dd=dd, use=use, P=P, keep_idx=idx[keep], tip_uv=cR[-1], tail_uv=cR[0]), "ok"

def ray_on_plane(uv, C, n):
    r = np.r_[(uv[0] - CX) / FX, (uv[1] - CY) / FX, 1.0]; return r * ((C @ n) / (r @ n))

if __name__ == "__main__":
    snap, rx, ry, jx, jy, out = sys.argv[1], *map(float, sys.argv[2:6]), sys.argv[6]; os.makedirs(out, exist_ok=True)
    R = cv2.imread(snap + "/right_rect.png"); L = cv2.imread(snap + "/left_rect.png")
    m, why = measure(R, L, (rx, ry), (jx, jy)); print(why)
    if m is None: sys.exit(1)
    print("row offset %.2f px (n=%d sd %.2f); arc points %d/%d rms %.2f mm radius %.2f; normal %s centre %s" % (m["row_offset"], m["row_n"], m["row_sd"], m["n"], m["n_all"], m["rms_mm"], m["radius_mm"], np.round(m["normal"], 3).tolist(), np.round(m["centre"], 1).tolist()))
    k = np.linspace(0, len(m["cR"]) - 1, 13).astype(int); print("along arc tail->tip  uv / disp / Z:"); [print("  ", np.round(m["cR"][i]).tolist(), m["dd"][i], round(FB / m["dd"][i], 1), "used" if m["use"][i] else "-") for i in k]
    tip = ray_on_plane(m["tip_uv"], m["centre"], m["normal"]); tail = ray_on_plane(m["tail_uv"], m["centre"], m["normal"]); print("tip", np.round(tip, 1).tolist(), "tail", np.round(tail, 1).tolist(), "chord %.1f" % np.linalg.norm(tip - tail))
    fs = np.load(snap + "/fs_disp.npy"); print("FS disp along arc", [round(float(np.median(fs[int(v) - 1:int(v) + 2, int(u) - 1:int(u) + 2])), 1) for u, v in m["cR"][k]])
    def P2(X, shift=0): return np.c_[FX * X[:, 0] / X[:, 2] + CX - shift * FB / X[:, 2], FX * X[:, 1] / X[:, 2] + CY + shift * m["row_offset"]]
    th = np.radians(np.arange(0, 361, 3)); n = m["normal"]; e1 = (tip - m["centre"]) / np.linalg.norm(tip - m["centre"]); e2 = np.cross(n, e1); X = m["centre"] + m["radius_mm"] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
    oR, oL = R.copy(), L.copy(); cv2.polylines(oR, [np.int32(P2(X))], False, (0, 255, 0), 1); cv2.polylines(oL, [np.int32(P2(X, 1))], False, (0, 255, 0), 1)
    cv2.imwrite(out + "/arc.png", np.hstack([cv2.resize(oL[380:800, 240:800], None, fx=1.1, fy=1.1), cv2.resize(oR[380:800, 330:890], None, fx=1.1, fy=1.1)]))
    json.dump(dict(centre=m["centre"], normal=m["normal"], radius_mm=m["radius_mm"], tip=tip, tail=tail, rms_mm=m["rms_mm"], n=m["n"], row_offset=m["row_offset"]), open(out + "/arc.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
