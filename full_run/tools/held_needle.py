#!/usr/bin/env python3
"""READ-ONLY. Automatic detector for the needle HELD in the jaw (2026-11-10; replaces the operator's tip clicks).

    python3 held_needle.py <snap dir> <background snap dir> [--prior u,v] [--out result.json]

Method (right = reference image, left the same way):
  1. foreground = colour difference (Lab, lightness down-weighted) to a background snapshot of the same scene taken before
     the grasp (tool at the base, needle still on the pad); where the background itself shows a foreign object (tool, lying
     needle, thread) only pixels that are not pad-coloured NOW count, so the place the needle was lifted from is not foreground;
  2. the instrument = the thick part of the foreground; the thin parts attached to it are needle and thread;
  3. each thin part is skeletonised and walked from the instrument to its far end; the needle = the widest long one
     (the thread is thinner); its far end = the TIP;
  4. stereo: the left tip is the left far end; the tip disparity is cross-checked with a row-wise match of the two centre lines.
Returns the centre line jaw -> tip (right pixels), tip right/left, a body point ~2 mm behind the tip, and quality numbers.
"""
import json, os, sys
import numpy as np, cv2

FX, CX, CY = 1860.724932, 690.538345, 576.803638; FB = FX * 5.4476353950418295; DV = 2.2
HI, LO = 16.0, 8.0          # colour-difference thresholds (hysteresis)
TOOL_HALF = 13              # px: foreground thicker than this half-width is the instrument
MIN_LEN, MIN_W = 110, 14.0  # px: shortest centre line / smallest median width (of the FS foreground) accepted as the needle; 11-10: needle 17-25, thread 8-12.5


def lab(img): return cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 1.2), cv2.COLOR_BGR2LAB).astype(np.float32)


def pink(L): return (L[..., 1] > 140) & (L[..., 0] > 90)


B_MIN = 123.0              # Lab b of the centre line: violet thread 115-119, steel needle 128-130 (11-10, A176; the thread is as wide as 14.8 px)
DMIN = 2.5                  # px: foreground = disparity at least this much larger than the background snapshot's (closer to the camera)


def base_tool_mask(bg_dir):
    """where the instrument stands in the background snapshot (tool at the base): well in front (> 12 px, grown down to 5 px) of a plane fitted to the pad."""
    f = bg_dir + "/fs_tool_mask.png"
    if os.path.exists(f) and os.path.getmtime(f) >= os.path.getmtime(bg_dir + "/fs_disp.npy"): return cv2.imread(f, 0) > 0
    D = np.load(bg_dir + "/fs_disp.npy").astype(np.float32); h, w = D.shape; vv, uu = np.mgrid[0:h:4, 0:w:4]; A = np.c_[uu.ravel(), vv.ravel(), np.ones(uu.size)]; y = D[::4, ::4].ravel(); keep = np.ones(len(y), bool)
    for _ in range(8):
        c = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]; r = y - A @ c; keep = np.abs(r) < max(2.0, 2.0 * np.median(np.abs(r[keep])) * 1.48)
    V, U = np.mgrid[0:h, 0:w]; e = D - (c[0] * U + c[1] * V + c[2]); n, lb, st, _ = cv2.connectedComponentsWithStats((e > 5.0).astype(np.uint8)); ok = np.zeros(n, bool)
    for k in np.unique(lb[e > 12.0]):
        if k and st[k, 4] > 5000: ok[k] = True
    m = cv2.dilate(ok[lb].astype(np.uint8), np.ones((31, 31), np.uint8)); cv2.imwrite(f, m * 255); return m > 0


def foreground(cur_dir, bg_dir):
    """things in front of the static scene: FoundationStereo disparity of this snapshot minus the background's (both aligned with right_rect).
    Background = the pre-grasp snapshot (tool at the base, needle on the pad); where the base instrument stands in it, the smallest disparity
    seen there in the trial's other snapshots (the pad shows once the instrument has left).  2026-11-10 (A176 r2): after the rotations the needle
    and the jaw stood where the base instrument is in the background snapshot -> negative difference -> nothing found.  (A plane instead of the
    real pad there made the mouldings foreground.)  Places an object has LEFT get a negative difference, so they are not foreground."""
    cur = np.load(cur_dir + "/fs_disp.npy").astype(np.float32); B = np.load(bg_dir + "/fs_disp.npy").astype(np.float32).copy(); M = base_tool_mask(bg_dir); low = cur[M].copy(); par = os.path.dirname(os.path.abspath(cur_dir))
    for x in sorted(os.listdir(par)):
        f = f"{par}/{x}/fs_disp.npy"
        if x.endswith("_dry") or x == "padonly" or not os.path.exists(f) or not os.path.exists(f"{par}/{x}/right_rect.png") or os.path.getmtime(f) < os.path.getmtime(f"{par}/{x}/right_rect.png"): continue
        if os.path.abspath(f"{par}/{x}") in (os.path.abspath(cur_dir), os.path.abspath(bg_dir)): continue
        low = np.minimum(low, np.load(f)[M])
    B[M] = np.minimum(B[M], low); d = cur - B
    fg = cv2.morphologyEx((d > DMIN).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)); return fg > 0, d


def skeleton_paths(thin, tool_d):
    """for every thin component touching the instrument: centre line from the contact to the far end, with widths."""
    out = []; n, lb, st, _ = cv2.connectedComponentsWithStats(thin.astype(np.uint8), connectivity=8)
    dt_all = cv2.distanceTransform(thin.astype(np.uint8), cv2.DIST_L2, 5)
    for k in range(1, n):
        if st[k, 4] < 150: continue
        comp = lb == k
        if not (comp & tool_d).any(): continue
        x, y, w, h = st[k, :4]; x0, y0 = max(x - 2, 0), max(y - 2, 0); sub = comp[y0:y + h + 2, x0:x + w + 2].astype(np.uint8) * 255
        sk = cv2.ximgproc.thinning(sub) > 0; ys, xs = np.nonzero(sk)
        if len(ys) < 30: continue
        idx = -np.ones(sk.shape, int); idx[ys, xs] = np.arange(len(ys)); near = tool_d[y0:y + h + 2, x0:x + w + 2][ys, xs]
        if not near.any(): near = cv2.dilate(tool_d[y0:y + h + 2, x0:x + w + 2].astype(np.uint8), np.ones((15, 15), np.uint8))[ys, xs] > 0
        if not near.any(): continue
        # breadth-first geodesic distance along the skeleton from the instrument contact
        dist = np.full(len(ys), -1.0); prev = -np.ones(len(ys), int); q = list(np.nonzero(near)[0]); dist[q] = 0; head = 0
        while head < len(q):
            i = q[head]; head += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = ys[i] + dy, xs[i] + dx
                    if (dy or dx) and 0 <= yy < sk.shape[0] and 0 <= xx < sk.shape[1] and idx[yy, xx] >= 0 and dist[idx[yy, xx]] < 0:
                        j = idx[yy, xx]; dist[j] = dist[i] + np.hypot(dy, dx); prev[j] = i; q.append(j)
        end = int(np.argmax(dist)); path = [end]
        while prev[path[-1]] >= 0: path.append(prev[path[-1]])
        path = path[::-1]; P = np.c_[xs[path] + x0, ys[path] + y0].astype(float); W = 2 * dt_all[ys[path] + y0, xs[path] + x0]
        out.append(dict(P=P, W=W, len=float(dist[end]), area=int(st[k, 4])))
    return out


def smooth(P, k=9):
    if len(P) < 2 * k: return P
    ker = np.ones(k) / k; Q = P.copy()
    for i in (0, 1): Q[k // 2:-(k // 2), i] = np.convolve(P[:, i], ker, mode="valid")
    return Q


def detect(snap, bg, prior=None, out_png=None, cal_file=None):
    R, Lf = cv2.imread(snap + "/right_rect.png"), cv2.imread(snap + "/left_rect.png"); D = np.load(snap + "/fs_disp.npy").astype(np.float32)
    fg, dd = foreground(snap, bg); fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    core = cv2.distanceTransform(fg, cv2.DIST_L2, 5) > TOOL_HALF; n, lb, st, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8)); big = np.zeros(n, bool)
    for k in range(1, n): big[k] = st[k, 4] > 1500
    core = big[lb]; tool = (cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOOL_HALF + 7, 2 * TOOL_HALF + 7))) > 0) & (fg > 0)
    # jaw_mask() (kinematic jaw) is NOT used: 8+ mm off once wrist pitch / yaw leave the base values (A176 r2); pad_background() made it unnecessary
    thin = (fg > 0) & ~tool; tool_d = cv2.dilate(tool.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    cands = skeleton_paths(thin, tool_d); Lb = lab(R); best = None
    for c in cands:
        P, W = c["P"], c["W"]; c["w_med"] = float(np.median(W)); pi = P.astype(int); c["b_med"] = float(np.median(Lb[pi[:, 1], pi[:, 0], 2])); c["a_med"] = float(np.median(Lb[pi[:, 1], pi[:, 0], 1]))
        c["ok"] = bool(c["len"] >= MIN_LEN and c["w_med"] >= MIN_W and c["b_med"] >= B_MIN)
        c["score"] = c["len"] * min(c["w_med"], 14.0) - (0.0 if prior is None else 2.0 * np.linalg.norm(P[-1] - np.asarray(prior)))
        if c["ok"] and (best is None or c["score"] > best["score"]): best = c
    def draw():
        v = R.copy(); v[tool] = (0.6 * v[tool] + 0.4 * np.array([0, 0, 255])).astype(np.uint8); v[thin] = (0.6 * v[thin] + 0.4 * np.array([0, 255, 255])).astype(np.uint8)
        for c in cands: cv2.polylines(v, [np.int32(c["P"])], False, (0, 255, 0) if c is best else (255, 0, 0), 1); cv2.putText(v, "%.0f/%.0f" % (c["len"], c["w_med"]), tuple(np.int32(c["P"][-1]) + 6), 0, 0.5, (255, 255, 255), 1)
        if prior is not None: cv2.drawMarker(v, tuple(np.int32(prior)), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 14, 1)
        return v
    if best is None:
        if out_png: cv2.imwrite(out_png, draw())
        return dict(ok=False, why="no needle-like thin structure attached to the instrument", n_cands=len(cands))
    P = smooth(best["P"]); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]; tipR = P[-1]
    res = dict(ok=True, tipR=tipR.tolist(), len_px=best["len"], width_px=best["w_med"], n_cands=len(cands), n_ok=sum(c["ok"] for c in cands))
    ib = int(np.argmin(np.abs(s - (s[-1] - 45.0)))); res["bodyR"] = P[ib].tolist(); step = max(1, len(P) // 60); idx = list(range(0, len(P), step)); idx = idx + ([len(P) - 1] if idx[-1] != len(P) - 1 else [])
    # disparity along the centre line: FS (largest value across the needle width: the background bleeds in from the sides) and NCC on
    # the wire itself; robust quadratic in arc length, evaluated at the tip
    pi = P.astype(int); dfs = np.array([D[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4].max() for u, v in pi]); res["disp_fs_tip"] = float(np.median(dfs[-12:]))
    sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261106/S1/P_place")); from ncc_disp import disp as ncc
    dn = []
    for k in idx:
        u, v = P[k]; b = None
        if 12 < u < R.shape[1] - 12 and 12 < v < R.shape[0] - 14:
            for h in (7, 9):
                try:
                    r_ = ncc(R, Lf, u, v, hw=h, d0=float(dfs[k]), rng=10)
                    if b is None or r_[0] > b[0]: b = r_
                except Exception: pass
        dn.append((s[k], b[1], b[0]) if b else (s[k], np.nan, 0.0))
    dn = np.array(dn); good = dn[:, 2] > 0.6
    def fit(sv, dv):
        keep = np.ones(len(sv), bool); p = None
        for _ in range(4):
            if keep.sum() < 6: return None, None
            p = np.polyfit(sv[keep], dv[keep], 2 if np.ptp(sv[keep]) > 150 else 1); e = dv - np.polyval(p, sv); keep = np.abs(e) < max(1.0, 2.5 * np.std(e[keep]))
        return p, float(np.std(e[keep]))
    pf, ef = fit(s, dfs); pn, en = fit(dn[good, 0], dn[good, 1]) if good.sum() >= 6 else (None, None)
    if pf is not None: res.update(disp_fs_fit=float(np.polyval(pf, s[-1])), disp_fs_rms=ef)
    if pn is not None: res.update(disp_ncc_fit=float(np.polyval(pn, s[-1])), disp_ncc_rms=en, n_ncc=int(good.sum()))
    cand = [res[k] for k in ("disp_ncc_fit", "disp_fs_fit") if k in res]
    if not cand: res.update(ok=False, why="no tip disparity"); return res
    use = cand[0]; res["disp_src"] = "ncc centre-line fit" if "disp_ncc_fit" in res else "fs centre-line fit"
    if len(cand) == 2: res["disp_ncc_minus_fs"] = cand[0] - cand[1]
    res["disp"] = float(use); res["tipL"] = [float(tipR[0] - use), float(tipR[1] + DV)]; Z = FB / use; res["tip_cam"] = [(tipR[0] - CX) * Z / FX, (tipR[1] - CY) * Z / FX, Z]
    pd = pn if pn is not None else pf; res["polyR"] = P[idx].tolist(); res["polyD"] = np.polyval(pd, s[idx]).tolist()
    if not (60 < use < 260): res.update(ok=False, why=f"implausible tip disparity {use:.1f}")
    if len(cand) == 2 and abs(cand[0] - cand[1]) > 5: res.update(ok=False, why=f"NCC and FS tip disparity differ by {cand[0] - cand[1]:.1f} px")
    if out_png:
        v = draw(); cv2.drawMarker(v, tuple(np.int32(tipR)), (255, 255, 255), cv2.MARKER_CROSS, 22, 1); cv2.circle(v, tuple(np.int32(res["bodyR"])), 5, (255, 0, 255), 1)
        w = Lf.copy(); cv2.drawMarker(w, tuple(np.int32(res["tipL"])), (255, 255, 255), cv2.MARKER_CROSS, 22, 1)
        cv2.putText(v, "tip R (%.0f,%.0f) disp %.1f (%s) Z %.1f mm" % (tipR[0], tipR[1], use, res["disp_src"], Z), (10, 30), 0, 0.8, (255, 255, 255), 2); cv2.imwrite(out_png, np.hstack([v, w]))
    return res


if __name__ == "__main__":
    a = sys.argv[1:]; prior = None; out = None
    if "--prior" in a: prior = [float(x) for x in a[a.index("--prior") + 1].split(",")]
    if "--out" in a: out = a[a.index("--out") + 1]
    r = detect(a[0], a[1], prior, (out or a[0] + "/held_needle.json").replace(".json", ".png"))
    json.dump(r, open(out or a[0] + "/held_needle.json", "w"), indent=1)
    print(json.dumps({k: (np.round(v, 1).tolist() if isinstance(v, (list, float)) else v) for k, v in r.items() if k not in ("polyR", "polyD")}))
    sys.exit(0 if r.get("ok") else 1)
