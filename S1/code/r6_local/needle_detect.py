#!/usr/bin/env python3
"""Find the needle on the physical-left image automatically -> R6 clicks.json.

    python3 tools/needle_detect.py --snap <trial>/snap0 --out <trial>/clicks.json

Replaces the hand-clicked polyline ``r6_perceive.py`` takes.  A curved needle
is a circle in 3D, an ellipse in the image:

1. colour difference to the local pad (Lab, 41 px median) picks out thin
   things that are not pad -- needle, thread, letter edges, the instrument;
   blobs the size of an instrument are dropped;
2. RANSAC ellipses through those pixels, scored by the longest *contiguous*
   run of inliers (an arc, not scattered edges), semi-axes in the range a
   6..13 mm needle can have at the pad's depth;
3. the run's two ends: the one the thread (lavender) leaves from is the tail.

Writes ``polyline_px`` tail -> tip, as ``r6_perceive.py`` wants it, plus the
ellipse and a check image.  Camera convention as R6 (right_rect.png =
physical left).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import os
import numpy as np

FX = 1860.724932
FB = FX * 5.4476353950418295


def diff_map(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    bg = np.dstack([cv2.medianBlur(lab[..., i], 41) for i in range(3)]).astype(np.float32)
    return np.linalg.norm(lab.astype(np.float32) - bg, axis=2)


def thread_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (135, 25, 110), (175, 95, 230)) > 0


def candidates(img, thr, max_blob):
    d = diff_map(img)
    m = (d > thr).astype(np.uint8)
    # drop the instrument (and anything else that big): its strong-difference blob
    strong = (d > 2 * thr).astype(np.uint8)
    # the instrument's texture is fragmented; close it into one blob (a thin
    # needle arc stays thin and small)
    strong = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(strong)
    big = np.zeros_like(m)
    for k in range(1, n):
        if st[k, 4] > max_blob:
            big[lab == k] = 1
    # metal is unsaturated next to the pink pad: big low-saturation blobs too
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grey = ((hsv[..., 1] < 70) & (hsv[..., 2] > 40)).astype(np.uint8)
    _vmax = os.environ.get("NEEDLE_GREY_VMAX")   # 2026-11-06 17:2x (user: adjust, scene fixed): washed-out pad (S 24-63, V ~250) reads as grey metal; metal/needle V 135-215. Unset = unchanged.
    if _vmax: grey &= (hsv[..., 2] < float(_vmax)).astype(np.uint8)
    grey = cv2.morphologyEx(grey, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    grey = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grey)
    for k in range(1, n):
        if st[k, 4] > 15000:
            big[lab == k] = 1
    big = cv2.morphologyEx(big, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))   # 2026-11-04: thin thread/needle must not join the wood + jaw blob
    big = cv2.dilate(big, np.ones((25, 25), np.uint8))
    m[big > 0] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return m.astype(bool), d, big.astype(bool)


def ellipse_residual(pts, e):
    (cx, cy), (w, h), ang = e
    a, b = w / 2.0, h / 2.0
    t = np.deg2rad(ang)
    c, s = np.cos(t), np.sin(t)
    x, y = pts[:, 0] - cx, pts[:, 1] - cy
    xr, yr = c * x + s * y, -s * x + c * y
    rho = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
    phi = np.arctan2(yr / b, xr / a)
    return np.abs(rho - 1.0) * np.sqrt(a * b), phi


def longest_run(phi, gap_deg):
    """Longest contiguous run of angles (wrapping); returns (indices, start, span) in radians."""
    order = np.argsort(phi)
    p = phi[order]
    if len(p) < 2:
        return order, 0.0, 0.0
    gaps = np.diff(np.r_[p, p[0] + 2 * np.pi])
    gap = np.deg2rad(gap_deg)
    if np.all(gaps <= gap):
        return order, -np.pi, 2 * np.pi
    breaks = np.nonzero(gaps > gap)[0]
    best = None
    for j in range(len(breaks)):
        s = (breaks[j] + 1) % len(p)
        e = breaks[(j + 1) % len(breaks)]
        idx = np.arange(s, e + 1) if e >= s else np.r_[np.arange(s, len(p)), np.arange(0, e + 1)]
        span = (p[e] - p[s]) % (2 * np.pi)
        if best is None or span > best[2]:
            best = (order[idx], p[s], span)
    return best


def ransac_ellipse(pts, rmin, rmax, iters=4000, tol=3.0, gap_deg=12.0, seed=0):
    rng = np.random.default_rng(seed)
    tree_pts = pts
    best = None
    for _ in range(iters):
        i0 = rng.integers(len(pts))
        near = np.nonzero(np.linalg.norm(tree_pts - tree_pts[i0], axis=1) < 2.2 * rmax)[0]
        if len(near) < 5:
            continue
        sel = tree_pts[rng.choice(near, 5, replace=False)]
        try:
            e = cv2.fitEllipse(sel.astype(np.float32))
        except cv2.error:
            continue
        a, b = sorted(e[1])
        if not (2 * rmin <= a and b <= 2 * rmax and a / b >= 0.6):
            continue
        res, phi = ellipse_residual(pts, e)
        inl = res < tol
        if inl.sum() < 60:
            continue
        idx, start, span = longest_run(phi[inl], gap_deg)
        score = len(idx) * min(1.0, np.degrees(span) / 100.0)
        if best is None or score > best[0]:
            best = (score, e, np.nonzero(inl)[0][idx], np.degrees(span))
    return best


def band_profile(img, d, thr, at, ph0, ph1, step_deg=3.0, half=10, exclude=None):
    """Per angle step from ph0 to ph1: width of the not-pad band across the arc and its lavender share."""
    th = thread_mask(img)
    H, W = d.shape
    n = max(2, int(abs(np.degrees(ph1 - ph0)) / step_deg) + 1)
    rows = []
    for ph in np.linspace(ph0, ph1, n):
        q = at(ph)[0]
        q2 = at(ph + 1e-3)[0]
        t = (q2 - q) / np.linalg.norm(q2 - q)
        nrm = np.array([-t[1], t[0]])
        width, lav = 0, 0
        for r in range(-half, half + 1):
            x, y = (q + nrm * r).round().astype(int)
            if 0 <= x < W and 0 <= y < H and d[y, x] > thr and (exclude is None or not exclude[y, x]):
                width += 1
                lav += int(th[y, x])
        rows.append((float(ph), width, lav / max(width, 1)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=22.0, help="Lab difference to the local pad")
    ap.add_argument("--max-blob", type=int, default=12000, help="px; larger strong blobs = instrument")
    ap.add_argument("--radius-mm", type=float, nargs=2, default=[6.0, 13.0])
    ap.add_argument("--depth-mm", type=float, nargs=2, default=None,
                    help="pad depth range for the pixel-radius prior (default: from fs_disp.npy, else 60..200)")
    ap.add_argument("--n-poly", type=int, default=13)
    ap.add_argument("--lavender", type=float, default=0.65,
                    help="band share of thread colour above which an end bin is thread")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    snap = Path(a.snap)
    img = cv2.imread(str(snap / "right_rect.png"))
    m, d, big = candidates(img, a.thr, a.max_blob)
    if a.depth_mm:
        zlo, zhi = a.depth_mm
    elif (snap / "fs_disp.npy").exists():
        disp = np.load(snap / "fs_disp.npy")
        z = FB / np.clip(disp[~big], 1, None)
        zlo, zhi = np.percentile(z, 5) * 0.9, np.percentile(z, 95) * 1.1
    else:
        zlo, zhi = 60.0, 200.0
    rmin, rmax = FX * a.radius_mm[0] / zhi, FX * a.radius_mm[1] / zlo
    ys, xs = np.nonzero(m)
    pts = np.c_[xs, ys].astype(np.float64)
    if len(pts) > 30000:
        pts = pts[np.random.default_rng(1).choice(len(pts), 30000, replace=False)]
    best = ransac_ellipse(pts, rmin, rmax)
    out = {"ok": False, "method": "tools/needle_detect.py (auto)", "radius_px_prior": [rmin, rmax],
           "depth_prior_mm": [zlo, zhi]}
    if best is None:
        json.dump(out, open(a.out, "w"), indent=1)
        print("no needle-like arc found")
        return 1
    _, e, idx, span = best
    # refine on the run's pixels, then re-take the run
    e = cv2.fitEllipse(pts[idx].astype(np.float32))
    res, phi = ellipse_residual(pts, e)
    inl = res < 3.0
    run, start, span = longest_run(phi[inl], 12.0)
    (cx, cy), (w, h), ang = e
    A, B = w / 2.0, h / 2.0
    t = np.deg2rad(ang)
    Rm = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    def at(ph):
        ph = np.atleast_1d(ph)
        return (Rm @ np.vstack([A * np.cos(ph), B * np.sin(ph)])).T + [cx, cy]

    # the thread often continues along nearly the same circle from the swage:
    # trim bins at either end that are mostly lavender (the needle body itself
    # reflects some purple, up to ~0.55 of its band)
    prof = band_profile(img, d, a.thr, at, start, start + span, exclude=big)
    if a.verbose:
        for ph, wd, lv in prof:
            print(f"  {np.degrees(ph - start):6.1f} deg  {at(ph)[0].round()}  width {wd:2d}  lavender {lv:.2f}")
    wmed = float(np.median([w_ for _, w_, _ in prof]))

    def threadlike(row):
        return row[2] > a.lavender or row[1] < 0.5 * wmed

    def cut(rows):
        """Change point: the prefix length maximising threadlike minus needle-like bins."""
        score = np.cumsum([1 if threadlike(r) else -1 for r in rows])
        k = int(np.argmax(score)) + 1
        return k if score[k - 1] > 0 else 0

    i0 = cut(prof)
    i1 = len(prof) - 1 - cut(prof[::-1])
    if i1 - i0 < 5:
        i0, i1 = 0, len(prof) - 1
    trimmed = (i0, len(prof) - 1 - i1)
    start, span = prof[i0][0], prof[i1][0] - prof[i0][0]
    phs = start + np.linspace(0, span, a.n_poly)
    poly = at(phs)
    # tail: the end the thread leaves from (lavender pixels just beyond the end)
    th = thread_mask(img) & ~big & (d > a.thr)   # 2026-11-04: instrument reflections and pad pixels in the lavender hue range are not thread
    _ps = float(os.environ.get("NEEDLE_PX_SCALE", "1"))   # 2026-11-06 17:2x: tail-test pixel distances were set at ~146 mm; scale ~ 146 / pad depth. Unset = 1 = unchanged.

    def thread_near(end_ph, direction):
        cnt = 0
        for k in range(1, 6):
            q = at(end_ph + direction * np.deg2rad(4 * k))[0]
            x0, y0 = int(q[0]), int(q[1])
            _w = int(round(12 * _ps)); win = th[max(y0 - _w, 0):y0 + _w + 1, max(x0 - _w, 0):x0 + _w + 1]
            cnt += int(win.sum())
        # also anything lavender within 35 px of the end itself
        q = at(end_ph)[0]
        x0, y0 = int(q[0]), int(q[1])
        _w = int(round(35 * _ps)); cnt += int(th[max(y0 - _w, 0):y0 + _w + 1, max(x0 - _w, 0):x0 + _w + 1].sum())
        return cnt

    t_start, t_end = thread_near(start, -1), thread_near(start + span, +1)
    # a trimmed thread run is the strongest evidence of the tail
    t_start += 400 * trimmed[0]
    t_end += 400 * trimmed[1]
    if t_end > t_start:
        poly = poly[::-1]
    conf = abs(t_end - t_start) / max(t_end + t_start, 1)
    # 2026-11-04: the tail is the end the long thread is attached to (A02: 1421 vs 804 stray lavender px picked the tip, conf 0.28)
    _n, _lab, _st, _ = cv2.connectedComponentsWithStats(cv2.dilate(th.astype(np.uint8), np.ones((5, 5), np.uint8)))
    if _n > 1 and _st[1:, 4].max() >= 400 * _ps ** 2:
        _ys, _xs = np.nonzero(_lab == 1 + int(np.argmax(_st[1:, 4]))); _Q = np.c_[_xs, _ys].astype(np.float64)
        _Q = _Q[np.linalg.norm(_Q[:, None] - poly[None], axis=2).min(1) > 60.0 * _ps]   # 2026-11-04 fix B (user approved), A06/A13: the needle reads lavender and joins the thread (halo around the arc): keep only what is far off the arc
        if len(_Q) == 0: _Q = np.array([[1e6, 1e6]])
        _d0, _d1 = [float(np.linalg.norm(_Q - q, axis=1).min()) for q in (poly[0], poly[-1])]
        if min(_d0, _d1) <= 90 * _ps and max(_d0, _d1) >= 150 * _ps:
            if _d1 < _d0: poly = poly[::-1]
            conf = max(conf, 0.9)
    _hint = os.environ.get("NEEDLE_TAIL_HINT")   # 2026-11-06 17:3x (user: "上面的是针头，下面的是针尾"): human-given tail pixel "u,v"; orients the polyline, recorded in the output. Unset = unchanged.
    if _hint:
        _h = np.array([float(x) for x in _hint.split(",")])
        if np.linalg.norm(poly[-1] - _h) < np.linalg.norm(poly[0] - _h): poly = poly[::-1]
        out["tail_from_human_hint"] = _hint; conf = 1.0
        # 17:4x (user: adjust, scene fixed): the human tail point is the swage end; drop polyline points beyond it (thread continuing the arc)
        _k = int(np.argmin(np.linalg.norm(poly - _h, axis=1)))
        if len(poly) - _k >= 6: poly = np.vstack([_h, poly[_k + 1:]]) if np.linalg.norm(poly[_k] - _h) < 60 else poly[_k:]
    # an end at the image border may be a cut, not the needle's end: tail and
    # tip are then guesses -- refuse
    H, W = img.shape[:2]
    at_border = bool(any(min(q[0], q[1], W - 1 - q[0], H - 1 - q[1]) < 20 for q in (poly[0], poly[-1])))
    if _hint and at_border and min(poly[0][0], poly[0][1], W - 1 - poly[0][0], H - 1 - poly[0][1]) >= 20:   # 17:4x: tail given by a human and inside the image -> a tip cut by the border is accepted (flagged)
        out["tip_at_border_accepted"] = True; at_border = False
    out.update(trimmed_bins=list(trimmed), band_width_px=wmed, ok=bool(np.degrees(span) >= 70.0 and conf >= 0.3 and not at_border), polyline_px=poly.round(1).tolist(),
               ellipse={"centre": [cx, cy], "axes": [w, h], "angle_deg": ang},
               arc_span_deg_image=float(np.degrees(span)), n_inliers=int(len(run)),
               thread_px_at_ends={"tail": max(t_start, t_end), "tip": min(t_start, t_end)},
               tail_confidence=float(conf), end_at_border=at_border)
    json.dump(out, open(a.out, "w"), indent=1)
    vis = img.copy()
    vis[m] = (0.5 * vis[m] + [0, 127, 0]).astype(np.uint8)
    cv2.polylines(vis, [poly.round().astype(np.int32)], False, (0, 255, 255), 2)
    cv2.circle(vis, tuple(poly[0].round().astype(int)), 12, (255, 0, 0), 3)    # tail: blue
    cv2.circle(vis, tuple(poly[-1].round().astype(int)), 12, (0, 0, 255), 3)   # tip: red
    cv2.imwrite(str(Path(a.out).with_suffix(".png")), vis)
    print(f"needle arc: {np.degrees(span):.0f} deg in the image, {len(run)} px, ellipse axes "
          f"{w:.0f}x{h:.0f}, tail confidence {conf:.2f} -> {a.out} ({'ok' if out['ok'] else 'NOT ok'})")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
