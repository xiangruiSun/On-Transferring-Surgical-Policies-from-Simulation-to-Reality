"""Measured needle tip (camera mm) of the held needle: right image = 10-27 held_needle.detect (thin-structure arc, tip = end
away from the instrument); left image = NCC of a 25x25 patch at the right tip along the same row (|dv| <= 2 px) over the
disparity range of the site depth +-25 px; triangulation calib.tri.  Also a point 2.5 mm back along the arc for the tip
direction.  No robot access."""
import os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261027/place/code"))
sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261106/S1/code"))
import held_needle as hn
from calib import tri
from rig import FB, FX


def match_left(Rg, Lg, u, v, d0, span=25, h=12):
    """NCC along the row (|dv| <= 2): best disparity with a parabolic sub-pixel refinement over the disparity scores."""
    tp = Rg[v - h:v + h + 1, u - h:u + h + 1]
    if tp.shape != (2 * h + 1, 2 * h + 1): return None
    ds, sc, dvs = [], [], []
    for d in range(int(d0 - span), int(d0 + span) + 1):
        W = Lg[v - h - 2:v + h + 3, u - d - h:u - d + h + 1]
        if W.shape != (2 * h + 5, 2 * h + 1): continue
        r = cv2.matchTemplate(W, tp, cv2.TM_CCOEFF_NORMED)[:, 0]; k = int(np.argmax(r)); ds.append(d); sc.append(float(r[k])); dvs.append(k - 2)
    if len(sc) < 3: return None
    i = int(np.argmax(sc)); d = float(ds[i])
    if 0 < i < len(sc) - 1:
        a, b, c = sc[i - 1], sc[i], sc[i + 1]; den = a - 2 * b + c
        if den < 0: d += 0.5 * (a - c) / den
    return (sc[i], d, float(dvs[i]))


def detect_roi(R, z_hint_mm, roi_uv, roi_px, jaw_px=None):
    """held_needle.detect restricted to thin pixels within roi_px of roi_uv (the site): it otherwise locks onto instrument texture."""
    orig = hn.thin_mask
    def masked(img, k=15, thr=18):
        m, resp = orig(img, k, thr); H, W = m.shape; yy, xx = np.mgrid[0:H, 0:W]
        return m & ((xx - roi_uv[0]) ** 2 + (yy - roi_uv[1]) ** 2 <= roi_px ** 2), resp
    hn.thin_mask = masked
    try:
        best = None
        for seed in range(5):
            r = hn.detect(R, z_hint_mm, jaw_px=jaw_px, seed=seed)
            if r is None: continue
            tip = np.array(r["polyline_px"][-1]); dist = float(np.linalg.norm(tip - np.asarray(roi_uv)))
            if best is None or r["n_inliers"] > best[0]["n_inliers"]: best = (r, dist)
        return None if best is None else best[0]
    finally:
        hn.thin_mask = orig


def measure(R, L, z_hint_mm, back_mm=4.0, roi_uv=None, roi_px=260, span=15, jaw_px=None):
    r = hn.detect(R, z_hint_mm, jaw_px) if roi_uv is None else detect_roi(R, z_hint_mm, roi_uv, roi_px, jaw_px)
    if r is None: return None, "no held needle found in the right image"
    poly = np.array(r["polyline_px"]); tip = poly[-1]
    Rg = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY).astype(np.float32); Lg = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d0 = FB / z_hint_mm; out = {}
    seg = np.cumsum(np.r_[0, np.linalg.norm(np.diff(poly[::-1], axis=0), axis=1)])
    at = lambda mm: np.array([np.interp(mm * FX / z_hint_mm, seg, poly[::-1][:, 0]), np.interp(mm * FX / z_hint_mm, seg, poly[::-1][:, 1])])
    pts, S_mm = [], []
    for k, mm in enumerate(np.linspace(0.0, 8.0, 9)):
        p = tip if k == 0 else at(mm); u, v = int(round(p[0])), int(round(p[1])); m = match_left(Rg, Lg, u, v, d0, span=span)
        if k == 0 and (m is None or m[0] < 0.6): return None, f"tip: left match {None if m is None else round(m[0], 2)} < 0.6"
        if m is None or m[0] < 0.6: continue
        X = tri(p[0], p[1], p[0] - m[1]); pts.append(X); S_mm.append(mm)
        if k == 0: out["tip"] = dict(uv=[float(p[0]), float(p[1])], ncc=m[0], disp=m[1], dv=m[2], X=X.tolist())
    if len(pts) < 6: return None, f"only {len(pts)} arc points triangulated"
    P = np.array(pts)
    # tangent at the tip: quadratic X(s) = X0 + s t + s^2 k fitted to the triangulated points over arc length s (mm, image-arc spacing)
    S = np.array(S_mm); A_ = np.c_[np.ones(len(S)), S, S ** 2]; coef = np.linalg.lstsq(A_, P, rcond=None)[0]
    d = -coef[1]; d /= np.linalg.norm(d)                             # s grows away from the tip -> tip direction = -dX/ds
    X = coef[0]; res = P - A_ @ coef
    out["fit"] = dict(n_points=len(P), rms_mm=float(np.sqrt((res ** 2).sum(1).mean())), curvature_1_per_mm=float(2 * np.linalg.norm(coef[2])))
    return dict(tip_cam=X, dir_cam=d, poly=poly.tolist(), arc_span_deg=r["arc_span_deg_image"], **out), "ok"
