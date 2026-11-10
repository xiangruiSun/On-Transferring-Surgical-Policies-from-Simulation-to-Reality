"""Held-needle tip + tip direction from the WHOLE arc in both images (2026-10-30, replaces tipmeas' 8 mm NCC quadratic, whose
direction jumped 20+ deg between translation-only frames).

right_rect (= physical left, reference) and left_rect are rectified: a 3D point lies on the same row in both.  The arc is detected
in each image (held_needle via tipmeas.detect_roi, ellipse model), densified along its ellipse, and every right-arc point gets its
disparity from where the left arc crosses the same row (no patch matching on the thin wire).  Near-horizontal arc parts are skipped.
A 3D circle (plane SVD + Kasa fit, radius reported) is fitted to the triangulated points; tip = circle point that projects closest to
the right-image tip; direction = circle tangent at the tip, pointing out of the arc.  No robot access.
"""
import os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261106/S1/code"))
import tipmeas
from rig import FX, CX, CY, FB


def dense_arc(r, n=240):
    """points on the detected ellipse between the polyline ends (grasp end -> tip), n samples."""
    (cx, cy), (w, h), ang = r["ellipse"]["centre"], r["ellipse"]["axes"], r["ellipse"]["angle_deg"]
    t = np.deg2rad(ang); Rm = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    poly = np.array(r["polyline_px"]); q = (poly - [cx, cy]) @ Rm; ph = np.unwrap(np.arctan2(q[:, 1] / (h / 2), q[:, 0] / (w / 2)))
    phs = np.linspace(ph[0], ph[-1], n)
    return (Rm @ np.vstack([w / 2 * np.cos(phs), h / 2 * np.sin(phs)])).T + [cx, cy]


def row_crossings(curve, v):
    dv = curve[:, 1] - v; s = np.nonzero(np.sign(dv[:-1]) * np.sign(dv[1:]) <= 0)[0]; out = []
    for k in s:
        a, b = dv[k], dv[k + 1]
        f = 0.5 if a == b else a / (a - b); out.append(curve[k, 0] + f * (curve[k + 1, 0] - curve[k, 0]))
    return out


def measure(R, L, z_hint_mm, roi_uv, roi_px=300, jaw_px=None, min_slope=0.3):
    d0 = FB / z_hint_mm
    rR = tipmeas.detect_roi(R, z_hint_mm, roi_uv, roi_px, jaw_px)
    if rR is None: return None, "no arc in the right image"
    rL = tipmeas.detect_roi(L, z_hint_mm, (roi_uv[0] - d0, roi_uv[1]), roi_px, None if jaw_px is None else (jaw_px[0] - d0, jaw_px[1]))
    if rL is None: return None, "no arc in the left image"
    cR, cL = dense_arc(rR), dense_arc(rL)
    # left polyline orientation is irrelevant for row crossings
    seg = np.diff(cR, axis=0); seg = np.vstack([seg, seg[-1]]); slope = np.abs(seg[:, 1]) / (np.linalg.norm(seg, axis=1) + 1e-9)
    P, idx = [], []
    for i, (u, v) in enumerate(cR):
        if slope[i] < min_slope: continue
        us = [x for x in row_crossings(cL, v) if abs((u - x) - d0) < 60]
        if not us: continue
        uL = min(us, key=lambda x: abs((u - x) - d0)); d = u - uL; Z = FB / d
        P.append([(u - CX) * Z / FX, (v - CY) * Z / FX, Z]); idx.append(i)
    if len(P) < 30: return None, f"only {len(P)} arc points triangulated"
    P = np.array(P); idx = np.array(idx)
    keep = np.ones(len(P), bool)
    for _ in range(3):                                                    # robust plane + circle
        c0 = P[keep].mean(0); U, S, Vt = np.linalg.svd(P[keep] - c0); nrm = Vt[2]; e1, e2 = Vt[0], Vt[1]
        Q = np.c_[(P - c0) @ e1, (P - c0) @ e2]
        A = np.c_[2 * Q[keep], np.ones(keep.sum())]; bb = (Q[keep] ** 2).sum(1); sol = np.linalg.lstsq(A, bb, rcond=None)[0]
        cc = sol[:2]; rad = float(np.sqrt(sol[2] + cc @ cc))
        res = np.sqrt((np.linalg.norm(Q - cc, axis=1) - rad) ** 2 + ((P - c0) @ nrm) ** 2)
        keep = res < max(0.5, 2.5 * np.median(res[keep]))
    C = c0 + cc[0] * e1 + cc[1] * e2
    circ = lambda th: C + rad * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
    th = np.linspace(-np.pi, np.pi, 3600, endpoint=False); X = circ(th)
    uvX = np.c_[FX * X[:, 0] / X[:, 2] + CX, FX * X[:, 1] / X[:, 2] + CY]
    tip_uv = cR[-1]; k = int(np.argmin(np.linalg.norm(uvX - tip_uv, axis=1)))
    # theta of the observed arc points: direction of increasing arc toward the tip
    thP = np.arctan2(Q[keep][:, 1] - cc[1], Q[keep][:, 0] - cc[0]); i_keep = idx[keep]
    th_tip = th[k]; th_mid = thP[np.argsort(i_keep)][len(thP) // 2]
    sgn = 1.0 if np.sin(th_tip - th_mid) > 0 else -1.0
    tan = sgn * (-np.sin(th_tip) * e1 + np.cos(th_tip) * e2)
    return dict(tip_cam=X[k], dir_cam=tan / np.linalg.norm(tan), centre=C, normal=nrm, radius_mm=rad, n_points=int(keep.sum()),
                rms_mm=float(np.sqrt(np.mean(res[keep] ** 2))), tip_uv_err_px=float(np.linalg.norm(uvX[k] - tip_uv)),
                poly=rR["polyline_px"], poly_left=rL["polyline_px"], tip=dict(uv=tip_uv.tolist()), arc_R=cR.tolist()), "ok"


def measure2(R, L, z_hint_mm, roi_uv, roi_px=300, jaw_px=None, min_slope=0.3, band=40):
    """as measure(), but the left arc is not detected on its own: for every right-arc point the left thin-structure response is
    searched along the SAME row within d0 +- band px, and the disparity is tracked along the arc (continuity), then the circle fit."""
    sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261027/place/code")); import held_needle as hn
    d0 = FB / z_hint_mm
    rR = tipmeas.detect_roi(R, z_hint_mm, roi_uv, roi_px, jaw_px)
    if rR is None: return None, "no arc in the right image"
    cR = dense_arc(rR, 300); _, resp = hn.thin_mask(L); resp = cv2.GaussianBlur(resp.astype(np.float32), (5, 5), 1.2)
    seg = np.diff(cR, axis=0); seg = np.vstack([seg, seg[-1]]); slope = np.abs(seg[:, 1]) / (np.linalg.norm(seg, axis=1) + 1e-9)
    H, W = resp.shape; P, idx, ds = [], [], []
    order = np.argsort(np.abs(np.arange(len(cR)) - len(cR) // 2))          # start in the middle of the arc, then outward
    dprev = {}
    for i in order:
        u, v = cR[i]
        if slope[i] < min_slope: continue
        vi = int(round(v))
        if not (2 <= vi < H - 2): continue
        nb = [dprev[j] for j in (i - 1, i + 1, i - 2, i + 2) if j in dprev]
        dc = np.median(nb) if nb else d0; bw = 8 if nb else band
        lo, hi = int(u - dc - bw), int(u - dc + bw) + 1
        if lo < 1 or hi >= W - 1: continue
        row = resp[vi, lo:hi]
        if row.max() < 10: continue
        k = int(np.argmax(row)); x = lo + k
        if 0 < k < len(row) - 1:
            a_, b_, c_ = row[k - 1], row[k], row[k + 1]; den = a_ - 2 * b_ + c_
            if den < 0: x += 0.5 * (a_ - c_) / den
        d = u - x; dprev[i] = d; Z = FB / d
        P.append([(u - CX) * Z / FX, (v - CY) * Z / FX, Z]); idx.append(i)
    if len(P) < 30: return None, f"only {len(P)} arc points triangulated"
    P = np.array(P); idx = np.array(idx); o = np.argsort(idx); P, idx = P[o], idx[o]
    keep = np.ones(len(P), bool)
    for _ in range(4):
        c0 = P[keep].mean(0); U, S, Vt = np.linalg.svd(P[keep] - c0); nrm = Vt[2]; e1, e2 = Vt[0], Vt[1]
        Q = np.c_[(P - c0) @ e1, (P - c0) @ e2]
        A = np.c_[2 * Q[keep], np.ones(keep.sum())]; bb = (Q[keep] ** 2).sum(1); sol = np.linalg.lstsq(A, bb, rcond=None)[0]
        cc = sol[:2]; rad = float(np.sqrt(sol[2] + cc @ cc))
        res = np.sqrt((np.linalg.norm(Q - cc, axis=1) - rad) ** 2 + ((P - c0) @ nrm) ** 2)
        keep = res < max(0.4, 2.5 * np.median(res[keep]))
    C = c0 + cc[0] * e1 + cc[1] * e2
    th = np.linspace(-np.pi, np.pi, 3600, endpoint=False); X = C + rad * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
    uvX = np.c_[FX * X[:, 0] / X[:, 2] + CX, FX * X[:, 1] / X[:, 2] + CY]
    tip_uv = cR[-1]; k = int(np.argmin(np.linalg.norm(uvX - tip_uv, axis=1)))
    thP = np.arctan2(Q[keep][:, 1] - cc[1], Q[keep][:, 0] - cc[0]); th_mid = thP[len(thP) // 2]; th_tip = th[k]
    sgn = 1.0 if np.sin(th_tip - th_mid) > 0 else -1.0
    tan = sgn * (-np.sin(th_tip) * e1 + np.cos(th_tip) * e2)
    return dict(tip_cam=X[k], dir_cam=tan / np.linalg.norm(tan), centre=C, normal=nrm, radius_mm=rad, n_points=int(keep.sum()),
                rms_mm=float(np.sqrt(np.mean(res[keep] ** 2))), tip_uv_err_px=float(np.linalg.norm(uvX[k] - tip_uv)),
                poly=rR["polyline_px"], tip=dict(uv=tip_uv.tolist())), "ok"


def _basis(nrm):
    a = np.array([1.0, 0, 0]) if abs(nrm[0]) < 0.9 else np.array([0, 1.0, 0]); e1 = np.cross(nrm, a); e1 /= np.linalg.norm(e1); return e1, np.cross(nrm, e1)


def measure3(R, L, z_hint_mm, roi_uv, roi_px=300, jaw_px=None, radius_mm=9.96, n_samp=60):
    """model-based stereo: circle of KNOWN radius (A02 perception 9.96 / 9.91 mm), pose = centre C, normal (2 angles), arc start
    angle and span; projected arc must match the right-image arc (symmetric chamfer to the detected ellipse arc) and the left-image
    thin-structure distance transform.  The ellipse shape gives the plane tilt (up to a 2-fold ambiguity), stereo picks the branch
    and the depth.  Initialised from measure2 and from its tilt-mirrored normal; lowest cost wins."""
    from scipy.optimize import least_squares
    sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261027/place/code")); import held_needle as hn
    m2, why = measure2(R, L, z_hint_mm, roi_uv, roi_px, jaw_px)
    if m2 is None: return None, "init: " + why
    rR = tipmeas.detect_roi(R, z_hint_mm, roi_uv, roi_px, jaw_px); cR = dense_arc(rR, 200)
    mL, _ = hn.thin_mask(L); H, W = mL.shape
    d0 = FB / z_hint_mm; band = np.zeros((H, W), np.uint8)
    cv2.polylines(band, [np.round(cR - [d0, 0]).astype(np.int32)], False, 255, 2 * 60 + 1)
    mL = mL & (band > 0)
    DL = cv2.distanceTransform((~mL).astype(np.uint8), cv2.DIST_L2, 5)
    tip_uv = cR[-1]
    def proj2(P): return np.c_[FX * P[:, 0] / P[:, 2] + CX, FX * P[:, 1] / P[:, 2] + CY]
    def arc(x):
        C = x[:3]; nrm = np.array([np.sin(x[3]) * np.cos(x[4]), np.sin(x[3]) * np.sin(x[4]), np.cos(x[3])]); e1, e2 = _basis(nrm)
        th = x[5] + np.linspace(0, x[6], n_samp); return C + radius_mm * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2), nrm, e1, e2, th
    def dtL(uv):
        u = np.clip(uv[:, 0], 0, W - 1.001); v = np.clip(uv[:, 1], 0, H - 1.001); u0 = u.astype(int); v0 = v.astype(int); fu = u - u0; fv = v - v0
        return (DL[v0, u0] * (1 - fu) * (1 - fv) + DL[v0, u0 + 1] * fu * (1 - fv) + DL[v0 + 1, u0] * (1 - fu) * fv + DL[v0 + 1, u0 + 1] * fu * fv)
    def resid(x):
        P, *_ = arc(x); uvR = proj2(P); uvL = uvR - np.c_[FB / P[:, 2], np.zeros(len(P))]
        dRm = np.min(np.linalg.norm(uvR[:, None] - cR[None], axis=2), axis=1)       # model -> detected arc
        dRd = np.min(np.linalg.norm(cR[::4, None] - uvR[None], axis=2), axis=1)     # detected arc -> model (coverage)
        return np.r_[dRm, dRd, np.linalg.norm(uvR[-1] - tip_uv) * np.ones(5), 0.7 * dtL(uvL)]
    def init(nrm):
        e1, e2 = _basis(nrm); C = np.asarray(m2["centre"], float); Pt = np.asarray(m2["tip_cam"], float)
        ang = lambda X: np.arctan2((X - C) @ e2, (X - C) @ e1)
        # grasp end = triangulated point of the first right-arc sample: approximate by the ray of cR[0] at the centre depth
        g = np.r_[(cR[0][0] - CX) / FX, (cR[0][1] - CY) / FX, 1.0] * C[2]
        t0, t1 = ang(g), ang(Pt); span = (t1 - t0 + np.pi) % (2 * np.pi) - np.pi
        if abs(span) < np.radians(60): span = np.sign(span or 1) * np.radians(150)
        return np.r_[C, np.arccos(np.clip(nrm[2], -1, 1)), np.arctan2(nrm[1], nrm[0]), t0, span]
    n0 = np.asarray(m2["normal"], float); ray = np.asarray(m2["centre"], float) / np.linalg.norm(m2["centre"])
    n_mir = 2 * (n0 @ ray) * ray - n0                                                   # same image ellipse, other tilt branch
    best = None
    for nrm in (n0, -n0, n_mir, -n_mir):
        for flip in (1, -1):
            x0 = init(nrm); x0[6] *= flip
            try: s = least_squares(resid, x0, loss="soft_l1", f_scale=3.0, x_scale=np.r_[1, 1, 3, 0.1, 0.1, 0.1, 0.1], max_nfev=400)
            except Exception: continue
            cost = float(np.mean(np.abs(s.fun)))
            if best is None or cost < best[0]: best = (cost, s.x)
    cost, x = best; P, nrm, e1, e2, th = arc(x)
    tan = np.sign(x[6]) * (-np.sin(th[-1]) * e1 + np.cos(th[-1]) * e2); tan /= np.linalg.norm(tan)
    r = resid(x); nR = 2 * n_samp // 1
    return dict(tip_cam=P[-1], dir_cam=tan, centre=x[:3], normal=nrm, radius_mm=radius_mm, span_deg=float(np.degrees(abs(x[6]))), cost_px=cost,
                right_px=float(np.median(r[:n_samp])), left_px=float(np.median(r[-n_samp:]) / 0.7), poly=rR["polyline_px"], tip=dict(uv=tip_uv.tolist()),
                init=m2), "ok"
