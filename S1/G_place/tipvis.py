"""Needle tip of the held needle in the rectified stereo pair, searched near the predicted needle (grasp geometry + kinematics).

thin structures (10-27 held_needle.thin_mask: top-hat / black-hat on lightness) inside a band around the predicted arc;
every mask pixel gets the arc parameter of its nearest predicted arc point; the tip = median of the pixels in the last 3 %
of the parameter range that is still connected to the arc (largest parameter cluster).  Left image: same, then
triangulation (calib.tri; rows must agree within 3 px).
"""
import os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261027/place/code"))
sys.path.insert(0, os.path.expanduser("~/rl_live/sim2real_20261106/S1/code"))
from held_needle import thin_mask
from calib import tri, proj
from rig import FB


def tip_in_image(img, arc_uv, band_px=45, ext_px=60):
    """arc_uv: predicted needle polyline (grasp end -> tip), px.  Returns (tip_uv, info) or (None, why)."""
    m, _ = thin_mask(img)
    d = arc_uv[-1] - arc_uv[-2]; d /= np.linalg.norm(d)
    poly = np.vstack([arc_uv, arc_uv[-1] + d * np.linspace(5, ext_px, 12)[:, None]])     # allow the real tip beyond the prediction
    band = np.zeros(m.shape, np.uint8); cv2.polylines(band, [np.round(poly).astype(np.int32)], False, 1, 2 * band_px + 1)
    ys, xs = np.nonzero(m & band.astype(bool))
    if len(xs) < 40: return None, f"only {len(xs)} thin pixels near the predicted needle"
    P = np.c_[xs, ys].astype(float)
    seg = np.cumsum(np.r_[0, np.linalg.norm(np.diff(poly, axis=0), axis=1)])
    dense_t = np.linspace(0, seg[-1], 400); dense = np.c_[np.interp(dense_t, seg, poly[:, 0]), np.interp(dense_t, seg, poly[:, 1])]
    k = np.argmin(((P[:, None, :] - dense[None]) ** 2).sum(-1), axis=1); s = dense_t[k]
    # walk the arc parameter: keep the support contiguous from the grasp side (gaps <= 12 px), the tip is where it ends
    order = np.sort(s); gaps = np.diff(order); cut = np.nonzero(gaps > 12)[0]
    s_end = order[cut[0]] if len(cut) else order[-1]
    sel = (s <= s_end) & (s >= s_end - 6)
    tip = np.median(P[sel], 0)
    return tip, dict(n_px=int(len(xs)), s_end=float(s_end), s_pred_tip=float(seg[len(arc_uv) - 1]), beyond_pred_px=float(s_end - seg[len(arc_uv) - 1]))


def tip_3d(R, L, arc_cam_mm, M_unused=None):
    """arc_cam_mm: predicted needle points (camera mm, grasp end -> tip).  Returns (X_cam_mm, info) or (None, why)."""
    uvR = np.array([proj(p) for p in arc_cam_mm])
    tR, iR = tip_in_image(R, uvR)
    if tR is None: return None, "right: " + iR
    disp = FB / np.array([p[2] for p in arc_cam_mm])
    uvL = uvR - np.c_[disp, np.zeros(len(disp))]
    tL, iL = tip_in_image(L, uvL)
    if tL is None: return None, "left: " + iL
    if abs(tR[1] - tL[1]) > 3.0: return None, f"rows disagree {tR[1]:.1f} vs {tL[1]:.1f}"
    X = tri(tR[0], tR[1], tL[0])
    return X, dict(right=iR, left=iL, tip_R=tR.tolist(), tip_L=tL.tolist(), disp=float(tR[0] - tL[0]), disp_pred=float(disp[-1]))
