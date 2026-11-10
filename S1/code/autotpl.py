#!/usr/bin/env python3
"""Automatic jaw template for the R6 hover (replaces the operator's template clicks; user request 2026-10-28 13:2x).

Input: open-jaw stereo pair (open25), model pixels of cp (hinge) and TCP from the calibration, model TCP disparity.
  1. metal mask: low Lab chroma (pad is pink, wood yellow, tape blue) within 220 px of the model TCP, largest blobs
  2. jaw axis (image) = model hinge -> model TCP direction; pixels split by the side of the axis line through the hinge
  3. tip of each blade = mean of its pixels in the top 0.5 % along the axis; grasp point = midpoint of the two tips (R6 definition)
  4. hinge = intersection of the two blade principal lines; feature = hinge (solid metal)
  5. other view: NCC of 33x33 patches at each tip and at the feature over disparity model +-30 px, |dv| <= 3
Returns dict(feat_R, grasp_R, grasp_L, half=16, debug...) or raises ValueError (-> quality stop).

14:2x (after A10-A14): tips closer than MIN_TIP_SEP_PX (both found on one blade; valid A03-A08 165-181 px, A10-A14 13-52 px) or any
tip-step failure -> with grasp_prior (model TCP + tool-frame c_off prior from A06-A08) the grasp point is the prior ("fallback");
without a prior -> ValueError.  Fallback feature: the 33x33 patch nearest the grasp point that is >= MIN_METAL metal (A10/A13/A14 patches
were half wood: NCC 0.94 -> 0.56/0.40 after the 22 mm hover move).  Tips mode keeps the 13:4x feature rule (offline replay: applying the
patch rule there replaced the A06/A08 features, which had tracked online, by ones that did not)."""
import numpy as np
import cv2

MIN_TIP_SEP_PX = 100
MIN_METAL = 0.6   # 2026-11-04 user: relax (was 0.85; slotted blades never reach it)


def _ncc_disp(R, L, u, v, d0, span=30, h=16):
    tp = R[v - h:v + h + 1, u - h:u + h + 1]; best = None
    if tp.shape != (2 * h + 1, 2 * h + 1): raise ValueError("patch at the border")
    for d in range(int(d0 - span), int(d0 + span) + 1):
        W = L[v - h - 3:v + h + 4, u - d - h - 2:u - d + h + 3]
        if W.shape != (2 * h + 7, 2 * h + 5): continue
        r = cv2.matchTemplate(W, tp, cv2.TM_CCOEFF_NORMED); _, mx, _, (lx, ly) = cv2.minMaxLoc(r)
        if best is None or mx > best[0]: best = (float(mx), float(d - (lx - 2)), float(ly - 3))
    if best is None: raise ValueError("no stereo match")
    return best


def auto_template(Rbgr, Lbgr, cp_px, tcp_px, disp_model, chroma_max=14.0, grasp_prior=None, remove_thin=False, no_tips=False):
    """grasp_prior = dict(R=[u, v], L=[u, v]) of the predicted grasp point; enables the fallback (see module docstring)."""
    lab = cv2.cvtColor(Rbgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    H, W = chroma.shape; v, u = np.mgrid[0:H, 0:W]
    near = np.hypot(u - tcp_px[0], v - tcp_px[1]) < 220
    m = ((chroma < chroma_max) & (lab[..., 0] > 70) & near).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    if remove_thin:      # 2026-11-04 second pass: blades reflect the pink pad today (low-chroma mask finds ~7000 px, A03/A04) -> colour difference to the local pad, slots closed, thin needle removed
        _lab8 = cv2.cvtColor(Rbgr, cv2.COLOR_BGR2LAB); _bg = np.dstack([cv2.medianBlur(_lab8[..., i], 41) for i in range(3)]).astype(np.float32)
        _d = np.linalg.norm(_lab8.astype(np.float32) - _bg, axis=2); _s = cv2.morphologyEx((_d > 30).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        _o = cv2.morphologyEx(_s, cv2.MORPH_OPEN, np.ones((21, 21), np.uint8)); m = (_s & cv2.dilate(_o, np.ones((21, 21), np.uint8)) & (np.hypot(u - tcp_px[0], v - tcp_px[1]) < 400).astype(np.uint8))   # 400 px: the model TCP sits ~100 px beyond the real tips
    n, lab_cc, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: raise ValueError("no metal found near the model TCP")
    keep = [i for i in range(1, n) if st[i, 4] >= 0.15 * st[1:, 4].max()]
    sel = np.isin(lab_cc, keep)
    frac = cv2.boxFilter(sel.astype(np.float32), -1, (33, 33), borderType=cv2.BORDER_CONSTANT)   # metal fraction of the 33x33 patch
    try:
        if no_tips: raise ValueError("tips found but not trusted -> prior")   # 2026-11-06 (user OK): untrusted tips fall back to the prior
        tips, hinge, grasp, gL, stereo = _tips(Rbgr, Lbgr, sel, cp_px, tcp_px, disp_model)
        sep = float(np.linalg.norm(tips[0] - tips[1]))
        if sep < MIN_TIP_SEP_PX: raise ValueError(f"tips {sep:.0f} px apart < {MIN_TIP_SEP_PX} (both on one blade)")
        mode, why = "tips", ""
    except ValueError as e:
        if grasp_prior is None: raise
        mode, why = "fallback", str(e); tips, hinge, stereo = [], None, []
        grasp = np.asarray(grasp_prior["R"], float); gL = np.asarray(grasp_prior["L"], float)
    fu = fv = None
    if mode == "tips":
        # feature (13:4x): on the blade whose tip matched best, 30-110 px back from its tip along that blade (solid metal, same disparity
        # as the grasp point; the hinge was used before and its disparity lay outside the template-phase band: stereo NCC 0.11-0.44)
        for bi in np.argsort([-x[0] for x in stereo]):                 # best-matching blade first, then the other one
            tip_b = tips[bi]; dirb = tip_b - hinge; dirb /= max(np.linalg.norm(dirb), 1e-6)
            for back in (30, 40, 50, 60, 75, 90, 110):
                q = tip_b - back * dirb; qu, qv = int(round(q[0])), int(round(q[1]))
                if sel[max(qv - 4, 0):qv + 5, max(qu - 4, 0):qu + 5].mean() > 0.7: fu, fv = qu, qv; break   # 13:4x rule kept (replay: the patch rule lost A06/A08)
            if fu is not None: break
        if fu is None:                                                # 13:4x last resort: largest solid-metal spot within 150 px of the grasp point
            dist_ = cv2.distanceTransform(sel.astype(np.uint8), cv2.DIST_L2, 3); dist_[np.hypot(u - grasp[0], v - grasp[1]) > 150] = 0
            if dist_.max() < 5: raise ValueError("no solid metal near the jaw for the feature")
            fv, fu = np.unravel_index(int(np.argmax(dist_)), dist_.shape); fu, fv = int(fu), int(fv)
    else:                                                             # fallback mode: solid-metal patch nearest the grasp point
        dg = np.hypot(u - grasp[0], v - grasp[1]); ok = (frac >= MIN_METAL) & (dg <= 150)
        if not ok.any(): raise ValueError(f"no solid-metal patch (>= {MIN_METAL:.0%}) within 150 px of the grasp point" + (f" [{mode}: {why}]" if why else ""))
        fv, fu = np.unravel_index(int(np.argmin(np.where(ok, dg, np.inf))), dg.shape); fu, fv = int(fu), int(fv)
    return dict(feat_R=[fu, fv], grasp_R=[float(grasp[0]), float(grasp[1])], grasp_L=[float(gL[0]), float(gL[1])], half=16, mode=mode, fallback_why=why,
                debug=dict(tips_R=[t_.tolist() for t_ in tips], tip_stereo=stereo, hinge_est=None if hinge is None else hinge.tolist(), n_metal_px=int(sel.sum()),
                           feat_metal_frac=float(frac[fv, fu]), axis=None))


def auto_template_checked(Rbgr, Lbgr, cp_px, tcp_px, disp_model, grasp_prior, trusted):
    """2026-11-04: pass 1 = original mask. If it fails, falls back, or trusted(ck) rejects its tips -> pass 2 with thin structures (needle) removed.
    Tips from pass 2 are used only if trusted; otherwise the prior-based result of whichever pass produced one; both failed -> ValueError."""
    p1, err1 = None, None
    try: p1 = auto_template(Rbgr, Lbgr, cp_px, tcp_px, disp_model, grasp_prior=grasp_prior); p1["pass"] = 1
    except ValueError as e: err1 = str(e)
    if p1 is not None and p1["mode"] == "tips" and trusted(p1): return p1
    p2 = None
    try: p2 = auto_template(Rbgr, Lbgr, cp_px, tcp_px, disp_model, grasp_prior=grasp_prior, remove_thin=True); p2["pass"] = 2
    except ValueError as e:
        if p1 is None or p1["mode"] == "tips": err1 = f"{err1}; pass 2: {e}"   # 2026-11-06: no early raise, the prior-based passes below still run
    if p2 is not None and p2["mode"] == "tips" and trusted(p2): return p2
    if p1 is not None and p1["mode"] != "tips": return p1
    if p2 is not None and p2["mode"] != "tips": return p2
    for rt in (False, True):     # 2026-11-06 (user OK, T19): tips found but untrusted -> prior-based result, as when the tip step fails
        try: p3 = auto_template(Rbgr, Lbgr, cp_px, tcp_px, disp_model, grasp_prior=grasp_prior, remove_thin=rt, no_tips=True); p3["pass"] = 3 if not rt else 4; return p3
        except ValueError: pass
    raise ValueError(f"no trusted tips and no prior-based result (pass 1: {err1})")


def _tips(Rbgr, Lbgr, sel, cp_px, tcp_px, disp_model):
    """steps 2-5 of the module docstring (unchanged from 13:4x): blade tips, hinge, grasp point (midpoint) and its left-image position."""
    a = np.asarray(tcp_px, float) - np.asarray(cp_px, float)
    if np.linalg.norm(a) < 5: raise ValueError("model axis degenerate")
    a /= np.linalg.norm(a); nrm = np.array([-a[1], a[0]])
    ys, xs = np.nonzero(sel); P = np.c_[xs, ys].astype(float) - np.asarray(cp_px, float)
    s = P @ a; t = P @ nrm
    dist = s >= np.percentile(s, 55)                                   # distal part of the metal (blades), model used for the direction only
    td = t[dist]; c_lo, c_hi = np.percentile(td, 20), np.percentile(td, 80)
    for _ in range(20):                                                # 1-D 2-means on the perpendicular coordinate
        lab_ = np.abs(td - c_lo) < np.abs(td - c_hi); c_lo, c_hi = td[lab_].mean(), td[~lab_].mean()
    if abs(c_hi - c_lo) < 15: raise ValueError(f"blades not separated ({abs(c_hi - c_lo):.0f} px)")
    split = (c_lo + c_hi) / 2
    tips, lines = [], []
    for side in (+1, -1):
        k = dist & ((t - split) * side > 0) & (np.abs(t - (c_hi if side > 0 else c_lo)) < 0.6 * abs(c_hi - c_lo))
        if k.sum() < 200: raise ValueError(f"blade on side {side} not found ({int(k.sum())} px)")
        sk = s[k]; far = sk >= np.percentile(sk, 99.5)
        tips.append((P[k][far].mean(0)) + np.asarray(cp_px, float))
        Q = P[k] + np.asarray(cp_px, float); c = Q.mean(0); _, _, Vt = np.linalg.svd(Q - c); lines.append((c, Vt[0]))
    (c1, d1), (c2, d2) = lines
    A = np.c_[d1, -d2]
    if abs(np.linalg.det(A)) < 1e-3: hinge = np.asarray(cp_px, float)
    else: l = np.linalg.solve(A, c2 - c1); hinge = c1 + l[0] * d1
    grasp = (tips[0] + tips[1]) / 2
    if np.linalg.norm(grasp - np.asarray(tcp_px)) > 250: raise ValueError("grasp point far from the model TCP")
    Rg = cv2.cvtColor(Rbgr, cv2.COLOR_BGR2GRAY).astype(np.float32); Lg = cv2.cvtColor(Lbgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    stereo = []
    for tp_ in tips:
        mx, d, dv = _ncc_disp(Rg, Lg, int(round(tp_[0])), int(round(tp_[1])), disp_model, span=15); stereo.append((mx, d, dv))
    if min(x[0] for x in stereo) < 0.5: raise ValueError(f"tip stereo NCC too low {[round(x[0], 2) for x in stereo]}")
    if abs(stereo[0][1] - stereo[1][1]) > 6: raise ValueError(f"tip disparities disagree {stereo[0][1]:.0f} vs {stereo[1][1]:.0f} px (A03: 108 vs 71)")
    gL = np.mean([[tp_[0] - x[1], tp_[1] + x[2]] for tp_, x in zip(tips, stereo)], 0)
    return tips, hinge, grasp, gL, stereo


def overlay(Rbgr, res):
    o = Rbgr.copy()
    for t_ in res["debug"]["tips_R"]: cv2.circle(o, (int(t_[0]), int(t_[1])), 6, (255, 0, 0), 2)
    cv2.circle(o, tuple(res["feat_R"]), 9, (0, 255, 255), 2); cv2.circle(o, (int(res["grasp_R"][0]), int(res["grasp_R"][1])), 6, (0, 0, 255), 2)
    cv2.putText(o, f"auto template [{res.get('mode', 'tips')}]: yellow feature, red grasp, blue tips", (10, 30), 0, 0.8, (255, 255, 255), 2); return o
