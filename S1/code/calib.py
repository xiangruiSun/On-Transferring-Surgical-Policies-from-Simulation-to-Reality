#!/usr/bin/env python3
"""C (camera <-> ECM calibration) and D (pad plane by tip contact), 2026-10-28.  New code; R6 r6_calib.py is the model, not modified.

C  model as R6: X_cam(feature) = [p_mm, 1] @ M at ONE fixed orientation (the current one = reference), but p is the TCP point
   (cp + R_ref TCP) instead of cp, so that make_goal_json's ecm_from_cam() returns a grasp point (DEVIATION, logged).
   Feature: picked automatically by optical flow while retracting 3 mm along the shaft (calib_cam_cp.auto_feature, imported).
   Depth: right-image NCC for (u, v); left-image NCC only inside a +-4 px disparity band around the MODEL prediction, and FS
   disparity at the base and at the farthest grid point must agree with NCC within 3 px (the 10/25 failure: NCC locked onto the pad).
   Every grid point is cross-checked against the kinematic prediction (rough Jacobian, then the running affine): > 4 / 3 mm -> rejected.
   Gate: 4 held-out points each <= 3 mm; singular values of M[:3] within [0.8, 1.25].
D  4 tip contacts on flat pad spots chosen from FS (local plane rms), descent 0.5 mm steps along the shaft with the R6 relative
   contact test, tip = cp + R TCP at contact; plane by make_goal_json.make_pad (residual <= 1 mm, extent >= 15 mm).
"""
import json, os, subprocess, sys, time
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig import (FX, CX, CY, FB, TCP_MM, Guard, CaptureError, T_of, tip_of, geodesic_deg, LIM_NEAR, DEPLOY)

DEPLOY_TOOLS = os.path.expanduser("~/Desktop/surgicai_grasp_integration/surgic_ai_deployment_unpublished/deploy")
FS_ENV = dict(FS_REMOTE="FS_USER@FS_HOST",
              FS_RUN="cd ~/FoundationStereo && PYTHONPATH=~/FoundationStereo ~/miniconda3/envs/foundation_stereo/bin/python")


# ------------------------------------------------------------------------------------------------ helpers
def save_snap(io, d, k=6, extra=None):
    """k synced frames -> d/left_rect.png, right_rect.png (means) + state.json.  CaptureError propagates."""
    os.makedirs(d, exist_ok=True); b = io.frames(k)
    L = np.mean([x[0] for x in b], 0).round().astype(np.uint8); R = np.mean([x[1] for x in b], 0).round().astype(np.uint8)
    cv2.imwrite(d + "/left_rect.png", L); cv2.imwrite(d + "/right_rect.png", R)
    st = dict(t=time.time(), stamps=[x[2] for x in b], n_frames=len(b))
    for k_ in ("cp", "sp", "suj"):
        v = io.get(k_); st[k_] = None if v is None else v[1].tolist()
    for k_ in ("js", "jse", "spj", "jaw", "ecm"):
        v = io.get(k_); st[k_] = None if v is None else np.asarray(v[1]).tolist()
    st.update(extra or {}); json.dump(st, open(d + "/state.json", "w"), indent=1)
    return L, R, b


def run_fs(snapdirs, fake=False):
    """FoundationStereo on the 3090 for each snapshot dir -> <snap>/fs_disp.npy.  Any failure -> Guard."""
    if fake:
        for d in snapdirs:
            np.save(d + "/fs_disp.npy", np.load(d + "/fake_disp.npy")); json.dump(dict(fake=True), open(d + "/fs_host.json", "w"))
        return
    env = dict(os.environ, **FS_ENV)
    r = subprocess.run(["bash", DEPLOY_TOOLS + "/tools/remote_fs.sh", *snapdirs], env=env, capture_output=True, text=True, timeout=300)
    for d in snapdirs:
        if r.returncode != 0 or not os.path.exists(d + "/fs_disp.npy"):
            raise Guard(f"FS failed for {d}: rc {r.returncode} {r.stderr[-400:]}")


def match(img, tpl, cu, cv_, rad):
    th, tw = tpl.shape
    x0, y0 = int(max(cu - rad - tw / 2, 0)), int(max(cv_ - rad - th / 2, 0))
    x1, y1 = int(min(cu + rad + tw / 2, img.shape[1])), int(min(cv_ + rad + th / 2, img.shape[0]))
    if x1 - x0 < tw or y1 - y0 < th: return 0.0, float(cu), float(cv_)
    r = cv2.matchTemplate(img[y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED); _, mx, _, (lx, ly) = cv2.minMaxLoc(r)
    def sub(a, j):
        if 0 < j < len(a) - 1:
            p, q, s = a[j - 1], a[j], a[j + 1]; den = p - 2 * q + s
            return j + (0.5 * (p - s) / den if den < 0 else 0)
        return float(j)
    return float(mx), x0 + sub(r[ly], lx) + tw / 2 - 0.5, y0 + sub(r[:, lx], ly) + th / 2 - 0.5


def match_band(Lg, TL, u_pred_L, v, band):
    """left-image NCC restricted to |u - u_pred_L| <= band px, |dv| <= 2 px."""
    th, tw = TL.shape; x0 = int(round(u_pred_L - band - tw / 2)); y0 = int(round(v - 2 - th / 2))
    W = Lg[max(y0, 0):y0 + th + 4, max(x0, 0):x0 + tw + 2 * band]
    if W.shape[0] < th or W.shape[1] < tw: return 0.0, float(u_pred_L), float(v)
    r = cv2.matchTemplate(W, TL, cv2.TM_CCOEFF_NORMED); _, mx, _, (lx, ly) = cv2.minMaxLoc(r)
    def sub(a, j):
        if 0 < j < len(a) - 1:
            p, q, s = a[j - 1], a[j], a[j + 1]; den = p - 2 * q + s
            return j + (0.5 * (p - s) / den if den < 0 else 0)
        return float(j)
    return float(mx), max(x0, 0) + sub(r[ly], lx) + tw / 2 - 0.5, max(y0, 0) + sub(r[:, lx], ly) + th / 2 - 0.5


def tri(u, v, uL):
    d = u - uL
    if d <= 1: raise Guard(f"bad disparity {d:.1f}")
    return np.array([u - CX, v - CY, FX]) * (FB / d) / FX


def proj(X): return np.array([FX * X[0] / X[2] + CX, FX * X[1] / X[2] + CY])


def cam_points(disp):
    Z = FB / np.clip(disp, 1, None); H, W = disp.shape; v, u = np.mgrid[0:H, 0:W]
    return np.dstack([(u - CX) * Z / FX, (v - CY) * Z / FX, Z])


def fit_affine(P, X):
    A = np.c_[P, np.ones(len(P))]; M = np.linalg.lstsq(A, X, rcond=None)[0]
    return M, float(np.sqrt(((A @ M - X) ** 2).sum(1).mean()))


def rotation_from_affine(M):
    A = np.asarray(M)[:3].T; U, _, Vt = np.linalg.svd(A); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); return U @ D @ Vt


def cam2ecm_mm(X, M):
    return (np.asarray(X, float) - M[3]) @ np.linalg.inv(M[:3])


# ------------------------------------------------------------------------------------------------ C: calibration
class Tracker:
    def __init__(self, TR, TL):
        self.TR, self.TL = TR, TL

    def track(self, io, X_pred, rad=40, band=4, k=4):
        b = io.frames(k); rows = []
        uv = proj(X_pred); d_pred = FB / X_pred[2]
        for (Li, Ri, _) in b:
            Rg = cv2.cvtColor(Ri, cv2.COLOR_BGR2GRAY).astype(np.float32); Lg = cv2.cvtColor(Li, cv2.COLOR_BGR2GRAY).astype(np.float32)
            m1, u, v = match(Rg, self.TR, uv[0], uv[1], rad)
            m2, uL, vL = match_band(Lg, self.TL, u - d_pred, v, band)
            rows.append(dict(ncc=[m1, m2], uv=[u, v], uvL=[uL, vL]))
        good = [r for r in rows if min(r["ncc"]) >= 0.6]
        if len(good) < max(2, k // 2): return None, rows, b
        u = np.median([r["uv"][0] for r in good]); v = np.median([r["uv"][1] for r in good]); uL = np.median([r["uvL"][0] for r in good])
        return tri(u, v, uL), rows, b


def calibrate(io, mv, kin, out, log, fake=False, n_grid=16, n_test=4, seed=20261028, feature_override=None):
    if feature_override is not None and not fake: raise ValueError("feature_override is a FakeIO test hook; refused on the real robot")
    os.makedirs(out, exist_ok=True)
    frozen = mv.freeze(); json.dump(frozen, open(out + "/frozen_refs.json", "w"), indent=1); log(dict(ev="freeze", **frozen))
    Tb = mv.setpoint().copy(); Rref = Tb[:3, :3].copy(); qref = Rotation.from_matrix(Rref).as_quat()
    p_tcp = lambda T: tip_of(T) * 1e3
    mv.cube = (tip_of(Tb), 0.060)              # calibration volume: +-60 mm around the start tip (pad unknown -> near limits everywhere)
    # 1 base snapshot + FS
    L0, R0, _ = save_snap(io, out + "/base"); run_fs([out + "/base"], fake)
    disp = np.load(out + "/base/fs_disp.npy"); P = cam_points(disp)
    # 2 automatic feature: retract 3 mm along the shaft
    sys.path.insert(0, DEPLOY_TOOLS + "/tools"); from calib_cam_cp import auto_feature
    mv.translate(-Rref[:, 2] * 3e-3, tag="calib_retract")
    L1, R1, _ = save_snap(io, out + "/retracted")
    af = auto_feature(R0, R1, inset_px=30.0)
    mv.move_to(Tb, tag="calib_back")
    fu, fv = af["feature_px"] if feature_override is None else feature_override; H = 30
    vis = R0.copy(); cv2.circle(vis, tuple(int(x) for x in af["tip_px"]), 6, (0, 0, 255), 2); cv2.rectangle(vis, (int(fu - H), int(fv - H)), (int(fu + H), int(fv + H)), (0, 255, 0), 2)
    cv2.imwrite(out + "/auto_feature.png", vis)
    Rg = cv2.cvtColor(R0, cv2.COLOR_BGR2GRAY).astype(np.float32); Lg = cv2.cvtColor(L0, cv2.COLOR_BGR2GRAY).astype(np.float32)
    iu, iv = int(round(fu)), int(round(fv))
    if not (H + 5 < iu < Rg.shape[1] - H - 5 and H + 5 < iv < Rg.shape[0] - H - 5): raise Guard(f"auto feature too close to the border {fu:.0f},{fv:.0f}")
    TR = Rg[iv - H:iv + H + 1, iu - H:iu + H + 1].copy()
    d_fs = float(np.median(disp[iv - 3:iv + 4, iu - 3:iu + 4]))
    m2, uL, vL = match_band(Lg, TR, fu - d_fs, fv, 8)
    d_ncc = fu - uL
    log(dict(ev="feature", auto=dict((k, v) for k, v in af.items() if k != "_mask"), d_fs=d_fs, d_ncc=d_ncc, ncc_left=m2))
    if m2 < 0.6 or abs(d_ncc - d_fs) > 3.0: raise Guard(f"base feature: left NCC {m2:.2f}, NCC disparity {d_ncc:.1f} vs FS {d_fs:.1f} px (> 3 px)")
    iuL, ivL = int(round(uL)), int(round(vL)); TL = Lg[ivL - H:ivL + H + 1, iuL - H:iuL + H + 1].copy()
    cv2.imwrite(out + "/tpl_R.png", TR.astype(np.uint8)); cv2.imwrite(out + "/tpl_L.png", TL.astype(np.uint8))
    trk = Tracker(TR, TL); X0 = tri(fu, fv, uL)
    # 3 probe: +-2 mm along ECM x / y / z
    J = np.zeros((3, 3)); pts = [(p_tcp(Tb), X0, "base")]
    for i in range(3):
        Xs = []
        for s in (+1, -1):
            T = Tb.copy(); T[i, 3] += s * 2e-3; mv.move_to(T, tag=f"probe{i}{s:+d}")
            X, rows, _ = trk.track(io, X0, rad=50, band=5)
            if X is None: raise Guard(f"probe {i}{s:+d}: tracking failed {rows[:2]}")
            Xs.append(X); pts.append((p_tcp(mv.setpoint()), X, f"probe{i}{s:+d}"))
        J[:, i] = (Xs[0] - Xs[1]) / 4.0
        mv.move_to(Tb, tag="probe_back")
    sv = np.linalg.svd(J)[1]; log(dict(ev="probe_J", J=J.tolist(), sv=sv.tolist()))
    if sv.min() < 0.7 or sv.max() > 1.4 or np.linalg.det(J) <= 0: raise Guard(f"probe Jacobian implausible: sv {np.round(sv, 3).tolist()} det {np.linalg.det(J):.3f}")
    # 4 grid (camera-aligned offsets, kept >= 12 mm in front of the FS surface)
    rng = np.random.default_rng(seed); Ji = np.linalg.inv(J); grid = []; tries = 0
    while len(grid) < n_grid and tries < 4000:
        tries += 1; dX = np.r_[rng.uniform(-25, 25), rng.uniform(-10, 30), rng.uniform(-8, 4)]; X = X0 + dX; uv = proj(X)   # camera x, y (image down = toward the centre), depth
        if not (80 < uv[0] < 1220 and 80 < uv[1] < 944): continue
        u, v = int(uv[0]), int(uv[1]); Zs = P[max(v - 15, 0):v + 16, max(u - 15, 0):u + 16, 2]
        if Zs.size == 0 or float(np.min(Zs)) - X[2] < 12.0: continue
        dp = Ji @ dX; T = Tb.copy(); T[:3, 3] += dp * 1e-3
        try: mv.check_target(T)
        except Guard: continue
        grid.append(dict(dX=dX, T=T))
    if len(grid) < n_grid: raise Guard(f"only {len(grid)} safe grid points found")
    order = []; cur = np.zeros(3); left = list(range(n_grid))
    while left:
        j = min(left, key=lambda i: np.linalg.norm(grid[i]["dX"] - cur)); order.append(j); cur = grid[j]["dX"]; left.remove(j)
    test = sorted(rng.choice(n_grid, n_test, replace=False).tolist()); far = int(np.argmax([np.linalg.norm(g["dX"]) for g in grid]))
    json.dump(dict(grid_dX_cam_mm=[g["dX"].tolist() for g in grid], test_idx=test, order=order, seed=seed, far_idx=far), open(out + "/manifest.json", "w"), indent=1)
    acc = []; rej = 0; fs_check = None
    for j in order:
        mv.move_to(grid[j]["T"], tag=f"grid{j}")
        p = p_tcp(mv.setpoint())
        if len([a for a in acc if a[3] not in test]) >= 6:
            Mc, _ = fit_affine(np.array([a[1] for a in acc if a[3] not in test]), np.array([a[2] for a in acc if a[3] not in test]))
            Xp = np.r_[p, 1] @ Mc; tol = 3.0
        else:
            Xp = X0 + J @ (p - p_tcp(Tb)); tol = 4.0
        X, rows, b = trk.track(io, Xp)
        ok = X is not None and np.linalg.norm(X - Xp) <= tol
        log(dict(ev="grid_point", i=j, p_tcp_mm=p.tolist(), X=None if X is None else X.tolist(), X_pred=Xp.tolist(), ok=bool(ok), rows=rows))
        if j == far and ok:
            d = out + f"/fs_check_g{j}"; os.makedirs(d, exist_ok=True)
            cv2.imwrite(d + "/left_rect.png", b[0][0]); cv2.imwrite(d + "/right_rect.png", b[0][1])
            if fake: np.save(d + "/fake_disp.npy", np.full(b[0][1].shape[:2], FB / X[2], np.float32))
            run_fs([d], fake); dd = np.load(d + "/fs_disp.npy"); uv = proj(X); u, v = int(round(uv[0])), int(round(uv[1]))
            fs_check = dict(i=j, d_fs=float(np.median(dd[v - 3:v + 4, u - 3:u + 4])), d_ncc=float(FB / X[2])); log(dict(ev="fs_check", **fs_check))
            if abs(fs_check["d_fs"] - fs_check["d_ncc"]) > 3.0: raise Guard(f"FS check at grid {j}: FS {fs_check['d_fs']:.1f} vs NCC {fs_check['d_ncc']:.1f} px")
        if ok: acc.append((j, p, X, j)); rej = 0
        else:
            rej += 1
            if rej >= 3: raise Guard("3 consecutive grid points rejected by the kinematic cross-check / NCC")
    mv.move_to(Tb, tag="grid_back")
    tr = [a for a in acc if a[3] not in test]; te = [a for a in acc if a[3] in test]
    if len(tr) < 8 or len(te) < 3: raise Guard(f"too few accepted points: train {len(tr)} test {len(te)}")
    M, rms = fit_affine(np.array([a[1] for a in tr]), np.array([a[2] for a in tr]))
    terr = {int(a[0]): float(np.linalg.norm(np.r_[a[1], 1] @ M - a[2])) for a in te}; svM = np.linalg.svd(M[:3])[1]
    gate = bool(max(terr.values()) <= 3.0 and svM.min() >= 0.8 and svM.max() <= 1.25)
    cal = dict(M=M.tolist(), TCP_tool_mm=TCP_MM.tolist(), model="X_cam(feature) = [p_tcp_ecm_mm, 1] @ M; p_tcp = measured_cp + R_ref @ TCP (R6 used measured_cp; deviation 2026-10-28)",
               ref_orientation_xyzw=qref.tolist(), base_cp_m=Tb[:3, 3].tolist(), base_T=Tb.tolist(), train_rms_mm=rms, test_err_mm=terr, sv=svM.tolist(),
               n_train=len(tr), n_test=len(te), n_rejected=n_grid - len(acc), probe_J=J.tolist(), fs_check=fs_check, gate_pass=gate,
               feature=dict(uv_right=[fu, fv], disp_fs=d_fs, disp_ncc=d_ncc), frozen=frozen, created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(cal, open(out + "/calibration_r6.json", "w"), indent=1)
    log(dict(ev="calibration", **{k: v for k, v in cal.items() if k not in ("M", "base_T")}))
    if not gate: raise Guard(f"calibration gate failed: test errors {terr}, sv {np.round(svM, 3).tolist()}")
    return cal


# ------------------------------------------------------------------------------------------------ D: pad by tip contact
def pad_candidates(R0, P, M, n_pick=4, min_sep_mm=15.0):
    """flat, pink pad spots near the image centre from the base FS: local plane rms over a ~10 mm window."""
    lab = cv2.cvtColor(R0, cv2.COLOR_BGR2LAB).astype(np.float32)
    pink = (lab[..., 1] > 140) & (lab[..., 0] > 90)                 # a* high: the pink pad (not blue tape / wood / dark needle)
    clear = cv2.distanceTransform(pink.astype(np.uint8), cv2.DIST_L2, 5)   # px to the nearest non-pad pixel (needle, thread, tool, tape)
    cands = []
    for v in range(220, 880, 40):
        for u in range(260, 1100, 40):
            h = 80; win = P[v - h:v + h + 1:4, u - h:u + h + 1:4].reshape(-1, 3); pk = pink[v - h:v + h + 1:4, u - h:u + h + 1:4].mean()
            if pk < 0.97 or clear[v, u] < 100: continue      # >= 100 px (~6.7 mm; 120 px left only 3 spots 15 mm apart) from needle / thread / tool: the open jaw spans ~13 mm
            c = win.mean(0); _, S, Vt = np.linalg.svd(win - c); n = Vt[2]; n = n if n[2] < 0 else -n
            rms = float(S[2] / np.sqrt(len(win)))
            if rms > 0.4: continue
            cands.append(dict(uv=[u, v], X=c.tolist(), n_cam=n.tolist(), rms=rms, dc=float(np.hypot(u - 650, v - 560))))
    cands.sort(key=lambda c: c["dc"]); picked = []
    for c in cands:
        if all(np.linalg.norm(np.array(c["X"]) - np.array(p["X"])) >= min_sep_mm for p in picked): picked.append(c)
        if len(picked) == n_pick: break
    return picked, len(cands)


HIGH_MM = 35.0


def pad_touch(io, mv, kin, cal, out, log, fake=False, hover_mm=6.0):
    from make_goal_json import make_pad
    os.makedirs(out, exist_ok=True); M = np.array(cal["M"]); Rref = Rotation.from_quat(cal["ref_orientation_xyzw"]).as_matrix()
    Tb = np.array(cal["base_T"]); mv.cube = (tip_of(Tb), 0.060)
    if np.linalg.norm(mv.setpoint()[:3, 3] - Tb[:3, 3]) * 1e3 > 1.0: mv.move_to(Tb, tag="pad_to_base")
    base_dir = cal.get("_base_dir") or (os.path.dirname(out.rstrip("/")) + "/C_calib/base")
    R0 = cv2.imread(base_dir + "/right_rect.png"); P = cam_points(np.load(base_dir + "/fs_disp.npy"))
    picked, n_c = pad_candidates(R0, P, M)
    log(dict(ev="pad_candidates", n=n_c, picked=picked)); json.dump(dict(n_candidates=n_c, picked=picked), open(out + "/candidates.json", "w"), indent=1)
    if len(picked) < 4: raise Guard(f"only {len(picked)} flat pad spots >= 15 mm apart found")
    vis = R0.copy()
    for i, c in enumerate(picked): cv2.circle(vis, tuple(c["uv"]), 12, (0, 255, 0), 2); cv2.putText(vis, str(i), (c["uv"][0] + 14, c["uv"][1]), 0, 1.0, (0, 255, 0), 2)
    cv2.imwrite(out + "/touch_targets.png", vis)
    tips = []; rows_all = []
    for i, c in enumerate(picked):
        Xs = np.array(c["X"]); nc = np.array(c["n_cam"])               # toward the camera
        p_above = cam2ecm_mm(Xs + hover_mm * nc, M) * 1e-3              # TCP point hover_mm above the spot
        T = T_of(p_above - Rref @ (TCP_MM * 1e-3), Rref)
        p_high = cam2ecm_mm(Xs + HIGH_MM * nc, M) * 1e-3                   # 35 mm above the spot by the calibration (base is ~40 mm up)
        T_high = T_of(p_high - Rref @ (TCP_MM * 1e-3), Rref)
        mv.move_to(T_high, tag=f"pad{i}_high", max_total_mm=120.0)
        T = T_high                                                             # 2026-10-28 13:1x: search the whole way down from there (calibration height not trusted)
        z = Rref[:, 2]; n_ecm = rotation_from_affine(M).T @ nc; cosang = max(0.3, float(-(z @ n_ecm)))
        max_mm = min(60.0, HIGH_MM / cosang + 8.0)
        hit, rows = mv.descend_until_contact(z, max_mm, tag=f"pad{i}_descend")
        save_snap(io, out + f"/touch{i}", k=4)
        tip = tip_of(mv.measured()); qn = io.get("js")[1]
        log(dict(ev="pad_touch", i=i, contact_step=hit, tip_ecm=tip.tolist(), max_mm=max_mm, insertion_mm=float(qn[2] * 1e3), descended_mm=0.5 * len(rows),
                 predicted_descent_mm=HIGH_MM / cosang, tip_cam_model=(np.r_[tip * 1e3, 1] @ M).tolist(), spot_cam=c["X"])); rows_all.append(dict(i=i, contact_step=hit, rows=rows))
        S = mv.setpoint(); back = S.copy(); back[:3, 3] = T[:3, 3]
        mv.move_to(back, tag=f"pad{i}_retract", contact_ok=True)
        if hit is None: raise Guard(f"pad touch {i}: no contact within {max_mm:.1f} mm")
        tips.append(tip.tolist())
    mv.move_to(Tb, tag="pad_back", max_total_mm=120.0)
    json.dump(tips, open(out + "/tips.json", "w"), indent=1); json.dump(rows_all, open(out + "/touch_rows.json", "w"), indent=1)
    ref_goal = out + "/sign_ref_goal.json"; json.dump(dict(p_grasp_ecm=tip_of(Tb).tolist(), R_goal_ecm=Rref.tolist(), note="only used for the pad normal sign"), open(ref_goal, "w"))
    try:
        pad = make_pad(out + "/tips.json", ref_goal)
    except SystemExit as e:
        raise Guard(f"pad fit refused: {e}")
    pad["base_tip_height_mm"] = float((np.array(pad["normal"]) @ tip_of(Tb) + pad["d"]) * 1e3)
    json.dump(pad, open(out + "/pad.json", "w"), indent=1); log(dict(ev="pad", **pad))
    return pad
