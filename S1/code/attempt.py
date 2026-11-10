#!/usr/bin/env python3
"""One grasp attempt, 2026-10-28: perception -> approach -> wrist (C) -> open -> template -> goal/pad -> Reach start -> Reach ladder
dry / s0 / s0.5 / s1 (deploy_export/run_reach_dvrk.main, unchanged) -> handover error -> rotate to the goal orientation about the tip
-> hover visual correction (R6, <= 3 corrections) -> descent with contact search (0.5 mm) -> needle check -> slow close -> lift ->
hold 2 s -> put down -> open -> lift away -> back to the calibration base.

Error policy (user, 2026-10-28): hardware-safety events (Guard, CaptureError, Ctrl-C, unexpected errors) end the chain at once with no
further command.  Everything else is a QualityStop: recorded, the arm retreats to the base with the guarded mover, the session continues.
Grasp-protocol internals use the R6 step sizes (frozen protocol); today's limits apply to the new segments only.
"""
import json, os, subprocess, sys, time, traceback
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig import (FX, CX, CY, FB, TCP_MM, PAD_MIN_MM, Guard, CaptureError, QualityStop, T_of, tip_of, geodesic_deg, LIM_NEAR, LIM_DESCENT, DEPLOY)
from calib import save_snap, run_fs, match, match_band, tri, proj, rotation_from_affine, cam2ecm_mm, DEPLOY_TOOLS

R6S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r6_local")   # 2026-10-30: local copy of r6_realrobot (perceive flatness 25 deg)
OPEN_RAD, CLOSE_RAD = 0.6, -0.25
# R6 frozen grasp protocol step sizes (r6_grasp.py, 2026-10-16); today's 2 mm / 5 mm limits apply only to the new segments
R6_TRANS = (3.0, 3.0)          # move_rel_cam / move_cp_abs sub-steps <= 3 mm, orientation fixed
R6_WRIST_PITCH = (3.0, 4.0)    # rot_tipfixed: wrist pitch/yaw <= 4 deg per step, tip compensation <= 3 mm
R6_WRIST_ROLL = (3.0, 6.0)     # rot_tipfixed: roll <= 6 deg per step
REACH_CONTINUE = ("pad clearance", "rotation error grew", "pitch limit", "consecutive command geodesic", "step rotation component", "step translation")
LADDER = [None, 0.0, 0.5, 1.0]
S_GRASP = 0.20        # grasp point along the arc from the tail (R6 default 0.5); user 13:5x: near the thread end, chosen 0.20 (~5 mm)
NEEDLE_STAYED_PX = 15.0      # 15:3x grasp check: needle polyline after the lift within 15 px of the snap0 one -> not grasped (offline A18 4 px, A08 41 px)
APPROACH_MM = 12.0   # 2026-10-30 user: grasp the needle where it lies (left image edge); template pose 12 mm above instead of 25
HANDOVER_MAX_MM, HANDOVER_MAX_DEG = 15.0, 30.0   # 17:0x: Reach end within this of H10 -> correction in place (A20-A33: 2.1-2.3 mm / 7.8-7.9 deg)
TRACK_GATE_MM = float(os.environ.get("HOVER_TRACK_GATE_MM", "6.0"))   # 2026-11-06 17:2x env override (user: adjust, scene fixed; T22 template NCC 0.13, hover IQR 8-12 mm). Unset = 6.0 = unchanged. Earlier note: 14:5x: hover checks use the camera grasp point only within 4 mm of the kinematic reference (offline: init 0.2-1 mm; after moving A08 3.6-15.7, A11/A12/A16 11-20)
C_OFF_SHIFT_MAX_MM = 16.0     # 14:4x: largest c_off compensation of the hover start pose (A04-A15 c_off 6.3-8.6 mm; A03 12.4 mm with bad stereo)
GRASP_TOOL_MM = np.array([5.98, 1.03, -5.13])       # 16:3x (user '改吧'): aim point = where the needle sat in the jaw at closing in A08 (the one grasp),
                                                     # p* - R2-E TCP in the tool frame; A18-A25 (missed) had [3.47, -0.93, +1.2..2.2]. Used in both template modes.
AIM_BACK_MM = 1.0   # 2026-11-04
JAW_DQ_MIN, JAW_DEFF_MAX = 0.0015, -0.015   # 2026-11-04: grasped A07 +0.0050/-0.053, A08 +0.0033/-0.036, 10-28 about +0.004/-0.04; flicked A05 -0.0004/+0.005
VERTICAL_DESCENT = os.environ.get("VERTICAL_DESCENT", "1") == "1"   # 2026-11-04: hover, descent, lift and put down along the needle-site pad normal instead of the tilted jaw axis (orientation unchanged); "0" = old path
VERTICAL_GRASP = os.environ.get("VERTICAL_GRASP", "0") == "1"   # 2026-11-08 user ("都改"): jaw axis along the phantom normal at the grasp (opt-in; unset = frozen path)
# 2026-11-04 jaw midpoint - kinematic TCP in the tool frame = ROLL_A - (R^T Rref)[:, :2] ROLL_E_XY: the hinge-pin anchor (calibration) sits 4.7 mm off the tool axis, so the error turns with the roll.
# Fit on A01/03/05/08/09/10/11 (A_check/fit_roll.py); image error on A02/04/07, never fitted: 20/14/5 px, fixed value 81/49/18 px. Beyond about -22..+65 deg roll only A11 (-78 deg) supports it.
ROLL_A_MM = np.array([5.19, 0.51, -7.15]); ROLL_E_XY_MM = np.array([1.59, 1.32])   # 2026-11-08 refit after the recalibration (PSM1 SUJ moved; A_check/fit_roll_1108.py, LOO RMS 16 px vs 42 px with the 1106b values [3.87,0.18,-6.53]/[2.11,0.90]).   # 2026-11-06 17:1x refit after the recalibration (ECM moved; A_check/fit_roll_1106b.py, LOO 8-28 px at ~31 px/mm); 12:40 values were [5.11,0.40,-3.77]/[2.86,0.81]
CLICK_GATE_MM = 4.0   # against the roll model: fit residuals <= 1.5 mm; false tips of A02 about 20 mm off
OFF_TOOL_PRIOR_MM = np.array([-0.14, 0.04, -6.67])   # 2026-11-04 today's value (was [3.19, -0.64, -7.87]);   # 14:2x: median tip_off_tool of the valid auto templates A06-A08 (spread 1.3/0.3/1.6 mm); grasp prior for the autotpl fallback


def jaw_holds(dq, deff):
    """2026-11-04: jaw evidence vs the empty close at the base: something between the blades keeps the jaw open and raises the closing effort. Missing data -> no veto."""
    if dq is None or deff is None: return True
    return bool(dq >= JAW_DQ_MIN and deff <= JAW_DEFF_MAX)


def rodrigues(axis, ang):
    return Rotation.from_rotvec(np.asarray(axis, float) / np.linalg.norm(axis) * ang).as_matrix()


class Attempt:
    def __init__(self, io, mv, kin, cal, pad_path, tdir, placement, cell, log, fake=False, gui=True, fake_world=None):
        self.io, self.mv, self.kin, self.cal, self.pad_path, self.T = io, mv, kin, cal, pad_path, tdir
        if fake != (fake_world is not None): raise ValueError("fake_world is a FakeIO test hook: it requires fake=True and vice versa")
        if not fake and not gui: raise ValueError("gui=False is a FakeIO test hook; the real robot always uses the click windows")
        self.placement, self.cell, self.log, self.fake, self.gui, self.fw = placement, cell, log, fake, gui, fake_world
        self.M = np.array(cal["M"]); self.A3 = self.M[:3]; self.Rcam = rotation_from_affine(self.M)
        self.Rref = Rotation.from_quat(cal["ref_orientation_xyzw"]).as_matrix(); self.Tbase = np.array(cal["base_T"])
        self.S = dict(placement=placement, cell=cell["placement"] if cell else None, phases=[], t_start=time.time())
        pad = json.load(open(pad_path)); self.mv.pad = (np.array(pad["normal"]), float(pad["d"]))

    # ------------------------------------------------------------------ utils
    def save_state(self): json.dump(self.S, open(self.T + "/trial_state.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    def phase(self, name, **kw):
        self.S["phases"].append(name); self.S[name] = kw; self.log(dict(ev="phase", name=name, **kw)); self.save_state(); print(f"[{time.strftime('%T')}] {name}", {k: v for k, v in kw.items() if not isinstance(v, (list, dict))}, flush=True)
    def tip_cam(self, T=None):
        T = self.mv.setpoint() if T is None else T; return np.r_[tip_of(T) * 1e3, 1] @ self.M
    def cam2ecm(self, X): return cam2ecm_mm(X, self.M) * 1e-3
    def dcam2ecm(self, d): return np.asarray(d) @ np.linalg.inv(self.A3) * 1e-3
    def click_tool_model(self, R):
        """expected jaw midpoint - kinematic TCP (tool frame, mm) at orientation R."""
        return ROLL_A_MM - (R.T @ self.Rref)[:, :2] @ ROLL_E_XY_MM

    def down_axis(self, R=None):
        """unit ECM axis of the final approach: pad normal (into the pad) with VERTICAL_DESCENT, otherwise the jaw axis of R (default: current setpoint)."""
        if VERTICAL_DESCENT and getattr(self, "n", None) is not None:
            a = self.dcam2ecm(-np.asarray(self.n)); return a / np.linalg.norm(a)
        return (self.mv.setpoint()[:3, :3] if R is None else R)[:, 2]
    def save_frames(self, name, k=10, all_frames=False):
        d = f"{self.T}/{name}"; os.makedirs(d, exist_ok=True); b = self.io.frames(k)
        cv2.imwrite(d + "/right_rect.png", np.mean([x[1] for x in b], 0).round().astype(np.uint8)); cv2.imwrite(d + "/left_rect.png", np.mean([x[0] for x in b], 0).round().astype(np.uint8))
        if all_frames:
            for i, x in enumerate(b): cv2.imwrite(f"{d}/R{i:02d}.png", x[1]); cv2.imwrite(f"{d}/L{i:02d}.png", x[0])
        st = {k_: (None if self.io.get(k_) is None else np.asarray(self.io.get(k_)[1]).tolist()) for k_ in ("cp", "sp", "js", "jse", "jaw", "jaw_eff", "ecm")}
        st["t"] = time.time(); json.dump(st, open(d + "/state.json", "w"), indent=1)
        return b

    # ------------------------------------------------------------------ 1 perception
    def perceive(self):
        T = self.T
        if np.linalg.norm(self.mv.setpoint()[:3, 3] - self.Tbase[:3, 3]) * 1e3 > 1.5: raise Guard("arm is not at the calibration base: no attempt")
        self.retreat_ok = False
        self.mv.state_ok()
        # 15:3x empty-close reference at the base (jaw only): same close as the grasp (CLOSE_RAD, 10 x 0.3 s), sample, back to 0
        self.mv.jaw_to(CLOSE_RAD, n_steps=10, dt=0.3, tag="empty_close_ref"); self.jaw_ref = self.mv.jaw_sample()
        self.mv.jaw_to(0.0, n_steps=1, dt=1.8, tag="empty_close_ref_back"); self.S["jaw_empty_ref"] = self.jaw_ref
        self.log(dict(ev="jaw_empty_ref", **self.jaw_ref))
        json.dump(dict(t=time.time(), date=time.strftime("%Y-%m-%dT%H:%M:%S%z"), ecm_js=np.asarray(self.io.get("ecm")[1]).tolist()), open(T + "/start.json", "w"))
        for k in (0, 1):
            save_snap(self.io, f"{T}/snap{k}")
            if self.fake: self.fw.write_fake_disp(f"{T}/snap{k}")
        try: run_fs([f"{T}/snap0", f"{T}/snap1"], self.fake)
        except Guard as e: raise QualityStop(str(e))
        if self.fake:
            per = self.fw.fake_perception(T); json.dump(per, open(T + "/perception.json", "w"), indent=1)
        else:
            auto = None
            r = subprocess.run([sys.executable, R6S + "/needle_detect.py", "--snap", T + "/snap0", "--out", T + "/clicks_auto.json", "--radius-mm", "8.5", "11.0"], capture_output=True, text=True, timeout=120)   # 2026-10-30: R 9.5-9.9 measured; 6-13 let the ridges win (A07)
            if r.returncode == 0 and os.path.exists(T + "/clicks_auto.json"):
                ja = json.load(open(T + "/clicks_auto.json")); auto = np.array(ja["polyline_px"], float) if ja.get("polyline_px") else None
            img = cv2.imread(T + "/snap0/right_rect.png")
            if auto is not None:                                  # user 13:2x: no human step; auto polyline accepted, overlay saved for later review
                poly, how = np.asarray(auto, float).tolist(), "auto accepted without human review (user request 13:2x)"
            else:                                                 # 15:5x: no click window (A19 waited on it; user: no human steps) -> quality stop
                raise QualityStop(f"needle_detect: no accepted line (rc {r.returncode}): {(r.stdout + r.stderr).strip()[-200:]}")
            self.S["needle_clicks"] = dict(how=how, auto_available=auto is not None)
            if poly is None: raise QualityStop(f"needle clicks: {how}")
            json.dump(dict(polyline_px=poly, method=f"clickgui ({how}); auto = tools/needle_detect.py", t=time.time()), open(T + "/clicks.json", "w"), indent=1)
            ov = img.copy(); cv2.polylines(ov, [np.round(np.array(poly)).astype(np.int32)], False, (0, 255, 255), 2); cv2.circle(ov, tuple(np.round(poly[-1]).astype(int)), 7, (0, 0, 255), 2)
            cv2.putText(ov, how, (10, 30), 0, 0.9, (255, 255, 255), 2); cv2.imwrite(T + "/clicks_overlay.png", ov)
            r = subprocess.run([sys.executable, R6S + "/r6_perceive.py", "--trial", T, "--snaps", T + "/snap0", T + "/snap1", "--clicks", T + "/clicks.json", "--s", str(S_GRASP)], capture_output=True, text=True, timeout=180)
            open(T + "/perceive_stdout.txt", "w").write(r.stdout + "\n" + r.stderr)
            if r.returncode != 0 or not os.path.exists(T + "/perception.json"): raise QualityStop(f"r6_perceive failed: {r.stderr[-300:]}")
            per = json.load(open(T + "/perception.json"))
        if not per.get("ok"): raise QualityStop(f"perception not ok: {per.get('checks', per.get('fail', ''))}")
        self.per = per; self.p_star = np.array(per["p_star_mm"]); self.t_n = np.array(per["tangent"]); self.n = np.array(per["normal_toward_cam"])
        self.p_star_ecm = self.cam2ecm(self.p_star)
        self.mv.cube = None; self.mv.needle = self.p_star_ecm        # cube switched on when the arm arrives at H25 (base is outside it)
        h = self.mv.pad_dist_mm(self.p_star_ecm)
        self.phase("perception", p_star_cam=self.p_star, p_star_ecm=self.p_star_ecm, needle_height_above_pad_mm=h, R_mm=per["frames"][-1]["fit"]["R_mm"])

    # ------------------------------------------------------------------ 2 approach + wrist + open + template
    def approach(self):
        H25 = self.p_star + APPROACH_MM * self.n; p = self.cam2ecm(H25)          # 2026-10-30: APPROACH_MM (was 25): needle at the image edge, jaw left the view at 25 mm
        T = T_of(p - self.Rref @ (TCP_MM * 1e-3), self.Rref); bad = self.mv.precheck_path(T)
        if bad: raise QualityStop("approach path not reachable: " + bad[0])
        self.retreat_ok = True
        self.mv.move_to(T, tag="approach", max_total_mm=100.0, lim=LIM_NEAR)          # base -> H25: outside the cube, <= 2 mm / 3 deg (user 2026-10-28)
        self.mv.cube = (self.p_star_ecm, 0.040); self.mv.check_target(self.mv.setpoint())
        self.save_frames("approach", 6); self.phase("approach", H25_cam=H25, tip_height_mm=self.mv.pad_dist_mm(tip_of(self.mv.setpoint())))

    def angles(self, R=None):
        R = self.mv.setpoint()[:3, :3] if R is None else R; Rm = self.Rcam @ R
        ang = lambda v: float(np.degrees(np.arccos(min(1.0, abs(v @ self.t_n)))))
        z = Rm[:, 2]; tilt = float(np.degrees(np.arccos(min(1.0, abs(z @ self.n)))))
        return dict(x_tangent_deg=ang(Rm[:, 0]), y_tangent_deg=ang(Rm[:, 1]), angle_z_normal_deg=tilt, q=self.io.get("js")[1][:6].tolist())

    def wrist(self):
        info = dict(before=self.angles())
        for it in range(2):                                 # straighten (q4 = q5 = 0) about the tip; twice because the hinge moves
            q = np.array(self.io.get("js")[1][:6]); qs = q.copy(); qs[4] = qs[5] = 0.0
            Rs = self.kin.fk_ecm(qs, self.io.get("suj")[1])[:3, :3]
            self.mv.rotate_about_tip(Rs, tag=f"straighten{it}", lim=R6_WRIST_PITCH)
        info["straight"] = self.angles()
        S = self.mv.setpoint(); Rs = S[:3, :3]; a = Rs[:, 2]; best = None
        for dlt in np.radians(np.arange(-120, 120.1, 1.0)):
            y = (rodrigues(a, dlt) @ Rs)[:, 1]; ang = float(np.degrees(np.arccos(min(1, abs((self.Rcam @ y) @ self.t_n)))))
            key = (1, -abs(dlt)) if ang >= 85 else (0, ang)
            if best is None or key > best[0]: best = (key, dlt, ang)
        _, dlt, ang = best; Rg = rodrigues(a, dlt) @ Rs
        self.mv.rotate_about_tip(Rg, tag="roll", lim=R6_WRIST_ROLL)
        info.update(roll_deg=float(np.degrees(dlt)), pred_y_tangent_deg=ang, final=self.angles())
        if VERTICAL_GRASP:
            zv = self.dcam2ecm(-np.asarray(self.n)); zv /= np.linalg.norm(zv)                 # into the phantom at the needle site (perception normal)
            t = self.dcam2ecm(self.t_n); t = t - (t @ zv) * zv; t /= np.linalg.norm(t)          # needle tangent in ECM, in the surface plane
            Rc = self.mv.setpoint()[:3, :3]; cands = []
            for sg in (1.0, -1.0):                                                              # jaw opening axis y across the needle (as the frozen roll rule)
                y = sg * np.cross(zv, t); y /= np.linalg.norm(y); Rv = np.c_[np.cross(y, zv), y, zv]
                cands.append((geodesic_deg(Rc, Rv), Rv))
            dv, Rv = min(cands, key=lambda c: c[0])                                             # the one closer to the current (tilted) grasp
            S = self.mv.setpoint(); Tv = T_of(self.mv.tip(S) - Rv @ (TCP_MM * 1e-3 + self.mv.tip_off_tool), Rv)
            bad = self.mv.precheck_path(Tv)
            if bad: raise QualityStop("vertical grasp orientation not reachable: " + bad[0])
            self.mv.rotate_about_tip(Rv, tag="vertical", lim=R6_WRIST_PITCH)                  # tip fixed, <= 4 deg / 3 mm per step
            info.update(vertical_rot_deg=float(dv), final=self.angles())
        self.save_frames("after_wrist", 6); self.R_goal = self.mv.setpoint()[:3, :3].copy()
        self.phase("wrist", **info)

    def open_and_template(self):
        j = self.mv.jaw_to(OPEN_RAD, n_steps=1, dt=1.8, tag="open")                 # R6 'open': one command, 1.8 s
        self.T_tpl = self.mv.setpoint().copy()                                       # pose where the template is taken (H25, aligned)
        self.save_frames("open25", 10, all_frames=True)
        R = cv2.imread(self.T + "/open25/right_rect.png"); L = cv2.imread(self.T + "/open25/left_rect.png")
        Xt = self.tip_cam(); c = proj(Xt); dg = FB / Xt[2]
        if self.fake: ck = self.fw.fake_template_clicks(self)
        else:
            from autotpl import auto_template_checked, overlay
            cpx = proj(np.r_[self.mv.setpoint()[:3, 3] * 1e3, 1] @ self.M)
            Xp = Xt + (self.mv.setpoint()[:3, :3] @ self.click_tool_model(self.mv.setpoint()[:3, :3])) @ self.A3; gpR = proj(Xp)    # 14:2x grasp prior (model TCP + c_off prior)
            prior = dict(R=gpR.tolist(), L=[float(gpR[0] - FB / Xp[2]), float(gpR[1])])
            try:
                def _trusted(k):                                     # 2026-11-04: tips must agree with today's tool-frame value (thin needle taken for a blade otherwise)
                    z_ = FB / (k["grasp_R"][0] - k["grasp_L"][0]); X_ = np.array([(k["grasp_R"][0] - CX) * z_ / FX, (k["grasp_R"][1] - CY) * z_ / FX, z_])
                    return bool(np.linalg.norm(self.mv.setpoint()[:3, :3].T @ self.dcam2ecm(X_ - Xt) * 1e3 - self.click_tool_model(self.mv.setpoint()[:3, :3])) <= CLICK_GATE_MM)
                ck = auto_template_checked(R, L, cpx, c, dg, prior, _trusted)
                ck["how"] = f"auto (autotpl.py, user request 13:2x) mode {ck['mode']}"; ck["grasp_prior"] = prior
                cv2.imwrite(self.T + "/template_auto_overlay.png", overlay(R, ck)); ck = {k: v for k, v in ck.items()}
            except ValueError as e:                                  # 13:5x: never wait for a person (user: no human steps)
                raise QualityStop(f"auto template failed: {e}")
        json.dump(ck, open(self.T + "/template_clicks.json", "w"), indent=1)
        ov = R.copy(); cv2.circle(ov, tuple(ck["feat_R"]), 8, (0, 255, 255), 2); cv2.circle(ov, tuple(int(x) for x in ck["grasp_R"]), 6, (0, 0, 255), 2); cv2.imwrite(self.T + "/template_overlay_R.png", ov)
        ov = L.copy(); cv2.circle(ov, tuple(int(x) for x in ck["grasp_L"]), 6, (0, 0, 255), 2); cv2.imwrite(self.T + "/template_overlay_L.png", ov)
        Rimg = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY).astype(np.float32); Limg = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hf = ck.get("half", 16); fu, fv = ck["feat_R"]; TR = Rimg[fv - hf:fv + hf + 1, fu - hf:fu + hf + 1]
        if TR.shape != (2 * hf + 1, 2 * hf + 1): raise QualityStop("template feature too close to the image border")
        dgc = ck["grasp_R"][0] - ck["grasp_L"][0]; best = None
        for dd in np.arange(dgc - 6, dgc + 6.01, 1):
            x = int(round(fu - dd)); W = Limg[fv - hf - 3:fv + hf + 4, x - hf - 3:x + hf + 4]
            if W.shape[0] < TR.shape[0] or W.shape[1] < TR.shape[1]: continue
            r = cv2.matchTemplate(W, TR, cv2.TM_CCOEFF_NORMED); _, mx, _, (lx, ly) = cv2.minMaxLoc(r)
            if best is None or mx > best[0]: best = (mx, x - 3 + lx, fv - 3 + ly)
        if best is None: raise QualityStop("template: no stereo match")
        if best[0] < 0.6: self.log(dict(ev="note", what=f"template stereo NCC {best[0]:.2f} < 0.6 (R6 has no gate here; continuing)"))
        mx, xl, yl = best; disp0 = fu - xl; TL = Limg[yl - hf:yl + hf + 1, xl - hf:xl + hf + 1]
        cv2.imwrite(self.T + "/tpl_R.png", TR.astype(np.uint8)); cv2.imwrite(self.T + "/tpl_L.png", TL.astype(np.uint8))
        self.TR, self.TL = TR.astype(np.float32), TL.astype(np.float32)
        Xf = tri(fu, fv, xl); gu, gv = ck["grasp_R"]; Xg = tri(gu, gv, ck["grasp_L"][0])
        R_now = self.mv.setpoint()[:3, :3]; c_meas = Xg - Xt
        # 2026-11-04: aim = measured midpoint of the jaw tips, 1 mm toward the hinge. GRASP_TOOL_MM is relative to the kinematic TCP, whose offset depends on the day's pin anchor
        # and reference roll (A01: 7 mm beside the jaw, needle pushed). A measurement > CLICK_GATE_MM from today's tool-frame value is not trusted (A02: "tips" on the needle, 14 mm).
        click_tool = R_now.T @ self.dcam2ecm(c_meas) * 1e3; click_model = self.click_tool_model(R_now); aim_tool = click_model + np.array([0.0, 0.0, -AIM_BACK_MM]); click_used = False
        if ck.get("mode") == "tips" and np.linalg.norm(click_tool - click_model) <= CLICK_GATE_MM: aim_tool = click_tool + np.array([0.0, 0.0, -AIM_BACK_MM]); click_used = True
        Xg = Xt + (R_now @ aim_tool) @ self.A3; self.log(dict(ev="note", what=f"aim tool {np.round(aim_tool, 2).tolist()} (click tool {np.round(click_tool, 2).tolist()}, model {np.round(click_model, 2).tolist()}, used {click_used}, mode {ck.get('mode')})"))
        self.tpl = dict(clicks=ck, disp0=float(disp0), stereo_ncc=float(mx), X_feat=Xf, X_grasp_click=Xg, delta_cam=Xg - Xf, tip_model_cam=Xt, c_off_cam=Xg - Xt,
                        c_off_cam_template=c_meas, aim_tool_mm=GRASP_TOOL_MM)
        off_ecm = self.dcam2ecm(self.tpl["c_off_cam"])
        self.mv.tip_off_tool = R_now.T @ off_ecm                      # option 1 (user 13:4x): pad / cube checks use the camera-corrected TCP
        self.tpl["tip_off_tool_mm"] = self.mv.tip_off_tool * 1e3
        self.phase("template", jaw_rad=j, **self.tpl)

    # ------------------------------------------------------------------ 3 goal + Reach
    def make_goal(self):
        from make_goal_json import make_goal
        q = Rotation.from_matrix(self.R_goal).as_quat()
        g = make_goal(os.path.basename(self.T), self.T + "/perception.json", self.cal["_path"], q.tolist(), self.tpl["c_off_cam"], allow_stale=False)
        g["source"]["c_off_note"] = ("c_off = clicked grasp point (template, at the goal orientation) - calibration model at the same pose, camera mm. "
                                     "Self-referential: the same kind of template measurement drives the hover correction. Needed because make_goal_json "
                                     "(ecm_from_cam) and the R6 grasp script (cam2ecm + Rref TCP) use different point conventions.")
        self.goal_path = self.T + "/goal.json"; json.dump(g, open(self.goal_path, "w"), indent=1)
        os.system(f"cp {self.pad_path} {self.T}/pad.json")
        from reach_real import goal_hinge
        self.p_grasp = np.array(g["p_grasp_ecm"]); self.hinge = goal_hinge(self.p_grasp, self.R_goal)
        self.mv.cube = (self.p_grasp, 0.040); self.mv.needle = self.p_grasp
        self.phase("goal", p_grasp_ecm=self.p_grasp, quat_xyzw=q, goal_vs_pstar_ecm_mm=float(np.linalg.norm(self.p_grasp - self.p_star_ecm) * 1e3), hinge_goal=self.hinge.p)

    def reach_start(self, cells):
        from surgicai_rl_deploy.frames import FrameBridge, Pose
        from surgicai_rl_deploy.contract import R6_TRAINED_GOAL_VEC7
        trained = Pose.from_vec7(R6_TRAINED_GOAL_VEC7).with_jaw(0.0); X = FrameBridge.rebase(self.hinge, trained)
        order = [self.cell] + [c for c in cells if c is not self.cell]; tried = []
        for c in order:
            s = X.to_robot(Pose.from_vec7(np.array(c["start_vec7_policy_frame"], float))); T = T_of(s.p, s.R)
            bad = []
            try: self.mv.check_target(T)
            except Guard as e: bad.append(str(e))
            dh = np.abs(T[:3, 3] - self.p_grasp) * 1e3
            if np.any(dh > 40.0): bad.append(f"hinge outside reach_real's +-40 mm cube: {np.round(dh, 1).tolist()}")
            n_, d_ = self.mv.pad; tip_rr = self.mv.tip(T)                  # 15:1x (user option 1): reach_real's pad checks now use the camera-corrected TCP: mirror it
            if min((n_ @ T[:3, 3] + d_), (n_ @ tip_rr + d_)) * 1e3 < PAD_MIN_MM: bad.append(f"reach_real preflight pad clearance (corrected TCP {(n_ @ tip_rr + d_) * 1e3:.1f} mm)")
            if not bad: bad = self.mv.precheck_path(T, n=24)
            tried.append(dict(cell=c["placement"], ok=not bad, why=bad[:2], start_hinge=T[:3, 3].tolist(), rot_to_goal_deg=geodesic_deg(T[:3, :3], self.R_goal)))
            self.log(dict(ev="reach_start_candidate", **tried[-1]))
            if not bad:
                self.mv.move_to(T, tag=f"to_start_{c['placement']}", max_total_mm=120.0)
                self.T_start = T; self.used_cell = c["placement"]
                self.phase("reach_start", cell=c["placement"], tried=tried, deviation=None if c is self.cell else "assigned cell unreachable; next reachable cell used")
                return
        self.T_start = self.mv.setpoint().copy(); self.used_cell = None
        self.phase("reach_start", cell=None, tried=tried, deviation="no realstart cell reachable -> Reach from the current (post-wrist) pose")

    def reach_ladder(self):
        import dvrk_robot, run_reach_dvrk
        mv = self.mv
        base_cls = getattr(dvrk_robot, "_ORIG_DvrkRobot", dvrk_robot.DvrkRobot); dvrk_robot._ORIG_DvrkRobot = base_cls
        class GuardedDvrkRobot(base_cls):                        # same robot; adds the day's guards around each servo
            def __init__(s, io, kin, **kw):
                super().__init__(io, kin, post_reached_s=0.0 if mv.fake else 0.4, **kw)
            def servo(s, T_cmd):
                try: mv.state_ok()
                except Guard as e: raise dvrk_robot.RobotAbort(str(e))
                super().servo(T_cmd)
                m = mv.measured(); e_m = float(np.linalg.norm(m[:3, 3] - T_cmd.p) * 1e3); r_m = geodesic_deg(m[:3, :3], T_cmd.R)
                js = mv._need("js"); spj = mv._need("spj"); je = js[:6] - spj[:6]
                s.events.append(dict(ev="guard_after_servo", meas_err_mm=e_m, meas_err_deg=r_m, joint_err=je.tolist()))
                if e_m > 2.0 or r_m > 2.0: raise dvrk_robot.RobotAbort(f"measured not at the command {e_m:.2f} mm / {r_m:.2f} deg")
                if abs(je[2]) * 1e3 > 0.5 or np.degrees(np.abs(np.r_[je[:2], je[3:6]])).max() > 1.0: raise dvrk_robot.RobotAbort("joint tracking error (unexpected contact?)")
        dvrk_robot.DvrkRobot = GuardedDvrkRobot                 # run_reach_dvrk.main imports DvrkRobot at call time
        patch_reach_pad(mv.tip_off_tool)                         # 15:1x (user option 1): pad clearance with the camera-corrected TCP
        out = self.T + "/reach"; os.makedirs(out, exist_ok=True); gid = os.path.basename(self.T); res = []; self.reach_note = None
        cwd = os.getcwd(); os.chdir(DEPLOY)
        try:
            for sc in LADDER:
                if sc in (0.5, 1.0): mv.move_to(self.T_start, tag=f"back_to_start_s{sc}", max_total_mm=120.0)
                argv = ["--goal", self.goal_path, "--pad", self.T + "/pad.json", "--outdir", out]
                if sc is not None: argv += ["--send", "--scale", str(sc)]
                try:
                    r = run_reach_dvrk.main(argv, io=self.io, kin=self.kin, confirm=lambda p: "SEND " + p.split("SEND ")[1].split(" to continue")[0].strip())
                except SystemExit as e:          # preflight blocking / ladder refusal: not a hardware event -> no Reach, go on to hover
                    res.append(dict(scale=sc, status=f"refused: {e}")); self.reach_note = f"run_reach_dvrk refused rung {sc} ({e}); continued to hover"; break
                last = r["rows"][-1] if r["rows"] else {}
                res.append(dict(scale=sc, status=r["status"], steps=r["steps"], e_mm=last.get("e_mm"), e_rpy_deg=last.get("e_rpy_deg"), log=max(os.listdir(out), key=lambda f: os.path.getmtime(os.path.join(out, f)))))
                self.log(dict(ev="reach_rung", **res[-1])); print("   reach", res[-1], flush=True)
                st = r["status"]
                if st in ("success", "max_steps", "no_progress"): continue
                why = st.replace("abort: ", "")
                if why.startswith("robot:") or "workspace cube" in why or "tracking error" in why or "operator" in why:
                    raise Guard(f"Reach rung {sc}: {st}")                  # hardware-safety class -> stop, nothing more
                self.reach_note = f"Reach rung {sc}: {st} -> remaining rungs skipped, continued to hover (user rule 2026-10-28)"; break
        finally:
            os.chdir(cwd)
        m = mv.measured(); Rg = self.R_goal
        h10 = self.p_grasp - Rg @ (TCP_MM * 1e-3) - self.down_axis(Rg) * 0.010
        self.handover = dict(to_hover10_mm=float(np.linalg.norm(m[:3, 3] - h10) * 1e3), to_hover10_deg=geodesic_deg(m[:3, :3], Rg),
                             to_reach_goal_mm=float(np.linalg.norm(m[:3, 3] - self.hinge.p) * 1e3), vec_ecm_mm=((m[:3, 3] - h10) * 1e3).tolist(), note=self.reach_note)
        self.save_frames("reach_end", 6); self.phase("reach", rungs=res, handover=self.handover, note=self.reach_note)

    # ------------------------------------------------------------------ 4 hover + descend + close + lift + put down (R6 frozen protocol)
    def track10(self, name, pred_uv):
        b = self.io.frames(5); Xs = []; rows = []
        def _proc(bb):
            for (Li, Ri, _) in bb:
                Rg = cv2.cvtColor(Ri, cv2.COLOR_BGR2GRAY).astype(np.float32); Lg = cv2.cvtColor(Li, cv2.COLOR_BGR2GRAY).astype(np.float32)
                m1, u, v = match(Rg, self.TR, pred_uv[0], pred_uv[1], 60); m2, uL, vL = match(Lg, self.TL, u - self.tpl["disp0"], v, 25)
                X = tri(u, v, uL) if u - uL > 1 else np.full(3, np.nan); rows.append(dict(ncc=[m1, m2], uv=[u, v], uvL=[uL, vL], X=X.tolist()))
                if m1 >= 0.5 and m2 >= 0.5 and np.all(np.isfinite(X)): Xs.append(X)
        _proc(b); X5 = np.array(Xs) if Xs else None
        iqr5 = float(np.linalg.norm(np.percentile(X5, 75, 0) - np.percentile(X5, 25, 0))) if X5 is not None and len(X5) == 5 else None
        if iqr5 is None or iqr5 >= 0.3:                 # R6: 5 frames suffice only when all valid and IQR < 0.3 mm
            b2 = self.io.frames(5); _proc(b2); b = b + b2
        d = f"{self.T}/{name}"; os.makedirs(d, exist_ok=True); cv2.imwrite(d + "/right_rect.png", b[0][1]); cv2.imwrite(d + "/left_rect.png", b[0][0])
        out = dict(n_valid=len(Xs), n_frames=len(b), rows=rows)
        if len(Xs) >= 5:
            Xs = np.array(Xs); med = np.median(Xs, 0); iqr = np.percentile(Xs, 75, 0) - np.percentile(Xs, 25, 0)
            out.update(X_feat_med=med, iqr_norm=float(np.linalg.norm(iqr)), X_grasp=med + self.tpl["delta_cam"])
        self.log(dict(ev="track10", name=name, **{k: v for k, v in out.items() if k != "rows"}, rows=rows)); return out

    def move_rel_cam(self, dcam, tag, mode=False, lim=R6_TRANS):
        return self.mv.translate(self.dcam2ecm(dcam), tag=tag, contact_ok=mode, lim=lim)

    def hover(self):
        # deviation 13:3x (user: continue at this needle; option 2): Reach is evaluated and logged, then the arm goes back to the
        # aligned template pose (H25) and the R6 hover starts there, as in R6.  New segment: lift 10 mm along the pad normal, then
        # straight to the template pose, today's limits (near: 2 mm / 3 deg).
        # 14:4x (A14/A15 'hover move too large' 27.2/25.5 mm > R6 25): the template pose put the kinematic TCP at H25, the measured grasp point
        # sat ~8 mm lower (c_off), so the hover move was ~23 mm.  Shift the pose by -c_off (<= C_OFF_SHIFT_MAX_MM) so the measured point is at H25.
        # 17:0x (user: '改吧'): when Reach ended near H10 the correction starts where Reach stopped (rotate to the aligned orientation about the
        # aim point, then the kinematic move to H10); no return to the template pose and no camera tracking in the hover.
        ho = getattr(self, "handover", None) or {}
        self.in_place = bool(ho.get("to_hover10_mm") is not None and ho["to_hover10_mm"] <= HANDOVER_MAX_MM and ho["to_hover10_deg"] <= HANDOVER_MAX_DEG)
        if self.in_place:
            self.mv.rotate_about_tip(self.R_goal, tag="hover_rotate", lim=R6_WRIST_PITCH)
            self.log(dict(ev="note", what=f"hover from the Reach end ({ho['to_hover10_mm']:.1f} mm / {ho['to_hover10_deg']:.1f} deg to H10)"))
        else:
            off = self.dcam2ecm(self.tpl["c_off_cam"]); n_off = float(np.linalg.norm(off) * 1e3)
            T_h = self.T_tpl.copy()
            if n_off <= C_OFF_SHIFT_MAX_MM: T_h[:3, 3] -= off
            self.log(dict(ev="note", what=f"Reach end not near H10 ({ho.get('to_hover10_mm')}) -> template pose; shifted by -c_off {n_off:.1f} mm" if n_off <= C_OFF_SHIFT_MAX_MM else f"c_off {n_off:.1f} mm > {C_OFF_SHIFT_MAX_MM} -> no shift"))
            self.mv.safe_goto(T_h, tag="to_template_pose")                           # 13:4x: lift >= 15 mm, rotate about the tip, translate
        R = self.mv.setpoint()[:3, :3]; zc = self.Rcam @ R[:, 2]; self.zc = zc; H10 = self.p_star + 10.0 * self.n if VERTICAL_DESCENT else self.p_star - 10.0 * zc
        pred = proj(self.tip_cam() + self.tpl["c_off_cam"] - self.tpl["delta_cam"])
        if self.in_place: self.save_frames("hover_init", 3); tr = dict(n_valid=0)
        else:
            tr = self.track10("hover_init", pred)
            if tr["n_valid"] < 5: self.log(dict(ev="note", what=f"hover init tracking {tr['n_valid']}/10 valid -> kinematic reference (2026-11-04 user: relax; was a quality stop)"))
        X, src, eg = self.grasp_estimate(tr)
        d = H10 - X
        if np.linalg.norm(d) > 25: self.phase("hover", stop_reason=f"hover move too large {np.linalg.norm(d):.1f} mm"); raise QualityStop(f"hover move too large {np.linalg.norm(d):.1f} mm")
        init = dict(source=src, cam_vs_kin_mm=eg)
        self.move_rel_cam(d, "hover_move"); pred = proj(self.tip_cam() + self.tpl["c_off_cam"] - self.tpl["delta_cam"])
        corr, applied, reason = [], [], None
        for it in range(4):
            if self.in_place: self.save_frames(f"hover_c{it}", 3); tr = dict(n_valid=0)
            else: tr = self.track10(f"hover_c{it}", pred)
            X, src, eg = self.grasp_estimate(tr)
            e = H10 - X; en = float(np.linalg.norm(e)); iqr = tr.get("iqr_norm") if src == "cam" else 0.0
            corr.append(dict(it=it, err=e, err_norm=en, iqr_norm=iqr, n_valid=tr["n_valid"], source=src, cam_vs_kin_mm=eg))
            print(f"   hover check {it}: |e| {en:.2f} mm IQR {iqr:.2f} source {src} (cam-kin {eg if eg is None else round(eg, 1)})", flush=True)
            if en < 0.5: reason = "err<0.5"; break
            if iqr > 0.5 and len(applied) >= 2 and applied[-1] < iqr and applied[-2] < iqr: reason = "two consecutive corrections < IQR"; break
            if len(applied) == 3: reason = "max 3 corrections"; break
            step = e if en <= 3 else 3 * e / en; self.move_rel_cam(step, f"hover_corr{it}"); applied.append(float(np.linalg.norm(step)))
            pred = proj(self.tip_cam() + self.tpl["c_off_cam"] - self.tpl["delta_cam"])
        last = corr[-1] if corr else None
        if last:
            e = np.asarray(last["err"]); e_n = float(e @ self.n); last["lateral_mm"] = float(np.linalg.norm(e - e_n * self.n)); last["height_mm"] = -e_n
        self.phase("hover", in_place=self.in_place, init=init, corrections=corr, applied_mm=applied, n_corrections=len(applied), stop_reason=reason, final=last, angles=self.angles())

    def grasp_estimate(self, tr):
        """14:5x (A16: tracked point 9-18 mm from the kinematic one after the hover move, pad guard stop): camera X_grasp only when >= 5 valid
        frames and within TRACK_GATE_MM of the kinematic reference (model TCP + this attempt's c_off); otherwise the kinematic reference."""
        Xk = self.tip_cam() + self.tpl["c_off_cam"]
        if tr["n_valid"] < 5: return Xk, "kin", None
        eg = float(np.linalg.norm(tr["X_grasp"] - Xk))
        return (tr["X_grasp"], "cam", eg) if eg <= TRACK_GATE_MM else (Xk, "kin", eg)

    def needle_shift(self, cur_gray):
        ref = cv2.imread(self.T + "/snap0/right_rect.png", 0).astype(np.float32); cur = cur_gray.astype(np.float32)
        pts = np.array(self.per["method"]["clicks"]["polyline_px"], float); rows = []
        for (u, v) in pts:
            u, v = int(round(u)), int(round(v)); h = 12; Tp = ref[v - h:v + h + 1, u - h:u + h + 1]; W = cur[v - h - 30:v + h + 31, u - h - 30:u + h + 31]
            if Tp.shape != (25, 25) or W.shape != (85, 85): continue
            r = cv2.matchTemplate(W, Tp, cv2.TM_CCOEFF_NORMED); _, mx, _, (lx, ly) = cv2.minMaxLoc(r); rows.append(dict(uv=[u, v], ncc=float(mx), dx=lx - 30, dy=ly - 30))
        good = [r for r in rows if r["ncc"] >= 0.7]
        if len(good) < 3: return dict(ok=False, n_valid=len(good), shift_mm=None, reason="unverifiable", per_point=rows)
        d = np.median([[r["dx"], r["dy"]] for r in good], 0); mm = float(np.linalg.norm(d) * self.p_star[2] / FX)
        return dict(ok=mm <= 1.0, n_valid=len(good), shift_px=d.tolist(), shift_mm=mm, per_point=rows)

    def descend_close_lift(self):
        z = self.down_axis()
        # R6 descend 8 x 1 mm, now WITH the contact test after every mm (user 13:5x: do not press into the silicone); contact -> no touch phase
        # 16:4x (user OK): baseline after the first step (R6 default: start-up friction; A23/A25 ins +0.69/+0.71 at step 1 with 'before'),
        # insertion criteria only (j0/j1 effort drifted 0.04 -> 0.056 with the motion and stopped A18-A26 at step 2-3; logged, not used here)
        hit0, rows0 = self.mv.descend_until_contact(z, 8.0, tag="descend", step_mm=1.0, baseline="after_first", lim=(1.0, 1.0), use=("ins_effort", "ins_track_mm"), early_steps=4, early_ins=0.20)   # 2026-11-04 fix A (user approved): earliest real contact in the logs = step 5
        b = self.save_frames("pre_close", 10, all_frames=True)
        sh0 = self.needle_shift(cv2.cvtColor(np.mean([x[1] for x in b], 0).round().astype(np.uint8), cv2.COLOR_BGR2GRAY))
        if hit0 is None: hit, rows = self.mv.descend_until_contact(z, 6.0, tag="touch", step_mm=0.5, baseline="before")   # R6 touch --mm 6
        else: hit, rows = None, [dict(step=0, note=f"contact during the 1 mm descent at step {hit0}: touch phase skipped")]
        self.S["descend_contact_step_1mm"] = hit0
        b = self.save_frames("pre_close_touch", 10, all_frames=True)
        sh = self.needle_shift(cv2.cvtColor(np.mean([x[1] for x in b], 0).round().astype(np.uint8), cv2.COLOR_BGR2GRAY))
        self.phase("descend", contact_step=hit, touch_mm=0.5 * (len(rows) - 1), needle_check_after_descend=sh0, needle_check=sh, tip_pad_mm=self.mv.pad_dist_mm(tip_of(self.mv.setpoint())))
        if not sh["ok"]: raise QualityStop(f"needle moved / unverifiable before closing ({sh.get('shift_mm')} mm, {sh['n_valid']} patches): no close (R6)")
        self.S["close_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        jc = self.mv.jaw_to(CLOSE_RAD, n_steps=10, dt=0.3, tag="close")            # R6: 10 steps x 0.3 s
        self.save_frames("closed", 6)
        for k in range(10):                                                            # R6 lift: 10 x 1 mm along -axis
            self.mv.translate(-z * 0.001, tag=f"lift{k}", contact_ok="away", lim=(1.0, 1.0)); time.sleep(0 if self.fake else 0.15)
        self.save_frames("lifted", 6); time.sleep(0 if self.fake else 2.0)
        self.save_frames("lifted_2s", 10, all_frames=True)
        gc = self.grasp_check()
        self.phase("close_lift", jaw_after_close=jc, jaw_after_hold=float(self.io.get("jaw")[1][0]), grasp_check=gc)
        if not gc["grasped"]: raise QualityStop(f"not grasped: {gc['why']}")

    def grasp_check(self):
        """15:3x (user after A18: 'clearly not grasped; the jaw effort would change'): after the 2 s hold.
        Vision (decides): needle_detect on lifted_2s; the needle found where it lay before (snap0 polyline, symmetric mean nearest-point
        distance < NEEDLE_STAYED_PX) -> not grasped.  Offline A18 (not grasped) 4 px, A08 (grasped) 41 px.
        Jaw angle / effort vs the empty-close reference at the base: recorded only until there is data for a threshold."""
        hold = self.mv.jaw_sample(); ref = getattr(self, "jaw_ref", None)
        out = dict(jaw_hold=hold, jaw_empty_ref=ref, jaw_dq_rad=None if not ref else hold["q"] - ref["q"],
                   jaw_deff=None if not ref or hold["eff"] is None or ref["eff"] is None else hold["eff"] - ref["eff"])
        if self.fake:
            out.update(grasped=bool(getattr(self.fw, "grasped", True)), why="fake world", needle_stayed_px=None); return out
        d = self.T + "/lifted_2s"
        r = subprocess.run([sys.executable, R6S + "/needle_detect.py", "--snap", d, "--out", d + "/needle_detect.json"], capture_output=True, text=True, timeout=120)
        open(d + "/needle_detect_stdout.txt", "w").write(r.stdout + "\n" + r.stderr)
        poly = json.load(open(d + "/needle_detect.json")).get("polyline_px") if os.path.exists(d + "/needle_detect.json") else None
        if not poly:
            jaw_ok = jaw_holds(out["jaw_dq_rad"], out["jaw_deff"]); out.update(grasped=jaw_ok, jaw_ok=jaw_ok, why="no needle found on the pad after lifting" + ("" if jaw_ok else "; the jaw closed empty"), needle_stayed_px=None); return out
        ref_poly = np.array(json.load(open(self.T + "/clicks.json"))["polyline_px"], float); px = chamfer_px(ref_poly, np.array(poly, float))
        stayed = px < NEEDLE_STAYED_PX; jaw_ok = jaw_holds(out["jaw_dq_rad"], out["jaw_deff"])
        out.update(grasped=(not stayed) and jaw_ok, jaw_ok=jaw_ok, needle_stayed_px=px, why=f"needle still where it lay ({px:.0f} px < {NEEDLE_STAYED_PX})" if stayed else (f"needle moved {px:.0f} px with the jaw" if jaw_ok else f"needle moved {px:.0f} px but the jaw closed empty (dq {out['jaw_dq_rad']}, deff {out['jaw_deff']}): flicked (A05)"))
        return out

    def putdown_release(self):
        z = self.down_axis()
        for k in range(10):                                                            # R6 putdown: 10 x 1 mm down, then open slowly
            self.mv.translate(z * 0.001, tag=f"putdown{k}", contact_ok="descent", lim=(1.0, 1.0)); time.sleep(0 if self.fake else 0.3)
        j = self.mv.jaw_to(OPEN_RAD, n_steps=7, dt=0.3, tag="putdown_open")
        self.save_frames("putdown", 6); self.phase("putdown", jaw=j)
        self.release()

    def release(self):
        """R6 release: open 0.8, lift 15 mm (3 mm steps), back to base (new segment: <= 2 mm / 3 deg, no cube), close the jaw at base."""
        self.mv.jaw_to(0.8, n_steps=1, dt=1.8, tag="release_open")
        z = self.down_axis()
        lift = 15.0
        if self.mv.cube is not None:          # 14:4x (A14/A15): a quality stop at the hover start pose is ~25 mm above the goal; 15 mm more left the cube
            c, h = self.mv.cube; tip = self.mv.tip(self.mv.setpoint())
            while lift > 0 and np.any(np.abs(tip - z * lift * 1e-3 - c) > h - 1e-3): lift -= 0.5
            if lift < 15.0: self.log(dict(ev="note", what=f"release lift {lift:.1f} mm (cube edge); safe_goto lifts further without the cube"))
        if lift > 0: self.mv.translate(-z * lift * 1e-3, tag="release_lift", contact_ok="away", lim=R6_TRANS)
        self.mv.cube = None
        self.mv.safe_goto(self.Tbase, tag="back_to_base")
        self.mv.jaw_to(0.0, n_steps=1, dt=1.8, tag="release_close_at_base")
        self.save_frames("after_release", 6); self.phase("release", at_base_mm=float(np.linalg.norm(self.mv.setpoint()[:3, 3] - self.Tbase[:3, 3]) * 1e3))

    # ------------------------------------------------------------------ chain
    def run(self, cells):
        status, reason = "completed", None
        try:
            try:
                self.perceive(); self.approach(); self.wrist(); self.open_and_template(); self.make_goal()
                self.reach_start(cells); self.reach_ladder(); self.hover(); self.descend_close_lift(); self.putdown_release()
            except QualityStop as q:                     # not hardware: record, retreat to the base with the guarded mover, session continues
                status = "stopped"; reason = f"quality: {q}"; self.S["abort_class"] = "quality"
                self.log(dict(ev="quality_stop", reason=reason)); print("QUALITY STOP:", reason, "-> retreat", flush=True)
                if getattr(self, "retreat_ok", False): self.release()
        except BaseException as e:                       # hardware guard / capture failure / Ctrl-C / unexpected error: stop, send nothing more
            status = "aborted"; reason = f"{type(e).__name__}: {e}"
            self.log(dict(ev="abort", reason=reason, tb=traceback.format_exc())); print("ABORT:", reason, flush=True)
            self.S["abort_class"] = classify(e)
        self.S.update(status=status, abort_reason=reason, t_end=time.time(), handover=getattr(self, "handover", None), used_cell=getattr(self, "used_cell", None),
                      n_commands=self.mv.n_cmd, c_off_cam=(getattr(self, "tpl", None) or {}).get("c_off_cam"))
        self.save_state(); return self.S


def chamfer_px(a, b):
    """symmetric mean nearest-point distance of two polylines (each densified 10x), px."""
    dense = lambda p: np.vstack([p[i] + (p[i + 1] - p[i]) * t for i in range(len(p) - 1) for t in np.linspace(0, 1, 10, endpoint=False)] + [p[-1:]])
    A, B = dense(a), dense(b); D = np.linalg.norm(A[:, None] - B[None], axis=2); return float(0.5 * (D.min(1).mean() + D.min(0).mean()))


def patch_reach_pad(tip_off_tool):
    """15:1x (user, option 1): the deploy package (files unchanged) checks pad clearance with the R2-E TCP, ~8 mm along the tool axis from the
    grasp point the camera sees (tip_off_tool A06-A08 [3.2, -0.6, -7.9] mm) -> s0.5 stopped after 3-13 steps.  At run time, reach_real.precheck
    and run_reach_dvrk.preflight run unchanged without the pad, then the same pad test (hinge and tip >= pad_min_mm) uses tip = TCP + tip_off_tool.
    The goal (goal_hinge) still uses the R2-E TCP, as make_goal_json expects."""
    import reach_real, run_reach_dvrk
    pre0 = getattr(reach_real, "_ORIG_precheck", reach_real.precheck); reach_real._ORIG_precheck = pre0
    pf0 = getattr(run_reach_dvrk, "_ORIG_preflight", run_reach_dvrk.preflight); run_reach_dvrk._ORIG_preflight = pf0
    tcp = reach_real.TCP_TOOL_M + np.asarray(tip_off_tool, float)
    def precheck(cmd, prev_cmd, meas, a_scaled, goal, rot0_deg, grasp_ecm, pad, lim=reach_real.LIM):
        ok, why = pre0(cmd, prev_cmd, meas, a_scaled, goal, rot0_deg, grasp_ecm, None, lim)
        if ok and pad is not None:
            n, d = pad; tip = cmd.p + cmd.R @ tcp
            if (n @ cmd.p + d) * 1e3 < PAD_MIN_MM or (n @ tip + d) * 1e3 < PAD_MIN_MM: return False, "pad clearance"   # 16:1x: rig.PAD_MIN_MM
        return ok, why
    def preflight(robot, io, kin, p_grasp, R_goal, pad):
        rep, bad, meas = pf0(robot, io, kin, p_grasp, R_goal, None)
        if pad is not None:
            n, d = pad; tip = meas.p + meas.R @ tcp
            rep["pad_clear_mm"] = dict(hinge=round(float(n @ meas.p + d) * 1e3, 1), tip=round(float(n @ tip + d) * 1e3, 1), tcp="camera-corrected (15:1x)")
            if min(rep["pad_clear_mm"]["hinge"], rep["pad_clear_mm"]["tip"]) < PAD_MIN_MM:
                bad.append(f"pad clearance {rep['pad_clear_mm']} < {PAD_MIN_MM} mm (or wrong side)")
        return rep, bad, meas
    reach_real.precheck = precheck; run_reach_dvrk.preflight = preflight


def classify(e):
    s = str(e)
    for key in ("image capture", "goal_reached", "setpoint did not reach", "IK", "margin", "workspace cube", "pad clearance", "tracking error",
                "measured_cp differs", "operating state", "moved since freeze", "Reach rung"):
        if key.lower() in s.lower(): return key
    return type(e).__name__
