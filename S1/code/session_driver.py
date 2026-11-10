#!/usr/bin/env python3
"""Session driver, 2026-10-28.  One command per stage; every stage writes ~/rl_live/sim2real_20261106/S1/<stage>/ with a STATUS line.

    python3 session_driver.py selftest            # offline FakeIO: calibration, pad, 1 full attempt, capture-failure abort, abort rules
    python3 session_driver.py calib               # C  (robot moves: probe + grid, <= 2 mm / 3 deg steps)
    python3 session_driver.py pad                 # D  (robot moves: 4 tip contacts, 0.5 mm descent)
    python3 session_driver.py prereg              # E
    python3 session_driver.py attempt --n K       # F  attempt K of the placement plan (needle placed, user said "ok")
    python3 session_driver.py summary             # G
Abort rule: 2 consecutive aborted attempts, or the same abort class twice -> 'attempt' refuses until --user-ok is given.
Runs > 5 min: '[>5MIN]' line + terminal bell at the start and every 5 min.
"""
import argparse, hashlib, json, os, sys, threading, time, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.expanduser("~/rl_live/sim2real_20261106/S1")
from rig import (RigIO, FakeRigIO, Mover, Guard, CaptureError, load_kin, T_of, tip_of, TCP_MM, FX, CX, CY, FB, DEPLOY)
sys.path.insert(0, DEPLOY)

PLACEMENTS = [   # 13:2x: only the flat area right of the letter 'd', below the two bars, left of the blue tape (A02 on the letters: flatness fail)
    ("P1", "'Med' 与 'dmed.com' 之间（初始位置）", "针尖朝左"),
    ("P2", "'Med' 上方横凸起左端左边（在字母上，A02 平整度不过）", "针尖朝上"),
    ("P3", "平整区：字母 'd' 右边、两条横凸起下方、蓝胶带左边", "针尖朝上（画面上方）"),
    ("P4", "同一平整区", "针尖朝右"),
    ("P5", "同一平整区", "针尖朝下（画面下方）"),
    ("P6", "同一平整区", "针尖朝左"),
    ("P7", "同一平整区", "针尖朝右上 45°"),
    ("P8", "同一平整区", "针尖朝左下 45°"),
]


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def status_line(d, text, files=()):
    os.makedirs(d, exist_ok=True)
    h = " ".join(f"{os.path.basename(f)}:{sha(f)[:12]}" for f in files if os.path.exists(f))
    with open(d + "/STATUS", "a") as f: f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {text} | {h}\n")


def logger(path):
    f = open(path, "a")
    def log(rec):
        rec = dict(rec); rec.setdefault("t", time.time())
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"); f.flush()
    return log


def tty_say(msg):
    """bell + message to the user's terminal (S1028_TTY, e.g. /dev/pts/1) as well as stdout."""
    print(msg, flush=True); t = os.environ.get("S1028_TTY")
    if t:
        try:
            with open(t, "w") as f: f.write("\a\r\n" + msg + "\r\n")
        except OSError: pass


class FiveMin:
    def __init__(self, what, est_min):
        self.what = what; self.stop = threading.Event()
        tty_say(f"[>5MIN] {what}: expected ~{est_min} min. Ctrl-C / kill -INT stops it (no further command is sent).")
        threading.Thread(target=self._run, daemon=True).start(); self.t0 = time.time()
    def _run(self):
        while not self.stop.wait(300): tty_say(f"[>5MIN] {self.what} still running ({(time.time() - self.t0) / 60:.0f} min)")
    def done(self): self.stop.set()


# ------------------------------------------------------------------------------------------------ fake world (selftest)
class FakeWorld:
    def __init__(self, kin, fail_capture_at=None):
        import cv2
        sc = json.load(open(ROOT + "/C_calib/scene0/state.json"))
        from scipy.spatial.transform import Rotation
        s = sc["suj_psm1_cp"]; Tb = T_of(s[:3], Rotation.from_quat(s[3:]).as_matrix()); q0 = np.array(sc["js"][:6])
        T0 = kin.fk_ecm(q0, Tb); tcp0 = tip_of(T0); z = T0[:3, 2]
        self.n_ecm = -z; c_pad = tcp0 + z * 0.022; self.d = -float(self.n_ecm @ c_pad)
        a = np.array([0, 0.3, -0.954]); a /= np.linalg.norm(a)              # camera direction of the pad normal (toward the camera)
        self.Rc = Rotation.align_vectors([a], [self.n_ecm])[0].as_matrix(); self.tc = np.array([-8.0, -15.0, 100.0]) - self.Rc @ (tcp0 * 1e3)
        img = cv2.imread(ROOT + "/C_calib/scene0/right_rect.png")
        bg = cv2.resize(img[560:1000, 0:560], (1300, 1024)); patch = img[170:231, 530:591].copy()
        self.io = FakeRigIO(kin, Tb, q0, self.n_ecm, self.d, self.Rc, self.tc, patch, bg, fail_capture_at=fail_capture_at)
        self.kin = kin

    def cam_plane(self):
        n = self.Rc @ self.n_ecm; c = self.Rc @ (-self.d * self.n_ecm * 1e3) + self.tc; return n, c

    def disp_map(self):
        n, c = self.cam_plane(); v, u = np.mgrid[0:1024, 0:1300]; r = np.dstack([(u - CX) / FX, (v - CY) / FX, np.ones_like(u, float)])
        Z = (n @ c) / (r @ n); D = (FB / Z).astype(np.float32)
        X = self.io.feat_cam(); u0, v0 = int(FX * X[0] / X[2] + CX), int(FX * X[1] / X[2] + CY); h = self.io.patch.shape[0] // 2
        D[max(v0 - h, 0):v0 + h + 1, max(u0 - h, 0):u0 + h + 1] = FB / X[2]          # the instrument patch at its own depth
        return D

    def write_fake_disp(self, d): np.save(d + "/fake_disp.npy", self.disp_map())

    def fake_perception(self, T):
        n, c = self.cam_plane(); r = np.array([(650 - CX) / FX, (640 - CY) / FX, 1.0]); Z = (n @ c) / (r @ n); p = r * Z + 3.0 * n          # fake needle 3 mm above the pad so the whole chain runs
        t = np.cross(n, [1.0, 0, 0]); t /= np.linalg.norm(t)
        poly = [[600 + 12 * i, 700 + 3 * i] for i in range(10)]
        return dict(ok=True, created_t=time.time(), p_star_mm=p.tolist(), tangent=t.tolist(), normal_toward_cam=n.tolist(), consistency=dict(p_spread_mm=0.1),
                    frames=[dict(snap=T + "/snap0", fit=dict(centre_mm=(p + 9 * t).tolist(), R_mm=9.0, arc_deg=150.0), surface=dict(normal_variation_deg_tail_mid_tip=2.0))],
                    method=dict(clicks=dict(polyline_px=poly)), fake=True)

    def fake_template_clicks(self, att):
        X = self.io.feat_cam(); u, v = FX * X[0] / X[2] + CX, FX * X[1] / X[2] + CY
        Tn = self.io._T(); Xg = self.Rc @ (tip_of(Tn) * 1e3) + self.tc; gu, gv = FX * Xg[0] / Xg[2] + CX, FX * Xg[1] / Xg[2] + CY
        return dict(feat_R=[int(round(u)), int(round(v))], grasp_R=[gu, gv], grasp_L=[gu - FB / Xg[2], gv], half=16)


# ------------------------------------------------------------------------------------------------ stages
def get_rig(fake=None):
    kin = load_kin()
    if fake is not None: return fake.io, kin
    io = RigIO(); t0 = time.time()
    while time.time() - t0 < 6 and any(io.get(k) is None for k in ("cp", "sp", "suj", "js", "spj", "jaw", "ecm")): time.sleep(0.1)
    miss = [k for k in ("cp", "sp", "suj", "js", "spj", "jaw", "ecm") if io.get(k) is None]
    if miss: raise SystemExit(f"missing topics {miss}")
    return io, kin


def stage_calib(root, fake=None, feature_override=None):
    from calib import calibrate
    d = root + "/C_calib"; os.makedirs(d, exist_ok=True); log = logger(d + "/calib_log.jsonl")
    io, kin = get_rig(fake); mv = Mover(io, kin, log, fake=fake is not None)
    fm = FiveMin("C calibration (retract 3 mm, 6 probe moves, 16 grid points, 2 FS runs)", 8) if fake is None else None
    try:
        cal = calibrate(io, mv, kin, d, log, fake=fake is not None, feature_override=feature_override)
        status_line(d, f"C: calibration ok, test err {np.round(list(cal['test_err_mm'].values()), 2).tolist()} mm, train rms {cal['train_rms_mm']:.2f}, sv {np.round(cal['sv'], 3).tolist()}, rejected {cal['n_rejected']}, {mv.n_cmd} commands",
                    [d + "/calibration_r6.json", d + "/calib_log.jsonl"])
        return cal
    except BaseException as e:
        status_line(d, f"C: ABORTED {type(e).__name__}: {e} ({mv.n_cmd} commands sent)", [d + "/calib_log.jsonl"]); log(dict(ev="abort", reason=str(e), tb=traceback.format_exc())); raise
    finally:
        if fm: fm.done()


def convert_r6_cal(root):
    """R6 r6_calib output (cp-based affine, manifest) -> calibration_tcp.json used by the attempt chain and make_goal_json.
    M_tcp[:3] = M[:3]; M_tcp[3] = M[3] - (R_ref TCP) @ M[:3]  so that [p_tcp, 1] @ M_tcp == [cp, 1] @ M at the reference orientation."""
    from scipy.spatial.transform import Rotation
    d = root + "/C_r6cal"; man = json.load(open(d + "/manifest.json")); c6 = json.load(open(d + "/calibration_r6.json")); st = json.load(open(d + "/pre3/state.json"))
    M = np.array(c6["M"]); Rref = Rotation.from_quat(man["ref_orientation_xyzw"]).as_matrix()
    # hinge-pin correction (2026-10-28, after the D contact): the tracked feature is on the clevis, not at cp. Measure cp (= jaw hinge pin)
    # in the camera at the base with FS (cross-checked by NCC) and shift the affine offset so that [cp, 1] @ M_cp lands on the pin.
    import cv2
    from calib import match_band, tri
    si = json.load(open(d + "/scene_inputs.json")); u, v = si["pin_uv"]; disp = np.load(d + "/pre3/fs_disp.npy")
    R0 = cv2.imread(d + "/pre3/right_rect.png", 0).astype(np.float32); L0 = cv2.imread(d + "/pre3/left_rect.png", 0).astype(np.float32)
    d_fs = float(np.median(disp[v - 3:v + 4, u - 3:u + 4])); ncc, uL, _ = match_band(L0, R0[v - 15:v + 16, u - 15:u + 16], u - d_fs, v, 8)
    X_pin = np.array([u - CX, v - CY, FX]) * (FB / d_fs) / FX; X_pin_ncc = tri(u, v, uL)
    pin = dict(uv=[u, v], d_fs=d_fs, d_ncc=float(u - uL), ncc=ncc, fs_vs_ncc_mm=float(np.linalg.norm(X_pin - X_pin_ncc)))
    cp0 = np.array(man["base_cp_m"]) * 1e3; bias = X_pin - np.r_[cp0, 1] @ M; pin["bias_cam_mm"] = bias.tolist()
    if pin["fs_vs_ncc_mm"] > 1.5: raise SystemExit(f"hinge pin: FS and NCC disagree by {pin['fs_vs_ncc_mm']:.2f} mm")
    Mcp = M.copy(); Mcp[3] = M[3] + bias
    Mt = Mcp.copy()      # pin-anchored affine maps ANY ECM point to its camera position (13:1x fix: the extra -(R_ref TCP)@A put the "TCP" at the pin)
    s = st["suj_psm1_cp"]; suj = T_of(s[:3], Rotation.from_quat(s[3:]).as_matrix())
    gate = bool(c6.get("gate_pass_final", c6.get("gate_pass")))
    cal = dict(M=Mt.tolist(), M_r6_cp=M.tolist(), pin_correction=pin, TCP_tool_mm=TCP_MM.tolist(), model="X_cam(point) = [p_ecm_mm, 1] @ M for any rigid point at the reference orientation (R6 affine, offset anchored on the hinge pin = cp)",
               ref_orientation_xyzw=man["ref_orientation_xyzw"], base_cp_m=man["base_cp_m"], base_T=T_of(man["base_cp_m"], Rref).tolist(),
               frozen=dict(ecm_js=list(man["ecm_js"])[:4], suj_psm1=suj.tolist()), gate_pass=gate, test_err_mm=c6.get("test_err_mm"), train_rms=c6.get("train_rms"), sv=c6.get("sv"),
               source=dict(r6_calibration_sha256=sha(d + "/calibration_r6.json"), manifest_sha256=sha(d + "/manifest.json")), converted=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(cal, open(d + "/calibration_tcp.json", "w"), indent=1)
    status_line(d, f"C: converted to calibration_tcp.json, gate_pass {gate}, test err {c6.get('test_err_mm')}", [d + "/calibration_tcp.json"])
    return cal


def fs_pad(root, margin_mm=2.0):
    """pad.json from the base FS (pre3) of the calibration: RANSAC plane (0.5 mm) on flat pad pixels >= 40 px from needle/tool/tape,
    inliers mapped to ECM with calibration_tcp.json, SVD plane, normal toward the base tip, shifted margin_mm toward free space."""
    import cv2
    from calib import cam_points, cam2ecm_mm
    cal = load_cal(root); M = np.array(cal["M"]); d = root + "/C_r6cal"
    R0 = cv2.imread(d + "/pre3/right_rect.png"); P = cam_points(np.load(d + "/pre3/fs_disp.npy"))
    lab = cv2.cvtColor(R0, cv2.COLOR_BGR2LAB).astype(np.float32); pink = (lab[..., 1] > 140) & (lab[..., 0] > 90)
    clear = cv2.distanceTransform(pink.astype(np.uint8), cv2.DIST_L2, 5) >= 40
    Q = P[clear][::25]; rng = np.random.default_rng(0); best = None
    for _ in range(500):
        s3 = Q[rng.choice(len(Q), 3, replace=False)]; n = np.cross(s3[1] - s3[0], s3[2] - s3[0])
        if np.linalg.norm(n) < 1e-9: continue
        n /= np.linalg.norm(n); k = int((np.abs((Q - s3[0]) @ n) < 0.5).sum())
        if best is None or k > best[0]: best = (k, n, s3[0])
    I = Q[np.abs((Q - best[2]) @ best[1]) < 0.5]
    E = np.array([cam2ecm_mm(x, M) for x in I]) * 1e-3; c = E.mean(0); _, S, Vt = np.linalg.svd(E - c); n = Vt[2]
    tip0 = tip_of(np.array(cal["base_T"]))
    if (tip0 - c) @ n < 0: n = -n
    resid = np.abs((E - c) @ n) * 1e3; dd = -float(n @ c) - margin_mm * 1e-3
    pad = dict(normal=n.tolist(), d=dd, n_points=int(len(E)), inlier_frac=float(len(I) / len(Q)), max_resid_mm=float(np.percentile(resid, 95)), rms_mm=float(np.sqrt((resid ** 2).mean())),
               margin_mm=margin_mm, base_tip_height_mm=float((n @ tip0 + dd) * 1e3),
               source="FS plane (pre3 of C_r6cal) mapped with calibration_tcp.json (pin-corrected); shifted margin_mm toward free space (user option A 2026-10-28)")
    os.makedirs(root + "/D_pad", exist_ok=True); p = root + "/D_pad/pad.json"
    if os.path.exists(p): os.rename(p, p.replace(".json", time.strftime("_superseded_%H%M%S.json")))
    json.dump(pad, open(p, "w"), indent=1); status_line(root + "/D_pad", f"D: FS pad plane, margin {margin_mm} mm, {len(E)} pts, inlier frac {pad['inlier_frac']:.2f}, rms {pad['rms_mm']:.2f} mm, base tip height {pad['base_tip_height_mm']:.1f} mm", [p])
    return pad


def load_cal(root):
    p = root + "/C_r6cal/calibration_tcp.json"; cal = json.load(open(p)); cal["_path"] = p; cal["_base_dir"] = root + "/C_r6cal/pre3"
    if not cal.get("gate_pass"): raise SystemExit("calibration gate not passed")
    return cal


def stage_pad(root, fake=None):
    from calib import pad_touch
    d = root + "/D_pad"; os.makedirs(d, exist_ok=True); log = logger(d + "/pad_log.jsonl")
    io, kin = get_rig(fake); mv = Mover(io, kin, log, fake=fake is not None); cal = load_cal(root)
    mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = np.array(cal["frozen"]["suj_psm1"])
    fm = FiveMin("D pad: 4 tip contacts (2 mm steps, 0.5 mm descent)", 7) if fake is None else None
    try:
        pad = pad_touch(io, mv, kin, cal, d, log, fake=fake is not None)
        status_line(d, f"D: pad plane ok, resid {pad['max_resid_mm']:.2f} mm, base tip height {pad['base_tip_height_mm']:.1f} mm, {mv.n_cmd} commands", [d + "/pad.json", d + "/tips.json"])
        return pad
    except BaseException as e:
        status_line(d, f"D: ABORTED {type(e).__name__}: {e} ({mv.n_cmd} commands sent)", [d + "/pad_log.jsonl"]); log(dict(ev="abort", reason=str(e), tb=traceback.format_exc())); raise
    finally:
        if fm: fm.done()


def stage_prereg(root):
    d = root + "/E_prereg"; os.makedirs(d, exist_ok=True)
    cal = json.load(open(root + "/C_r6cal/calibration_tcp.json")); pad = json.load(open(root + "/D_pad/pad.json"))
    rp = ROOT + "/a_replay_R6calib/replay_result.json"; replay = json.load(open(rp)) if os.path.exists(rp) else None
    import reach_real
    pr = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), session="sim2real_20261106/S1", note="no PREREG_TEMPLATE file exists on this host; fields follow the user's 2026-10-28 instructions (hybrid plan)",
              objective="as many valid attempts as safely possible: policy Reach -> hover visual correction -> grasp",
              package=dict(zip_sha256="73bd9e8c5bd82bf026754527c5d284d285f5f64da7d4ad4fe5c7cb205b97f295", all_files_sha256=sha(DEPLOY + "/ALL_FILES.sha256"),
                           actor_weights_sha256=sha(DEPLOY + "/r6_actor/actor_weights.npz"), reach_real_sha256=sha(DEPLOY + "/reach_real.py"), run_reach_dvrk_sha256=sha(DEPLOY + "/run_reach_dvrk.py")),
              code_sha256={f: sha(HERE + "/" + f) for f in ("rig.py", "calib.py", "attempt.py", "session_driver.py", "clickgui.py", "watchdog.py")},
              calibration=dict(method="R6 r6_calib.py (copy in C_calib_r6/code: ROOT path changed; feature pixel and FS pad box are today's scene inputs in scene_inputs.json; no parameter changed), read-only watchdog.py alongside",
                               sha256=sha(root + "/C_r6cal/calibration_tcp.json"), test_err_mm=cal.get("test_err_mm"), sv=cal.get("sv"), gate_pass=cal.get("gate_pass")),
              pad=dict(method="4 tip contacts (calib.pad_touch: flat FS spots, 0.5 mm descent, relative effort test), jaw as it was", sha256=sha(root + "/D_pad/pad.json"), max_resid_mm=pad["max_resid_mm"]),
              reach=dict(limits=reach_real.LIM, success=f"hinge <= {reach_real.SUCCESS_MM} mm and policy RPY <= {reach_real.SUCCESS_RPY_DEG} deg",
                         ladder="dry -> s0 -> s0.5 -> s1, pass = success|max_steps|no_progress; back to the start pose before s0.5 and s1",
                         goal="GO_day: orientation after the wrist phase; grasp point = perception p* mapped with calibration_tcp.json and c_off (see deviations)",
                         start="realstart_cells.json offset of the assigned cell, rebased (frames.py); unreachable -> next cell; none -> current pose (deviation)", align_rpy_branch=False),
              motion_limits=dict(new_segments="free <= 5 mm / 5 deg, within 20 mm of pad or needle <= 2 mm / 3 deg; base -> above-needle and back-to-base always <= 2 mm / 3 deg",
                                 grasp_protocol="R6 frozen values: translations <= 3 mm, wrist <= 4 deg (pitch/yaw) / 6 deg (roll) per step, descend 8 x 1 mm, touch 0.5 mm steps up to 6 mm, lift 10 x 1 mm, put-down 10 x 1 mm, release lift 15 mm",
                                 jaw="R6: open 0.6 rad one command; close -0.25 rad 10 x 0.3 s; put-down open 0.6 rad 7 x 0.3 s; release open 0.8, close 0.0 at base"),
              stop_policy=dict(hardware_stop=["PSM1 FAULT / not ENABLED", "goal_reached timeout", "setpoint != target (silent IK clip)", "joint margin < 3 deg / 5 mm", "workspace cube / pad violation",
                                              "effort anomaly / unexpected contact (joint tracking)", "measured_cp vs SUJ x FK > 0.5 mm", "ECM / SUJ moved", "any image capture failure (whole chain)",
                                              "any hardware abort -> wait for the user (covers 2 consecutive)"],
                               everything_else="quality stop: recorded, arm retreats to base with the guarded mover, next placement"),
              guards=["workspace +-40 mm cube around the grasp point applied to the TIP (hinge checked by reach_real)", "pad >= 3 mm (hinge and tip) always on once pad.json exists, except descend/touch/put-down; lift moves may not get closer"],
              grasp_protocol="R6 condition C (r6_grasp.py 10/16): approach H25, wrist straighten + roll (y perp tangent), open, template, [Reach], hover 10 mm above p* along the jaw axis with <= 3 stereo corrections, descend 8 mm, touch, needle shift <= 1 mm, slow close, lift 10 mm, hold 2 s, put down, release",
              records="per attempt: clicks_overlay.png + how (auto accepted / manual), template overlays, c_off, pre_close, pre_close_touch, closed, lifted, lifted_2s, putdown images; all Reach logs; handover error (mm, deg) Reach end -> hover target; hover result; success judged by the user",
              reporting="numbers only, no success rates, no conclusions; every attempt incl. stops and aborts is listed",
              deviations=[
                  "calib.py (rewritten calibration) NOT used: offline replay of the 10/16 R6 calibration reproduced the FIT (identical M, rms 0.738 mm, held-out 0.35 / 1.58 mm; g11 original 3.25 vs R6 re-measured 2.93) "
                  "but NOT the TRACKING (base stereo NCC 0.47 < 0.6 gate, 3 of 12 grid points accepted vs 11 in R6, positions 0.30-6.47 mm from R6's). " + ("replay_result.json sha256 " + sha(rp) if replay else ""),
                  "r6_calib.py copy: ROOT path changed; FEAT_UV and the FS pad box (hardcoded for the 10/16 images) read from scene_inputs.json with today's values",
                  "calibration converted to the TCP point (calibration_tcp.json: M_tcp[3] = M[3] - (R_ref TCP) @ M[:3]); c_off per attempt = template-click grasp point - model at the goal orientation, passed as make_goal_json --c-off. "
                  "Self-referential: c_off comes from the same kind of template measurement as the hover correction. Needed because make_goal_json (ecm_from_cam) and the R6 grasp script (cam2ecm + R_ref TCP) disagree on the point convention.",
                  "Reach stopped by reach_real's pad guard is recorded as 'abort: pad clearance' (steps, mm, deg) and the attempt CONTINUES to hover with no distance gate; same for the other policy-independent precheck stops (rotation grew, pitch, step sizes, consecutive geodesic). Workspace cube, tracking error and robot-level aborts stop the session.",
                  "after Reach the arm rotates about the tip to the goal orientation (new segment) before hover, because the template was taken at that orientation",
                  "wrist phase executed as Cartesian tip-fixed rotations with R6's per-step bounds instead of R6's joint moves + compensation",
                  "hover tracking, descend, touch, close, lift, put-down: R6 code ported 1:1 into attempt.py with R6 numbers; touch effort baseline taken before stepping (R6), not after the first step (10/26 note)",
                  "no contact within the 6 mm touch: closes anyway as in R6 (recorded)",
                  "needle click window: green auto polyline is accepted only with the explicit key 'a'; otherwise manual clicks; how + overlay saved per attempt",
                  "run_reach_dvrk.main called in-process with automatic SEND confirmation (user authorisation 2026-10-28); DvrkRobot subclassed in memory to add the day's guards",
                  "pad touches with the jaw as it was (not closed); spots: flat FS window (rms <= 0.4 mm), >= 100 px (~6.7 mm) from any non-pad pixel (120 px left only 3 spots 15 mm apart)",
                  "C try 1 (feature at the jaw hinge 690,555) lost tracking on the first 2 mm probe (R6 'low NCC' assert, NCC 0.51/0.46); try 2 with the feature on the clevis (780,640), same parameters: train rms 0.114 mm, held-out 0.13 / 0.67 / 0.49 mm",
                  "C try 2 grid run 1 stopped by the watchdog at command 33 (measured 1.04 mm from the command: outer-yaw steady tracking -0.55 deg, user confirmed no contact); measured-vs-command threshold raised 1.0 mm / 1.5 deg -> 2.0 mm / 2 deg (watchdog, rig, attempt); run-1 grid records set aside, grid re-run from base",
                  "base (user-placed calibration start) occludes the image centre-right: placements are on the left half of the pad",
                  "C try 3 (user lowered the base, jaw closed; feature 690,880) failed: grid g05 feature left the image -> R6 'too low' abort, gate fail; retry try 4 at the same base with feature 625,800: train rms 0.28 mm, held-out 0.86 / 0.70 / 0.77 mm (this is the calibration used)",
                  "hinge-pin correction: affine offset re-anchored on the jaw hinge pin (= cp) measured by FS at the base (FS vs NCC 0.11 mm); then [p, 1] @ M maps any ECM point at the reference orientation to its camera position",
                  "conversion bug found and fixed 13:1x: an extra -(R_ref TCP) @ A put the model TCP at the hinge pin (~24 mm); D runs made with it are invalid",
                  "D (pad): tip contact abandoned. Run 1 (12:53, try-2 mapping) real pad contact while the planned height was wrong (unexpected contact guard, ins effort up to 3.8, retracted 6 mm with user OK). Run 2 (13:07, buggy mapping): 4 'contacts' were false positives of the j0/j1 effort test (0.05) during long descents, 9.5-12.4 mm above the pad. pad.json = FS plane (pre3 of try 4) mapped with the fixed calibration, margin 2 mm toward free space (5 mm was mentioned for option A; 2 mm used because 8 mm effective clearance conflicts with the R6 hover height). Cross-check: the 12:53 contact onset lies 1-2 mm above this plane"],
              placements=[dict(id=p[0], where=p[1], orientation=p[2]) for p in PLACEMENTS])
    p = d + "/PREREG_20261028.json"
    if os.path.exists(p): os.rename(p, p.replace(".json", time.strftime("_superseded_%H%M%S.json")))
    json.dump(pr, open(p, "w"), indent=1, ensure_ascii=False); h = sha(p); open(p + ".sha256", "w").write(f"{h}  PREREG_20261028.json\n")
    status_line(d, f"E: prereg written sha256 {h}", [p]); return p, h


def ledger_path(root): return root + "/F_attempts/ledger.jsonl"


def abort_rule(root):
    p = ledger_path(root)
    if not os.path.exists(p): return None
    L = [json.loads(l) for l in open(p) if l.strip()]
    if len(L) >= 2 and L[-1]["status"] == "aborted" and L[-2]["status"] == "aborted" and not L[-1].get("user_ok_after"): return "2 consecutive hardware-safety aborts"
    if L and L[-1]["status"] == "aborted" and not L[-1].get("user_ok_after"): return "last attempt ended on a hardware-safety abort: arm not at base, user must look"
    return None


def stage_attempt(root, n, fake=None, user_ok=False, gui=True, where_note=None):
    from attempt import Attempt
    why = abort_rule(root)
    if why and not user_ok: raise SystemExit(f"STOP RULE: {why} -> wait for the user (rerun with --user-ok after they decide)")
    cells = json.load(open(DEPLOY + "/reach_support_pack/realstart_cells.json"))
    pid, where, ori = PLACEMENTS[(n - 1) % len(PLACEMENTS)]; cell = cells[(n - 1) % len(cells)]
    cell = next((c for c in cells if c["placement"] == "debug6_P0e_S"), cell)   # 15:4x user: debug6 first (only cell that converges in reach_real's own self-test); unreachable -> next cell as before
    if where_note: pid, where, ori = "Prep", where_note, ""
    tdir = f"{root}/F_attempts/A{n:02d}_{pid}"
    if os.path.exists(tdir): raise SystemExit(f"{tdir} exists: attempts are never overwritten")
    if fake is None and not os.environ.get("DISPLAY"): raise SystemExit("DISPLAY not set: the click windows need the console display")
    os.makedirs(tdir); log = logger(tdir + "/trial_log.jsonl")
    io, kin = get_rig(fake); mv = Mover(io, kin, log, fake=fake is not None); cal = load_cal(root)
    mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = np.array(cal["frozen"]["suj_psm1"])
    fm = FiveMin(f"attempt A{n:02d} ({pid}: {where}, {ori}; start cell {cell['placement']})", 12) if fake is None else None
    try:
        S = Attempt(io, mv, kin, cal, root + "/D_pad/pad.json", tdir, pid, cell, log, fake=fake is not None, gui=gui, fake_world=fake).run(cells)
    finally:
        if fm: fm.done()
    row = dict(n=n, placement=pid, where=where, orientation=ori, cell=cell["placement"], used_cell=S.get("used_cell"), status=S["status"], abort_reason=S.get("abort_reason"),
               abort_class=S.get("abort_class"), handover=S.get("handover"), reach=(S.get("reach") or {}).get("rungs"), contact_step=(S.get("descend") or {}).get("contact_step"),
               last_phase=S["phases"][-1] if S["phases"] else None, dir=tdir, n_commands=S.get("n_commands"), user_ok_after=bool(user_ok), t=time.time())
    with open(ledger_path(root), "a") as f: f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    status_line(tdir, f"F: A{n:02d} {pid} status {S['status']} last phase {row['last_phase']} reason {S.get('abort_reason')}", [tdir + "/trial_state.json", tdir + "/trial_log.jsonl"])
    print(json.dumps(row, indent=1, default=str)); return row


def stage_summary(root):
    p = ledger_path(root); L = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    lines = ["| # | 摆放 | 起点 cell | 状态 | 最后阶段 | 交接误差 mm / deg | Reach 各档 | 接触步 | 中止原因 | 文件夹 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in L:
        h = r.get("handover") or {}; rs = " / ".join(f"{x['scale']}:{x['status']}({x['steps']})" for x in (r.get("reach") or []))
        lines.append(f"| {r['n']} | {r.get('placement')} {r.get('where')} {r.get('orientation')} | {r.get('used_cell') or r.get('cell')} | {r['status']} | {r.get('last_phase')} | "
                     f"{h.get('to_hover10_mm', float('nan')):.1f} / {h.get('to_hover10_deg', float('nan')):.1f} | {rs} | {r.get('contact_step')} | {r.get('abort_reason') or ''} | {r.get('dir')} |")
    out = root + "/G_summary.md"; open(out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines)); return out


# ------------------------------------------------------------------------------------------------ selftest
def fake_r6_cal(fw, root):
    """what convert_r6_cal would produce for the fake world (exact affine), + pre3 snapshot for the pad candidates."""
    from scipy.spatial.transform import Rotation
    import calib as C
    d = root + "/C_r6cal"; os.makedirs(d + "/pre3", exist_ok=True)
    T0 = fw.io._T(); Rref = T0[:3, :3]
    A = fw.Rc.T; b = fw.Rc @ (Rref @ (fw.io.feat - TCP_MM)) + fw.tc; M = np.vstack([A, b])
    C.save_snap(fw.io, d + "/pre3"); np.save(d + "/pre3/fs_disp.npy", fw.disp_map())
    cal = dict(M=M.tolist(), TCP_tool_mm=TCP_MM.tolist(), ref_orientation_xyzw=Rotation.from_matrix(Rref).as_quat().tolist(), base_cp_m=T0[:3, 3].tolist(), base_T=T0.tolist(),
               frozen=dict(ecm_js=fw.io.ecm.tolist(), suj_psm1=fw.io.base.tolist()), gate_pass=True, test_err_mm={}, fake=True)
    json.dump(cal, open(d + "/calibration_tcp.json", "w"), indent=1); return cal


def selftest():
    import shutil, tempfile
    kin = load_kin(); tmp = tempfile.mkdtemp(prefix="s1028_selftest_", dir=os.environ.get("S1028_TMP")); os.makedirs(tmp + "/C_calib")
    shutil.copytree(ROOT + "/C_calib/scene0", tmp + "/C_calib/scene0")
    fw = FakeWorld(kin)
    import calib as C, attempt as A
    orig = C.save_snap
    def save_snap_fake(io, d, k=6, extra=None):
        r = orig(io, d, k, extra); np.save(d + "/fake_disp.npy", fw.disp_map()); return r
    C.save_snap = save_snap_fake; A.save_snap = save_snap_fake
    print("== selftest hooks refused on the real path")
    try: C.calibrate(fw.io, Mover(fw.io, kin, lambda r: None), kin, tmp + "/x", lambda r: None, fake=False, feature_override=[1, 2]); raise AssertionError("feature_override accepted")
    except ValueError as e: print("   calibrate:", e)
    for kw in (dict(fake=False, fake_world=fw), dict(fake=True, fake_world=None), dict(fake=False, gui=False)):
        try: A.Attempt(fw.io, Mover(fw.io, kin, lambda r: None), kin, dict(M=np.eye(4, 3).tolist(), ref_orientation_xyzw=[0, 0, 0, 1], base_T=np.eye(4).tolist()), None, tmp, "P", None, lambda r: None, **kw); raise AssertionError(f"hook accepted {kw}")
        except ValueError as e: print("   Attempt", {k: (v if not isinstance(v, FakeWorld) else "fw") for k, v in kw.items()}, "->", e)
    print("== selftest C (fake R6 calibration file)"); fake_r6_cal(fw, tmp)
    print("== selftest D (fake pad plane = fake world plane; robot uses the FS plane)"); os.makedirs(tmp + "/D_pad", exist_ok=True)
    json.dump(dict(normal=fw.n_ecm.tolist(), d=fw.d, max_resid_mm=0.0, fake=True), open(tmp + "/D_pad/pad.json", "w"))
    print("== selftest E"); p, h = stage_prereg(tmp); print("   prereg", h[:12])
    print("== selftest F (full attempt, cell debug6)"); r = stage_attempt(tmp, 8, fake=fw, gui=False)
    assert r["status"] == "completed", r
    print("== selftest F (quality stop: hover tracking failure -> retreat to base, session continues)")
    real_track = A.Attempt.track10
    A.Attempt.track10 = lambda self, name, pred: dict(n_valid=0, rows=[])
    r2 = stage_attempt(tmp, 17, fake=fw, gui=False); A.Attempt.track10 = real_track
    assert r2["status"] == "stopped" and r2["last_phase"] == "release", r2; print("   ", r2["status"], r2["abort_reason"], "-> last phase", r2["last_phase"])
    print("== selftest F (capture failure on the 4th capture -> abort, no command after it)")
    fw.io.fail_at = fw.io.n_frames + 4
    r3 = stage_attempt(tmp, 26, fake=fw, gui=False); assert r3["status"] == "aborted" and "capture" in r3["abort_reason"], r3
    lines = [json.loads(l) for l in open(r3["dir"] + "/trial_log.jsonl")]; ia = max(i for i, x in enumerate(lines) if x.get("ev") == "abort")
    assert not any(x.get("ev") in ("step", "jaw") for x in lines[ia:]), "a command was logged after the abort"
    fw.io.fail_at = None; print("    aborted, no command after it")
    print("== selftest stop rule (last attempt = hardware abort)")
    try: stage_attempt(tmp, 35, fake=fw, gui=False); raise AssertionError("stop rule did not fire")
    except SystemExit as e: print("   ", e)
    stage_summary(tmp); print("SELFTEST PASSED; files in", tmp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "calib", "convert", "pad", "fspad", "prereg", "attempt", "summary"]); ap.add_argument("--n", type=int); ap.add_argument("--user-ok", action="store_true"); ap.add_argument("--where")
    a = ap.parse_args()
    try:
        if a.cmd == "selftest": selftest()
        elif a.cmd == "calib": stage_calib(ROOT)
        elif a.cmd == "convert": print(json.dumps({k: v for k, v in convert_r6_cal(ROOT).items() if k not in ("M", "M_r6_cp", "base_T")}, indent=1))
        elif a.cmd == "pad": stage_pad(ROOT)
        elif a.cmd == "fspad": print(json.dumps(fs_pad(ROOT), indent=1))
        elif a.cmd == "prereg": print(stage_prereg(ROOT))
        elif a.cmd == "attempt": stage_attempt(ROOT, a.n, user_ok=a.user_ok, where_note=a.where)
        elif a.cmd == "summary": stage_summary(ROOT)
    except SystemExit as e:
        print("STOP:", e, flush=True); sys.stdout.flush(); os._exit(2)
    except BaseException as e:
        traceback.print_exc(); print("STOP:", type(e).__name__, e, flush=True); sys.stdout.flush(); os._exit(1)
    sys.stdout.flush(); os._exit(0)
