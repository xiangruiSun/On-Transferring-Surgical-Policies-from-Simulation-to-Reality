#!/usr/bin/env python3
"""2026-10-28 session: robot I/O + guarded motion for PSM1 (ProGrasp 400093).  New file; nothing from earlier sessions is modified.

RigIO      ROS 2 layer.  Same get()/clear_reached()/publish_move_cp() interface as deploy_export/dvrk_robot.RclpyIO, so the
           unchanged DvrkRobot / run_reach_dvrk.main can drive it; adds images, jaw, efforts, setpoint_js, ECM, operating state.
FakeRigIO  offline stand-in (kin.py FK/IK, synthetic stereo images of a textured patch, a pad plane that produces effort on contact).
Mover      every PSM1 motion of the day goes through Mover.step(): one move_cp per call, checked before AND after:
             before: fresh state, measured_cp vs SUJ x FK <= 0.5 mm, operating state ENABLED, ECM / SUJ base unchanged since
                     freeze, IK reachable with joint margins >= 3 deg / 5 mm, hinge+tip inside the +-40 mm cube, hinge+tip >= 3 mm
                     above the pad (except contact phases), step size within the active limit
             after : goal_reached within 10 s, setpoint == target (<= 0.3 mm / 0.3 deg: no silent IK clip), measured within
                     1.0 mm / 1.5 deg of the target, joint tracking (unexpected contact) <= 0.5 mm insertion / 1.0 deg
           any failure raises Guard; there is no retry and no further command.
Step limits (user, 2026-10-28): free space <= 5 mm / 5 deg; within 20 mm of the pad or the needle <= 2 mm / 3 deg;
descent / contact search <= 0.5 mm.
"""
import json, os, sys, threading, time
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

DEPLOY = os.path.expanduser("~/rl_live/sim2real_20261106/S1/deploy/deploy_export")
sys.path.insert(0, DEPLOY)

FX, CX, CY = 1860.724932, 690.538345, 576.803638
BASELINE_MM = 5.4476353950418295
FB = FX * BASELINE_MM
TCP_MM = np.array([-2.70, 1.08, 23.81])                      # ProGrasp 400093 grasp point in the cp (hinge) frame, R2-E
LIM_FREE = (5.0, 5.0); LIM_NEAR = (2.0, 3.0); LIM_DESCENT = (0.5, 1.0)
NEAR_MM = 20.0; PAD_MIN_MM = -1.0; CUBE_HALF_MM = 40.0   # 16:1x user: pad clearance relaxed 3.0 -> -1.0 mm vs pad.json (= FS plane + 2 mm), i.e. hinge and corrected tip >= 1 mm above the FS plane
CONTACT = dict(ins_effort=0.10, j01_effort=0.05, ins_track_mm=0.3)   # R6 relative contact test (|delta| vs a baseline); ins 0.15 -> 0.10 user 13:5x (avoid pinching the silicone)


class Guard(Exception):
    """A guard fired: stop everything, send nothing more."""


class CaptureError(Guard):
    """Image capture failed: the whole chain stops (10/16 T3_C lesson)."""


class QualityStop(Exception):
    """Not a hardware-safety event (quality gate, perception, tracking): the attempt ends, the arm may retreat, the session continues."""


PAD_OFF = ("descent", "contact", True)        # contact_ok modes that skip the pad clearance check
TRACK_OFF = ("contact", True)                 # modes that skip the joint-tracking (unexpected contact) check


def geodesic_deg(Ra, Rb):
    return float(np.degrees(Rotation.from_matrix(np.asarray(Ra).T @ np.asarray(Rb)).magnitude()))


def T_of(p_m, R):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p_m; return T


def tip_of(T):
    """grasp point (TCP) position in ECM metres for a hinge pose T."""
    return T[:3, 3] + T[:3, :3] @ (TCP_MM * 1e-3)


def load_kin():
    from dvrk_robot import load_session_kin
    return load_session_kin()


# ================================================================================================ ROS I/O
class RigIO:
    def __init__(self, node_name="s20261028", images=True):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState, Image
        from std_msgs.msg import Bool
        self.rclpy = rclpy; self._PoseStamped = PoseStamped; self._JointState = JointState
        self.L = {}; self.lock = threading.Lock(); self.hist = deque(maxlen=400)
        rclpy.init(); self.node = n = rclpy.create_node(node_name)
        def mat(m):
            p, o = m.pose.position, m.pose.orientation
            return T_of([p.x, p.y, p.z], Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix())
        def put(k, conv):
            def cb(m):
                v = conv(m)
                with self.lock: self.L[k] = (time.monotonic(), v, getattr(getattr(m, "header", None), "frame_id", ""))
            return cb
        qs = qos_profile_sensor_data
        n.create_subscription(PoseStamped, "/PSM1/measured_cp", put("cp", mat), 10)
        n.create_subscription(PoseStamped, "/PSM1/setpoint_cp", put("sp", mat), 10)
        n.create_subscription(PoseStamped, "/SUJ/PSM1/measured_cp", put("suj", mat), qs)
        def js_cb(m):
            t = time.monotonic(); pos = np.array(m.position, float); eff = np.array(m.effort, float) if len(m.effort) else np.zeros(len(pos))
            with self.lock:
                self.L["js"] = (t, pos, ""); self.L["jse"] = (t, eff, "")
                spj = self.L.get("spj"); self.hist.append((t, pos[:6].copy(), eff[:6].copy(), None if spj is None else spj[1][:6].copy()))
        n.create_subscription(JointState, "/PSM1/measured_js", js_cb, qs)
        n.create_subscription(JointState, "/PSM1/setpoint_js", put("spj", lambda m: np.array(m.position, float)), qs)
        def jaw_cb(m):                                       # 15:3x: jaw effort too (grasp check)
            t = time.monotonic()
            with self.lock:
                self.L["jaw"] = (t, np.array(m.position, float), ""); self.L["jaw_eff"] = (t, np.array(m.effort, float) if len(m.effort) else np.full(1, np.nan), "")
        n.create_subscription(JointState, "/PSM1/jaw/measured_js", jaw_cb, qs)
        n.create_subscription(JointState, "/ECM/measured_js", put("ecm", lambda m: np.array(m.position[:4], float)), qs)
        n.create_subscription(Bool, "/PSM1/goal_reached", put("reached", lambda m: bool(m.data)), 10)
        try:
            from crtk_msgs.msg import OperatingState
            tl = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            n.create_subscription(OperatingState, "/PSM1/operating_state", put("opstate", lambda m: str(m.state)), tl)
        except Exception:
            pass
        self.pub = n.create_publisher(PoseStamped, "/PSM1/move_cp", 1)
        self.pub_jaw = n.create_publisher(JointState, "/PSM1/jaw/move_jp", 1)
        self.grab = None
        if images:
            import message_filters as mf
            from cv_bridge import CvBridge
            self.br = CvBridge()
            subs = [mf.Subscriber(n, Image, f"/jhu_daVinci/{c}/image_rect", qos_profile=qs) for c in ("left", "right")]
            self.sync = mf.ApproximateTimeSynchronizer(subs, 10, 0.02); self.sync.registerCallback(self._img)
        threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()

    def _img(self, l, r):
        g = self.grab
        if g is not None and len(g) < self.want:
            g.append((self.br.imgmsg_to_cv2(l, "bgr8"), self.br.imgmsg_to_cv2(r, "bgr8"), l.header.stamp.sec + 1e-9 * l.header.stamp.nanosec))

    def get(self, k):
        with self.lock: return self.L.get(k)

    def clear_reached(self):
        with self.lock: self.L.pop("reached", None)

    def publish_move_cp(self, frame_id, T):
        m = self._PoseStamped(); m.header.frame_id = frame_id
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, T[:3, 3])
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = map(float, q)
        self.pub.publish(m)

    def publish_jaw(self, q):
        m = self._JointState(); m.position = [float(q)]; self.pub_jaw.publish(m)

    def frames(self, k=6, timeout=10.0):
        """k synced (left, right, stamp) frames, topic names (physical left = topic RIGHT).  CaptureError if not all arrive."""
        for _ in range(2):
            self.want = k; self.grab = []; t = time.time()
            while len(self.grab) < k and time.time() - t < timeout: time.sleep(0.02)
            b, self.grab = self.grab, None
            if len(b) == k: return b
        raise CaptureError(f"image capture failed: {len(b)}/{k} synced frames in {timeout:.0f} s")

    def samples(self, dt):
        """(pos, eff, err) means of measured_js over the next dt seconds."""
        t0 = time.monotonic(); time.sleep(dt)
        with self.lock: S = [h for h in self.hist if h[0] >= t0 and h[3] is not None]
        if len(S) < 3: raise Guard(f"measured_js stream stalled ({len(S)} samples in {dt} s)")
        P = np.array([s[1] for s in S]); E = np.array([s[2] for s in S]); SP = np.array([s[3] for s in S])
        return P.mean(0), E.mean(0), (P - SP).mean(0)

    def shutdown(self):
        sys.stdout.flush(); sys.stderr.flush()


# ================================================================================================ fake I/O (offline tests)
class FakeRigIO:
    """kin.py robot with instant moves; synthetic camera X_cam = Rc @ tip_ecm(mm) + tc with a textured feature patch;
    pad plane in ECM; touching it raises the insertion effort.  fail_capture_at: raise on the n-th frames() call."""
    def __init__(self, kin, T_base, q0, pad_n, pad_d_m, Rc, tc_mm, patch, bg, fail_capture_at=None, feat_tool_mm=(-1.5, 0.5, 16.0)):
        self.kin, self.base, self.q = kin, T_base, np.array(q0, float)
        self.pad = (np.asarray(pad_n, float), float(pad_d_m)); self.Rc, self.tc = Rc, np.asarray(tc_mm, float)
        self.patch, self.bg = patch, bg; self.fail_at = fail_capture_at; self.n_frames = 0
        self.feat = np.asarray(feat_tool_mm, float); self.jawq = 0.66; self.reached = None; self.sent = []; self.ecm = np.array([-0.314, -0.3725, 0.08066, -0.9947])
        self.sp_T = kin.fk_ecm(self.q, T_base); self.t0 = time.monotonic(); self.hist = deque(maxlen=50)

    def _T(self): return self.kin.fk_ecm(self.q, self.base)

    def get(self, k):
        t = time.monotonic()
        if k == "cp": return (t, self._T(), "ECM")
        if k == "sp": return (t, self.sp_T.copy(), "ECM")
        if k == "suj": return (t, self.base.copy(), "ECM")
        if k == "js": return (t, np.r_[self.q, 0.0], "")
        if k == "spj": return (t, np.r_[self.q, 0.0], "")
        if k == "jse": return (t, np.r_[self._eff(), 0.0], "")
        if k == "jaw": return (t, np.array([self.jawq]), "")
        if k == "jaw_eff": return (t, np.array([0.0]), "")
        if k == "ecm": return (t, self.ecm.copy(), "")
        if k == "opstate": return (t, "ENABLED", "")
        if k == "reached": return None if self.reached is None else (t, self.reached, "")

    def _eff(self):
        n, d = self.pad; tip = tip_of(self._T()); depth = -(n @ tip + d) * 1e3       # mm below the pad
        e = np.zeros(6); e[2] = -0.3 + (0.6 * max(depth + 0.2, 0.0) if depth > -0.2 else 0.0); return e

    def clear_reached(self): self.reached = None

    def publish_move_cp(self, frame, T):
        self.sent.append(T.copy())
        q, ep, er = self.kin.ik_ecm(T[:3, 3], T[:3, :3], self.q, self.base)
        self.q = q; self.sp_T = self.kin.fk_ecm(q, self.base); self.reached = True

    def publish_jaw(self, q): self.jawq = float(q)

    def samples(self, dt):
        time.sleep(min(dt, 0.01)); return self.q.copy(), self._eff(), np.zeros(6)

    def feat_cam(self, T=None):
        T = self._T() if T is None else T
        p = (T[:3, 3] + T[:3, :3] @ (self.feat * 1e-3)) * 1e3
        return self.Rc @ p + self.tc

    def frames(self, k=6, timeout=10.0):
        import cv2
        self.n_frames += 1
        if self.fail_at is not None and self.n_frames == self.fail_at: raise CaptureError("fake capture failure (test)")
        X = self.feat_cam(); u = FX * X[0] / X[2] + CX; v = FX * X[1] / X[2] + CY; d = FB / X[2]
        out = []
        for _ in range(k):
            R = self.bg.copy(); Lm = self.bg.copy(); h = self.patch.shape[0] // 2
            for img, uu in ((R, u), (Lm, u - d)):
                M = np.float32([[1, 0, uu - h], [0, 1, v - h]])
                warped = cv2.warpAffine(self.patch, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                mask = cv2.warpAffine(np.ones(self.patch.shape[:2], np.float32), M, (img.shape[1], img.shape[0]))
                img[:] = (img * (1 - mask[..., None]) + warped * mask[..., None]).astype(np.uint8)
            out.append((Lm, R, time.time()))
        return out

    def shutdown(self): pass


# ================================================================================================ guarded motion
class Mover:
    def __init__(self, io, kin, log, *, fake=False):
        self.io, self.kin, self.log = io, kin, log
        self.fake = fake
        self.ecm0 = None; self.suj0 = None          # frozen references (set by freeze())
        self.cube = None                            # (centre ECM m, half m)
        self.pad = None                             # (n, d): n.x + d = 0, n into free space, ECM metres
        self.needle = None                          # ECM metres, for the 'near' step limit
        self.post_s = 0.0 if fake else 0.3
        self.tip_off_tool = np.zeros(3)             # metres, tool frame: camera-measured correction of the TCP (c_off), user option 1 13:4x
        self.n_cmd = 0

    # ------------------------------------------------------------------ state
    def _need(self, k, max_age=0.5, wait=2.0):
        t0 = time.monotonic()
        while True:
            v = self.io.get(k)
            if v is not None and time.monotonic() - v[0] <= max_age: return v[1]
            if time.monotonic() - t0 > wait: raise Guard(f"no fresh {k} (stale > {max_age} s)")
            time.sleep(0.01)

    def measured(self):
        # v2 17:5x (user 'as recommended'): measured_cp and measured_js are read separately (not time-aligned); right after a
        # sub-step one can still be the previous one (A22 2.00 mm, A44 1.79 mm with 2 mm sub-steps; 10/10 checks fine afterwards).
        # Re-read up to 3 times 0.1 s apart; only a persistent mismatch is a hardware stop.
        ds = []
        for k in range(3):
            cp = self._need("cp"); js = self._need("js"); suj = self._need("suj")
            Tfk = self.kin.fk_ecm(js[:6], suj)
            d = float(np.linalg.norm(Tfk[:3, 3] - cp[:3, 3]) * 1e3); ds.append(d)
            if d <= 0.5:
                if k: self.log(dict(ev="note", what=f"measured_cp vs SUJ x FK transient {[round(x, 2) for x in ds]} mm (re-read ok)"))
                return cp
            time.sleep(0.0 if self.fake else 0.1)
        raise Guard(f"measured_cp differs from SUJ x FK by {[round(x, 2) for x in ds]} mm (3 reads)")

    def setpoint(self): return self._need("sp")

    def state_ok(self):
        op = self.io.get("opstate")
        if op is not None and op[1] != "ENABLED": raise Guard(f"PSM1 operating state {op[1]} (FAULT/disabled)")
        if self.ecm0 is not None:
            e = self._need("ecm", max_age=1.0)
            d = np.abs(e - self.ecm0); d_deg = float(np.degrees(np.r_[d[:2], d[3]]).max()); d_mm = float(d[2] * 1e3)
            if d_deg > 0.1 or d_mm > 0.1: raise Guard(f"ECM moved since freeze ({d_deg:.3f} deg / {d_mm:.3f} mm)")
        if self.suj0 is not None:
            s = self._need("suj", max_age=1.0)
            dp = float(np.linalg.norm(s[:3, 3] - self.suj0[:3, 3]) * 1e3); dr = geodesic_deg(s[:3, :3], self.suj0[:3, :3])
            if dp > 0.3 or dr > 0.1: raise Guard(f"PSM1 SUJ base moved since freeze ({dp:.2f} mm / {dr:.3f} deg)")

    def freeze(self):
        self.ecm0 = self._need("ecm", max_age=1.0).copy(); self.suj0 = self._need("suj", max_age=1.0).copy()
        return dict(ecm_js=self.ecm0.tolist(), suj_psm1=self.suj0.tolist())

    def tip(self, T):
        """TCP for the pad / cube / near checks: kinematic TCP + the camera-measured correction (zero until a template is taken)."""
        return tip_of(T) + T[:3, :3] @ self.tip_off_tool

    def pad_dist_mm(self, p):
        if self.pad is None: return None
        n, d = self.pad; return float((n @ p + d) * 1e3)

    def is_near(self, T):
        """within 20 mm of the pad (hinge or tip) or of the needle -> near limits.  Unknown pad -> near (conservative)."""
        if self.pad is None: return True
        tip = self.tip(T)
        if min(self.pad_dist_mm(T[:3, 3]), self.pad_dist_mm(tip)) < NEAR_MM: return True
        if self.needle is not None and np.linalg.norm(tip - self.needle) * 1e3 < NEAR_MM: return True
        return False

    # ------------------------------------------------------------------ checks
    def check_target(self, T, contact_ok=False):
        js = self._need("js"); suj = self._need("suj")
        ok, q, why = self.kin.check_pose(T[:3, 3], T[:3, :3], js[:6], suj, min_deg=3.0, min_mm=5.0)
        if not ok: raise Guard("IK / joint margin: " + why)
        tip = self.tip(T)
        if self.cube is not None:
            c, h = self.cube
            for nm, p in (("tip", tip),):                    # the grasp point; the hinge is ~24 mm behind it along the shaft (reach_real checks the hinge itself)
                if np.any(np.abs(p - c) > h): raise Guard(f"workspace cube: {nm} {np.round((p - c) * 1e3, 1).tolist()} mm from centre (half {h * 1e3:.0f})")
        if self.pad is not None and contact_ok not in PAD_OFF:
            dh, dt = self.pad_dist_mm(T[:3, 3]), self.pad_dist_mm(tip); lim_h = lim_t = PAD_MIN_MM
            if contact_ok == "away":                   # moving off a contact: allowed below 3 mm only if not getting closer
                S = self.setpoint(); lim_h = min(PAD_MIN_MM, self.pad_dist_mm(S[:3, 3]) - 0.05); lim_t = min(PAD_MIN_MM, self.pad_dist_mm(self.tip(S)) - 0.05)
            if dh < lim_h or dt < lim_t: raise Guard(f"pad clearance hinge {dh:.1f} / tip {dt:.1f} mm < {PAD_MIN_MM} (mode {contact_ok})")
        return q

    # ------------------------------------------------------------------ one command
    def step(self, T, lim, contact_ok=False, tag=""):
        """one move_cp to T (ECM 4x4, metres) after all pre-checks; verifies arrival.  Returns measured pose."""
        self.state_ok()
        sp0 = self.setpoint(); meas0 = self.measured()
        dmm = float(np.linalg.norm(T[:3, 3] - sp0[:3, 3]) * 1e3); ddeg = geodesic_deg(sp0[:3, :3], T[:3, :3])
        if dmm > lim[0] + 1e-3 or ddeg > lim[1] + 1e-3: raise Guard(f"step {dmm:.2f} mm / {ddeg:.2f} deg exceeds the active limit {lim}")
        q = self.check_target(T, contact_ok)
        frame = self.io.get("sp")[2] or "ECM"
        self.io.clear_reached(); self.io.publish_move_cp(frame, T); self.n_cmd += 1
        t0 = time.monotonic()
        while self.io.get("reached") is None and time.monotonic() - t0 < 10.0: time.sleep(0.01)
        r = self.io.get("reached")
        if r is None or not r[1]: raise Guard("goal_reached not received / false within 10 s")
        time.sleep(self.post_s); t_after = time.monotonic()
        sp1 = self.setpoint()
        e_sp = float(np.linalg.norm(sp1[:3, 3] - T[:3, 3]) * 1e3); r_sp = geodesic_deg(sp1[:3, :3], T[:3, :3])
        rec = dict(ev="step", tag=tag, lim=list(lim), dmm=dmm, ddeg=ddeg, target=T.tolist(), sp_err_mm=e_sp, sp_err_deg=r_sp, contact_ok=contact_ok)
        if e_sp > 0.3 or r_sp > 0.3: self.log(rec); raise Guard(f"setpoint did not reach the target (IK clipped?) {e_sp:.2f} mm / {r_sp:.2f} deg")
        m = self.measured(); e_m = float(np.linalg.norm(m[:3, 3] - T[:3, 3]) * 1e3); r_m = geodesic_deg(m[:3, :3], T[:3, :3])
        js = self._need("js"); spj = self._need("spj"); jerr = js[:6] - spj[:6]
        rec.update(meas_err_mm=e_m, meas_err_deg=r_m, joint_err=jerr.tolist(), q=js[:6].tolist(), eff=(self.io.get("jse") or (0, np.zeros(6)))[1][:6].tolist())
        self.log(rec)
        if e_m > 2.0 or r_m > 2.0: raise Guard(f"measured pose not at the target: {e_m:.2f} mm / {r_m:.2f} deg")
        if contact_ok not in TRACK_OFF and (abs(jerr[2]) * 1e3 > 0.5 or np.degrees(np.abs(np.r_[jerr[:2], jerr[3:6]])).max() > 1.0):
            raise Guard(f"joint tracking error (unexpected contact?) ins {jerr[2] * 1e3:.2f} mm, others {np.degrees(np.abs(np.r_[jerr[:2], jerr[3:6]])).max():.2f} deg")
        return m

    # ------------------------------------------------------------------ paths
    def move_to(self, T_goal, tag="", contact_ok=False, lim=None, max_total_mm=80.0):
        """straight line (position) + slerp (orientation) from the current setpoint, sub-steps sized by the active limit at each
        sub-step (near -> 2 mm / 3 deg, free -> 5 mm / 5 deg; lim overrides).  Every sub-step goes through step()."""
        T0 = self.setpoint(); tot = np.linalg.norm(T_goal[:3, 3] - T0[:3, 3]) * 1e3
        if tot > max_total_mm: raise Guard(f"refusing a {tot:.1f} mm move (> {max_total_mm} mm)")
        n_guard = 0
        while True:
            S = self.setpoint(); dmm = float(np.linalg.norm(T_goal[:3, 3] - S[:3, 3]) * 1e3); ddeg = geodesic_deg(S[:3, :3], T_goal[:3, :3])
            if dmm < 0.05 and ddeg < 0.05: return self.measured()
            L = lim or (LIM_NEAR if (self.is_near(S) or self.is_near(T_goal)) else LIM_FREE)
            f = min(1.0, L[0] / max(dmm, 1e-9), L[1] / max(ddeg, 1e-9))
            sl = Slerp([0, 1], Rotation.from_matrix(np.stack([S[:3, :3], T_goal[:3, :3]])))
            Tn = T_of(S[:3, 3] + f * (T_goal[:3, 3] - S[:3, 3]), sl(f).as_matrix())
            if lim is None and f < 1.0 and self.is_near(Tn) and L == LIM_FREE:
                L = LIM_NEAR; f = min(1.0, L[0] / max(dmm, 1e-9), L[1] / max(ddeg, 1e-9))
                Tn = T_of(S[:3, 3] + f * (T_goal[:3, 3] - S[:3, 3]), sl(f).as_matrix())
            self.step(Tn, L, contact_ok=contact_ok, tag=tag)
            n_guard += 1
            if n_guard > 400: raise Guard("move_to did not converge in 400 sub-steps")

    def precheck_path(self, T_goal, n=20):
        """IK / cube / pad check of n interpolated poses from the current setpoint WITHOUT moving.  Returns list of problems."""
        S = self.setpoint(); sl = Slerp([0, 1], Rotation.from_matrix(np.stack([S[:3, :3], T_goal[:3, :3]]))); bad = []
        for f in np.linspace(0, 1, n + 1)[1:]:
            T = T_of(S[:3, 3] + f * (T_goal[:3, 3] - S[:3, 3]), sl(f).as_matrix())
            try: self.check_target(T)
            except Guard as e: bad.append(f"f={f:.2f}: {e}")
        return bad

    def translate(self, d_m, tag="", contact_ok=False, lim=None):
        S = self.setpoint(); T = S.copy(); T[:3, 3] = S[:3, 3] + np.asarray(d_m, float)
        return self.move_to(T, tag=tag, contact_ok=contact_ok, lim=lim)

    def rotate_about_tip(self, R_goal, tag="", lim=None):
        """turn to R_goal keeping the TCP fixed (sub-steps via move_to on the hinge pose that keeps the tip)."""
        S = self.setpoint(); tip = self.tip(S)          # 14:0x: pivot on the camera-corrected tip (A11/A12: kinematic pivot swung it to the pad)
        T = T_of(tip - R_goal @ (TCP_MM * 1e-3 + self.tip_off_tool), R_goal)
        return self.move_to(T, tag=tag, lim=lim)

    def safe_goto(self, T_target, tag="", min_tip_mm=15.0, lim=None):
        """new-segment transfer that never swings the tip down: (1) lift along the pad normal until the tip is >= min_tip_mm above
        the pad, (2) rotate about the tip to the target orientation (tip fixed), (3) straight translation to the target."""
        L = lim or LIM_NEAR
        if self.pad is not None:
            S = self.setpoint(); h = self.pad_dist_mm(self.tip(S))
            if h < min_tip_mm:
                T = S.copy(); T[:3, 3] = S[:3, 3] + self.pad[0] * (min_tip_mm - h) * 1e-3
                self.move_to(T, tag=tag + "_lift", contact_ok="away", lim=L)
        self.rotate_about_tip(T_target[:3, :3], tag=tag + "_rotate", lim=L)
        S = self.setpoint(); T = T_target.copy(); T[:3, :3] = S[:3, :3]
        self.move_to(T, tag=tag + "_translate", lim=L, max_total_mm=150.0)
        return self.move_to(T_target, tag=tag + "_final", lim=L)

    # ------------------------------------------------------------------ contact search (R6 relative test)
    def descend_until_contact(self, axis_m, max_mm, tag="descend", step_mm=0.5, baseline="after_first", lim=None, use=("ins_effort", "j01_effort", "ins_track_mm"), early_steps=0, early_ins=None):
        """steps of 0.5 mm along axis (unit, ECM).  The effort baseline is taken after the first INSERTING step (friction is
        direction dependent, 10/26).  Contact = |d ins effort| > 0.15 or |d j0/j1 effort| > 0.05 or ins tracking > 0.3 mm.
        Returns (contact_step or None, rows)."""
        axis = np.asarray(axis_m, float) / np.linalg.norm(axis_m); rows = []; base = None
        if baseline == "before":                       # R6 r6_grasp touch: e0 = mean effort over 1 s before stepping
            _, base, _ = self.io.samples(1.0); rows.append(dict(step=0, baseline=base.tolist()))
        for k in range(int(round(max_mm / step_mm))):
            S = self.setpoint(); T = S.copy(); T[:3, 3] = S[:3, 3] + axis * step_mm * 1e-3
            self.step(T, lim or LIM_DESCENT, contact_ok=True, tag=tag)
            pos, eff, err = self.io.samples(0.3)
            if base is None: base = eff; rows.append(dict(step=k + 1, baseline=eff.tolist())); continue
            de = eff - base
            hit = dict(ins_effort=bool(abs(de[2]) > CONTACT["ins_effort"]), j01_effort=bool(max(abs(de[0]), abs(de[1])) > CONTACT["j01_effort"]),
                       ins_track_mm=bool(abs(err[2]) * 1e3 > CONTACT["ins_track_mm"]))
            rows.append(dict(step=k + 1, d_effort=de.tolist(), track_err=err.tolist(), hit=hit, tip_ecm=tip_of(self.measured()).tolist()))
            self.log(dict(ev="contact_probe", tag=tag, **rows[-1]))
            if k + 1 <= early_steps and abs(de[2]) <= early_ins: continue    # 2026-11-04 fix A (user approved): far above the pad only a clear hit counts (A12: d_ins 0.105 at step 3, ~7 mm above the needle)
            if any(hit[c] for c in use): return k + 1, rows                # 16:4x: 'use' = criteria that stop (all are logged)
        return None, rows

    # ------------------------------------------------------------------ jaw (grasp protocol only)
    def jaw_to(self, q_target, n_steps=10, dt=0.3, tag="jaw"):
        j0 = float(self._need("jaw", max_age=1.0)[0])
        for q in np.linspace(j0, q_target, n_steps + 1)[1:]:
            self.state_ok(); self.io.publish_jaw(q); time.sleep(0 if self.fake else dt)
        time.sleep(0 if self.fake else 1.0)
        j1 = float(self._need("jaw", max_age=1.0)[0]); je = self.io.get("jaw_eff")
        self.log(dict(ev="jaw", tag=tag, from_rad=j0, to_rad=q_target, measured_rad=j1, effort=None if je is None else je[1].tolist())); return j1

    def jaw_sample(self, n=20, dt=0.05):
        """15:3x: median jaw position / effort over n samples (grasp check)."""
        q, e = [], []
        for _ in range(n):
            q.append(float(self._need("jaw", max_age=1.0)[0])); je = self.io.get("jaw_eff"); e.append(float(je[1][0]) if je is not None and len(je[1]) else np.nan)
            time.sleep(0 if self.fake else dt)
        return dict(q=float(np.median(q)), eff=float(np.nanmedian(e)) if np.any(np.isfinite(e)) else None, eff_iqr=float(np.subtract(*np.nanpercentile(e, [75, 25]))) if np.any(np.isfinite(e)) else None)
