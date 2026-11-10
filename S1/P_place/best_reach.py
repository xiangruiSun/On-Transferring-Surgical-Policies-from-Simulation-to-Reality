#!/usr/bin/env python3
"""2026-11-05 (setting 1 trial 3): the learned placing orientation is unreachable with this grip (wrist yaw limit).
Go to the closest reachable orientation (least-squares orientation error, joint margins >= 3 deg / 5 mm) with the needle tip
--hover mm above the learned tip target, by a joint-space path (<= 4 deg per step). Dry run by default.
Guards as in place_repro.go: ECM/SUJ freeze + ENABLED (Mover.state_ok), joint settling, jaw effort >= 0.12, path clearance >= --min-clear."""
import argparse, json, os, sys, time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Ro
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", R0 + "/P_place", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K, needle3d as N3
from rig import RigIO, Mover, load_kin, Guard, FX, CX, CY, FB, TCP_MM
from calib import rotation_from_affine, cam2ecm_mm
from sensor_msgs.msg import JointState
ap = argparse.ArgumentParser(); ap.add_argument("--measure", required=True); ap.add_argument("--grip", required=True); ap.add_argument("--fssnap", required=True); ap.add_argument("--hover", type=float, default=15.0)
ap.add_argument("--min-clear", type=float, default=3.0); ap.add_argument("--out", required=True); ap.add_argument("--send", action="store_true"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
L = json.load(open("learned.json")); cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); pad = json.load(open(R0 + "/D_pad/pad.json")); M = np.array(cal["M"]); Rc = rotation_from_affine(M); SUJ = np.array(cal["frozen"]["suj_psm1"]); disp = np.load(a.fssnap + "/fs_disp.npy")
s = N3.site(json.load(open(a.measure))["stickers"]); B = np.c_[s["u"], s["m"], s["n"]]; g = json.load(open(a.grip)); Rct = B @ np.array(L["R_site_tool"]) @ np.array(g["R_delta"]); Rt = Rc.T @ Rct
so = L["site"]; So = np.c_[so["u"], np.cross(so["n"], so["u"]), so["n"]]; tip_t = s["e"] + B @ (So.T @ (np.array(L["tip_cam_rough"]) - np.array(so["e"]))); p_t = np.array(g["tip_tool_mm"]) * 1e-3; goal = cam2ecm_mm(tip_t + a.hover * s["n"], M) * 1e-3
logf = open(a.out + "/log.jsonl", "a"); log = lambda r: (logf.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"), logf.flush())
io = RigIO(images=True); kin = load_kin(); time.sleep(2.5); mv = Mover(io, kin, log); mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = SUJ; mv.pad = (np.array(pad["normal"]), pad["d"]); mv.state_ok()
pub = io.node.create_publisher(JointState, "/PSM1/move_jp", 1); time.sleep(0.5)
jsq = lambda: np.array(io.get("js")[1][:6]); eff = lambda: float(io.get("jaw_eff")[1][0])
def clear(q):
    T = K.fk_ecm(q, SUJ); best = 1e3
    for f in np.linspace(0, 1.15, 24):
        X = np.r_[(T[:3, 3] + T[:3, :3] @ (TCP_MM * 1e-3 * f)) * 1e3, 1] @ M; u, v = FX * X[0] / X[2] + CX, FX * X[1] / X[2] + CY
        if 0 <= u < disp.shape[1] and 0 <= v < disp.shape[0]: best = min(best, FB / disp[int(v), int(u)] - X[2])
    for f in np.linspace(0, 1, 12):                                     # needle tip too (tool-frame point from the grip fit)
        X = np.r_[(T[:3, 3] + T[:3, :3] @ (p_t * f + TCP_MM * 1e-3 * (1 - f))) * 1e3, 1] @ M; u, v = FX * X[0] / X[2] + CX, FX * X[1] / X[2] + CY
        if 0 <= u < disp.shape[1] and 0 <= v < disp.shape[0]: best = min(best, FB / disp[int(v), int(u)] - X[2])
    return float(best)
try:
    q0 = jsq(); lo = K.QMIN + np.r_[np.radians([3, 3]), 0.005, np.radians([3, 3, 3])]; hi = K.QMAX - np.r_[np.radians([3, 3]), 0.005, np.radians([3, 3, 3])]; best = None
    f = lambda q: np.r_[((K.fk_ecm(q, SUJ)[:3, 3] + K.fk_ecm(q, SUJ)[:3, :3] @ p_t) - goal) * 1e3 * 20, Ro.from_matrix(Rt.T @ K.fk_ecm(q, SUJ)[:3, :3]).as_rotvec() * 57.3]
    for r0 in np.radians(np.arange(-240, 241, 40)):
        for p0 in np.radians([-60, 0, 60]):
            for y0 in np.radians([-60, 0, 60]):
                r = least_squares(f, np.clip(np.r_[q0[:3], r0, p0, y0], lo, hi), bounds=(lo, hi), max_nfev=300); T = K.fk_ecm(r.x, SUJ)
                ep = np.linalg.norm(T[:3, 3] + T[:3, :3] @ p_t - goal) * 1e3; er = np.degrees(Ro.from_matrix(Rt.T @ T[:3, :3]).magnitude()); cost = er + 0.02 * np.degrees(abs(r.x[3] - q0[3]))
                if ep < 0.2 and (best is None or cost < best[0]): best = (cost, r.x, er)
    if best is None: raise Guard("no reachable pose")
    q1, er = best[1], best[2]; n = max(2, int(np.ceil(np.degrees(np.abs(q1 - q0)[[0, 1, 3, 4, 5]]).max() / 4.0))); P = [q0 + (q1 - q0) * k / n for k in range(1, n + 1)]; c = min(clear(q) for q in P)
    print("now wrist deg %s ins %.1f clearance %.1f | target: orientation error %.1f deg, wrist deg %s ins %.1f margins %s | path %d steps, min clearance %.1f" % (np.round(np.degrees(q0[3:]), 1).tolist(), q0[2] * 1e3, clear(q0), er, np.round(np.degrees(q1[3:]), 1).tolist(), q1[2] * 1e3, np.round(K.margins(q1), 1).tolist(), n, c), flush=True)
    log(dict(ev="plan", q0=q0, q1=q1, err_deg=er, n=n, min_clear=c))
    if c < a.min_clear: raise Guard(f"path clearance {c:.1f} < {a.min_clear}")
    print("dry-run checks passed", flush=True)
    if a.send:
        for k, qk in enumerate(P, 1):
            mv.state_ok(); m = JointState(); m.position = qk.tolist(); pub.publish(m); t0 = time.time()
            while time.time() - t0 < 6.0:
                time.sleep(0.05)
                if np.degrees(np.abs(jsq() - qk)[[0, 1, 3, 4, 5]]).max() < 1.0 and abs(jsq()[2] - qk[2]) < 5e-4: break
            else: raise Guard(f"joints did not settle at {k}/{n}")
            time.sleep(0.1); log(dict(ev="step", k=k, q=jsq(), eff=eff()))
            if abs(eff()) < 0.12: raise Guard(f"jaw effort {eff():.3f}: needle may be loose")
        time.sleep(0.6); print("reached: wrist deg %s ins %.1f jaw eff %.3f" % (np.round(np.degrees(jsq()[3:]), 1).tolist(), jsq()[2] * 1e3, eff()), flush=True)
except BaseException as e:
    import traceback; traceback.print_exc(); print("STOPPED:", e, flush=True); log(dict(ev="stop", why=str(e)))
sys.stdout.flush(); os._exit(0)
