#!/usr/bin/env python3
"""Placing phase 2: bring the measured needle tip to entry + h n, angle held as closely as the joint limits allow.  Tip = hand-checked stereo pixels, fixed in the tool frame.
IK: exact tip position, soft orientation, wrist margins >= 3 deg.  Small joint steps.  Dry run by default.  Stops: state/ECM/SUJ, settling, jaw effort, insertion-effort jump, orientation change limit."""
# 2026-11-10 18:4x COPY of S1/P_place/phase2.py (which stays unchanged); only change: settle tolerance of the wrist pitch / yaw joints 1.0 -> SETTLE_WRIST_DEG (default 2.5) deg.
# Why: A188 / A189 held the needle at the jaw tips with jaw effort -0.34..-0.38; wrist yaw then sits 1.03 deg from its setpoint (steady state, yaw effort -0.15) -> "joints did not settle" at the first steps of every rotation.
import os as _os, numpy as _np
SETTLE_TOL = _np.array([1.0, 1.0, 1.0, float(_os.environ.get("SETTLE_WRIST_DEG", "2.5")), float(_os.environ.get("SETTLE_WRIST_DEG", "2.5"))])
import argparse, json, os, sys, time
import numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Ro
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K
from rig import RigIO, Mover, load_kin, Guard, FX, CX, CY, FB
from calib import cam2ecm_mm
from sensor_msgs.msg import JointState
ap = argparse.ArgumentParser(); ap.add_argument("--tipR", type=float, nargs=2, required=True); ap.add_argument("--tipL", type=float, nargs=2, required=True); ap.add_argument("--h", type=float, required=True); ap.add_argument("--measure", default="m0/measure.json"); ap.add_argument("--along", type=float, default=0.0, help="mm along entry->exit from the sticker centre")
ap.add_argument("--frac", type=float, default=1.0); ap.add_argument("--max-move", type=float, default=25.0); ap.add_argument("--max-rot", type=float, default=10.0); ap.add_argument("--ins-eff", type=float, default=0.30); ap.add_argument("--out", required=True); ap.add_argument("--send", action="store_true"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); pad = json.load(open(R0 + "/D_pad/pad.json")); M = np.array(cal["M"]); SUJ = np.array(cal["frozen"]["suj_psm1"]); M3i = np.linalg.inv(M[:3])
st = json.load(open(a.measure))["stickers"]; en, ex = (st[1], st[0]) if st[1]["area"] < st[0]["area"] else (st[0], st[1]); e, xx = np.array(en["cam"]), np.array(ex["cam"]); n = np.array(en["n25"]); n = n if n[2] < 0 else -n; u = (xx - e) - ((xx - e) @ n) * n; u /= np.linalg.norm(u)
logf = open(a.out + "/log.jsonl", "a"); log = lambda r: (logf.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"), logf.flush())
io = RigIO(images=True); kin = load_kin(); time.sleep(2.5); mv = Mover(io, kin, log); mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = SUJ; mv.pad = (np.array(pad["normal"]), pad["d"]); mv.state_ok()
pub = io.node.create_publisher(JointState, "/PSM1/move_jp", 1); time.sleep(0.5)
jsq = lambda: np.array(io.get("js")[1][:6]); eff = lambda: float(io.get("jaw_eff")[1][0]); jaw = lambda: float(io.get("jaw")[1][0]); ins_e = lambda: float(io.get("jse")[1][2]) if io.get("jse") else float("nan")
try:
    q0 = jsq(); T0 = K.fk_ecm(q0, SUJ); dR = a.tipR[0] - a.tipL[0]; Z = FB / dR; tip = np.array([(a.tipR[0] - CX) * Z / FX, (a.tipR[1] - CY) * Z / FX, Z]); goal = tip + a.frac * ((e + a.along * u + a.h * n) - tip); dc = goal - tip
    print("tip cam %s (disp %.1f, row diff %.1f); goal cam %s; move %.2f mm cam; height above entry level now %.1f -> %.1f" % (np.round(tip, 2).tolist(), dR, a.tipL[1] - a.tipR[1], np.round(goal, 2).tolist(), np.linalg.norm(dc), (tip - e) @ n, (goal - e) @ n), flush=True)
    if np.linalg.norm(dc) > a.max_move: raise Guard(f"move > {a.max_move} mm")
    de = dc @ M3i * 1e-3; p_tip_e = cam2ecm_mm(tip, M) * 1e-3; p_t = T0[:3, :3].T @ (p_tip_e - T0[:3, 3]); goal_e = p_tip_e + de
    lo = K.QMIN + np.r_[np.radians([5, 5]), 0.01, np.radians([3, 3, 3])]; hi = K.QMAX - np.r_[np.radians([5, 5]), 0.01, np.radians([3, 3, 3])]
    def res(q): T = K.fk_ecm(q, SUJ); return np.r_[((T[:3, 3] + T[:3, :3] @ p_t) - goal_e) * 1e3 * 20, Ro.from_matrix(T0[:3, :3].T @ T[:3, :3]).as_rotvec() * 57.3 * 0.5]
    s = least_squares(res, np.clip(q0, lo, hi), bounds=(lo, hi), max_nfev=200); q = s.x; T = K.fk_ecm(q, SUJ); ep = np.linalg.norm((T[:3, 3] + T[:3, :3] @ p_t) - goal_e) * 1e3; rot = np.degrees(Ro.from_matrix(T0[:3, :3].T @ T[:3, :3]).magnitude())
    print("IK: tip residual %.3f mm; orientation change %.1f deg; wrist deg %s -> %s; ins %.1f -> %.1f; margins %s" % (ep, rot, np.round(np.degrees(q0[3:]), 1).tolist(), np.round(np.degrees(q[3:]), 1).tolist(), q0[2] * 1e3, q[2] * 1e3, np.round(K.margins(q), 1).tolist()), flush=True)
    if ep > 0.2: raise Guard(f"tip residual {ep:.2f} mm")
    if rot > a.max_rot: raise Guard(f"orientation change {rot:.1f} deg > {a.max_rot}")
    print("dry-run checks passed", flush=True)
    if a.send:
        N = max(4, int(np.ceil(np.linalg.norm(dc) / 0.5))); e0 = ins_e(); t00 = time.time()
        for k in range(1, N + 1):
            mv.state_ok(); qk = q0 + (q - q0) * k / N; m = JointState(); m.position = qk.tolist(); pub.publish(m); t0 = time.time()
            while time.time() - t0 < 6.0:
                time.sleep(0.05)
                if (np.degrees(np.abs(jsq() - qk)[[0, 1, 3, 4, 5]]) < SETTLE_TOL).all() and abs(jsq()[2] - qk[2]) < 5e-4: break
            else: raise Guard(f"joints did not settle at step {k}")
            time.sleep(0.12); log(dict(ev="step", k=k, q=jsq(), eff=eff(), ins=ins_e()))
            if abs(eff()) < 0.12: raise Guard(f"jaw effort {eff():.3f}: needle may be loose")
            if k == 2: e0 = ins_e()                 # baseline after the direction reversal (start-up friction), as in Mover's after_first
            if k > 2 and abs(ins_e() - e0) > a.ins_eff: raise Guard(f"insertion effort change {ins_e() - e0:.3f} at step {k}/{N}")
        time.sleep(0.6); b = io.frames(8); L = np.mean([z[0] for z in b], 0).round().astype(np.uint8); R = np.mean([z[1] for z in b], 0).round().astype(np.uint8); cv2.imwrite(a.out + "/left_rect.png", L); cv2.imwrite(a.out + "/right_rect.png", R)
        Xp = goal; up = (FX * Xp[0] / Xp[2] + CX, FX * Xp[1] / Xp[2] + CY); json.dump(dict(q=jsq().tolist(), cp=np.asarray(io.get("cp")[1]).tolist(), jaw=jaw(), eff=eff(), goal=goal.tolist(), goal_uv_R=up, tip_prev=tip.tolist()), open(a.out + "/state.json", "w"), indent=1)
        print("reached in %d steps; wrist deg %s; jaw %.3f eff %.3f; ins effort change %.3f; expected tip at right %s left %s" % (N, np.round(np.degrees(jsq()[3:]), 1).tolist(), jaw(), eff(), ins_e() - e0, np.round(up, 0).tolist(), np.round((up[0] - FB / Xp[2], up[1] + 2.2), 0).tolist()), flush=True)
except BaseException as ex2:
    import traceback; traceback.print_exc(); print("STOPPED:", ex2, flush=True); log(dict(ev="stop", why=str(ex2)))
sys.stdout.flush(); os._exit(0)
