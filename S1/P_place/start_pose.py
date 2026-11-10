#!/usr/bin/env python3
"""2026-11-05: move PSM1 (needle held) to a new trial start pose given in joints: j0 j1 (rad) ins (mm) roll pitch yaw (deg).
Route: (1) retract insertion to --ins-safe, (2) joint move of j0, j1 and the wrist at that insertion, (3) insert to the target.
Checks (dry run by default): joint margins >= 3 deg / 5 mm, tool + held needle clearance from the pad-only depth >= --min-clear on
every step, hinge inside the image. Guards while moving: Mover.state_ok (ENABLED, ECM/SUJ freeze), joint settling, jaw effort >= 0.12."""
import argparse, json, os, sys, time
import numpy as np
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", R0 + "/P_place", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K
from rig import RigIO, Mover, load_kin, Guard, FX, CX, CY, FB, TCP_MM
from sensor_msgs.msg import JointState
ap = argparse.ArgumentParser(); ap.add_argument("q", type=float, nargs=6); ap.add_argument("--grip", required=True); ap.add_argument("--fssnap", required=True); ap.add_argument("--ins-safe", type=float, default=95.0)
ap.add_argument("--min-clear", type=float, default=5.0); ap.add_argument("--out", required=True); ap.add_argument("--send", action="store_true"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); pad = json.load(open(R0 + "/D_pad/pad.json")); M = np.array(cal["M"]); SUJ = np.array(cal["frozen"]["suj_psm1"]); disp = np.load(a.fssnap + "/fs_disp.npy"); p_t = np.array(json.load(open(a.grip))["tip_tool_mm"]) * 1e-3
logf = open(a.out + "/log.jsonl", "a"); log = lambda r: (logf.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"), logf.flush())
io = RigIO(images=False); kin = load_kin(); time.sleep(2.5); mv = Mover(io, kin, log); mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = SUJ; mv.pad = (np.array(pad["normal"]), pad["d"]); mv.state_ok()
pub = io.node.create_publisher(JointState, "/PSM1/move_jp", 1); time.sleep(0.5)
jsq = lambda: np.array(io.get("js")[1][:6]); eff = lambda: float(io.get("jaw_eff")[1][0])
def uvz(p): X = np.r_[p * 1e3, 1] @ M; return FX * X[0] / X[2] + CX, FX * X[1] / X[2] + CY, X[2]
def clear(q):
    T = K.fk_ecm(q, SUJ); best = 1e3
    pts = [T[:3, 3] + T[:3, :3] @ (TCP_MM * 1e-3 * f) for f in np.linspace(0, 1.15, 24)] + [T[:3, 3] + T[:3, :3] @ (p_t * f + TCP_MM * 1e-3 * (1 - f)) for f in np.linspace(0, 1, 12)]
    for p in pts:
        u, v, z = uvz(p)
        if z > 25 and 0 <= u < disp.shape[1] and 0 <= v < disp.shape[0]: best = min(best, FB / disp[int(v), int(u)] - z)
    return float(best)
def seg(q0, q1, maxdeg=4.0, maxmm=2.0):
    d = np.abs(q1 - q0); n = max(1, int(np.ceil(max(np.degrees(d[[0, 1, 3, 4, 5]]).max() / maxdeg, d[2] * 1e3 / maxmm)))); return [q0 + (q1 - q0) * k / n for k in range(1, n + 1)]
try:
    qt = np.r_[a.q[0], a.q[1], a.q[2] * 1e-3, np.radians(a.q[3:])]; m = K.margins(qt); q0 = jsq()
    if m[3:].min() < 3 or m[:2].min() < 3 or m[2] < 5: raise Guard(f"target margins {np.round(m, 1).tolist()}")
    s = a.ins_safe * 1e-3; qa = q0.copy(); qa[2] = min(q0[2], s); qb = qt.copy(); qb[2] = min(qt[2], s)
    P = seg(q0, qa) + seg(qa, qb) + seg(qb, qt); c = [clear(q) for q in P]; u, v, z = uvz(K.fk_ecm(qt, SUJ)[:3, 3])
    k = int(np.argmin(c)); print("start pose: q %s | margins %s | hinge uv (%.0f, %.0f) Z %.0f | clearance at start %.1f, path min %.1f at step %d/%d (segments %d/%d/%d)" % (np.round(a.q, 3).tolist(), np.round(m, 1).tolist(), u, v, z, clear(qt), min(c), k + 1, len(P), len(seg(q0, qa)), len(seg(qa, qb)), len(seg(qb, qt))), flush=True)
    log(dict(ev="plan", q0=q0, qt=qt, min_clear=min(c), n=len(P)))
    if min(c) < a.min_clear: raise Guard(f"path clearance {min(c):.1f} < {a.min_clear}")
    if not (0 <= u < 1300 and 0 <= v < 1024): raise Guard("hinge outside the image")
    print("dry-run checks passed", flush=True)
    if a.send:
        k3 = len(P) - len(seg(qb, qt))
        for k, qk in enumerate(P, 1):
            mv.state_ok(); msg = JointState(); msg.position = qk.tolist(); pub.publish(msg); t0 = time.time()
            while time.time() - t0 < 6.0:
                time.sleep(0.05)
                if np.degrees(np.abs(jsq() - qk)[[0, 1, 3, 4, 5]]).max() < 1.0 and abs(jsq()[2] - qk[2]) < 5e-4: break
            else: raise Guard(f"joints did not settle at {k}/{len(P)}")
            log(dict(ev="step", k=k, q=jsq(), eff=eff()))
            if abs(eff()) < 0.12: raise Guard(f"jaw effort {eff():.3f}: needle may be loose")
            ie = float(io.get("jse")[1][2]) if io.get("jse") else 0.0          # 2026-11-05: insertion-effort contact guard (two start moves pressed into tissue, +3.5)
            if k == k3 + 2: e0 = ie                                              # guard only on the final insertion segment
            if k > k3 + 2 and ie - e0 > 0.6: raise Guard(f"insertion effort rise {ie - e0:.2f} at {k}/{len(P)}: contact, stopped")
        time.sleep(0.5); print("reached start: q %s jaw eff %.3f" % (np.round(np.r_[jsq()[:2], jsq()[2] * 1e3, np.degrees(jsq()[3:])], 2).tolist(), eff()), flush=True)
except BaseException as e:
    import traceback; traceback.print_exc(); print("STOPPED:", e, flush=True); log(dict(ev="stop", why=str(e)))
sys.stdout.flush(); os._exit(0)
