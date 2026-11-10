#!/usr/bin/env python3
"""Rotate the tool about an axis through the measured needle tip (fixed needle-in-tool over a small rotation).  Axis = site normal n by default; +deg follows the right-hand rule about n.  Dry run by default."""
import argparse, json, os, sys, time
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Ro
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", R0 + "/P_place", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K, needle3d as N3
from ncc_disp import disp, FB
from rig import RigIO, Mover, load_kin, Guard, FX, CX, CY
from calib import rotation_from_affine, cam2ecm_mm
from sensor_msgs.msg import JointState
ap = argparse.ArgumentParser(); ap.add_argument("--measure", required=True); ap.add_argument("--deg", type=float, required=True); ap.add_argument("--axis", type=float, nargs=3, default=None, help="camera-frame axis instead of n"); ap.add_argument("--tipR", type=float, nargs=2, required=True); ap.add_argument("--body", type=float, nargs=2, required=True, help="right-image needle point near the tip, used for stereo depth"); ap.add_argument("--tipZ", type=float, default=None); ap.add_argument("--frames", required=True); ap.add_argument("--out", required=True); ap.add_argument("--send", action="store_true"); a = ap.parse_args()
assert abs(a.deg) <= 35; os.makedirs(a.out, exist_ok=True); cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); pad = json.load(open(R0 + "/D_pad/pad.json")); M = np.array(cal["M"]); Rc = rotation_from_affine(M); SUJ = np.array(cal["frozen"]["suj_psm1"]); s = N3.site(json.load(open(a.measure))["stickers"])
R = cv2.imread(a.frames + "/right_rect.png"); L = cv2.imread(a.frames + "/left_rect.png"); r = disp(R, L, a.body[0], a.body[1], d0=FB / 100, rng=30); Z = a.tipZ or FB / r[1]; print("depth used %.1f; body point ncc %.2f disp %.1f" % (Z, r[0], r[1])); tip = np.array([(a.tipR[0] - CX) * Z / FX, (a.tipR[1] - CY) * Z / FX, Z]); ax = np.array(a.axis, float) if a.axis else s["n"]; ax = ax / np.linalg.norm(ax); print("tip cam", np.round(tip, 1).tolist(), "axis cam", np.round(ax, 3).tolist(), "deg", a.deg)
logf = open(a.out + "/log.jsonl", "a"); log = lambda x: (logf.write(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"), logf.flush())
io = RigIO(images=True); kin = load_kin(); time.sleep(2.5); mv = Mover(io, kin, log); mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = SUJ; mv.pad = (np.array(pad["normal"]), pad["d"]); mv.state_ok(); pub = io.node.create_publisher(JointState, "/PSM1/move_jp", 1); time.sleep(0.5)
jsq = lambda: np.array(io.get("js")[1][:6]); eff = lambda: float(io.get("jaw_eff")[1][0]); jaw = lambda: float(io.get("jaw")[1][0])
try:
    q0 = jsq(); T0 = K.fk_ecm(q0, SUJ); tip_e = cam2ecm_mm(tip, M) * 1e-3; pt = T0[:3, :3].T @ (tip_e - T0[:3, 3]); Rot_e = Rc.T @ Ro.from_rotvec(ax * np.radians(a.deg)).as_matrix() @ Rc; Rg = Rot_e @ T0[:3, :3]; pg = tip_e - Rg @ pt; q, ep, er = K.ik_ecm(pg, Rg, q0, SUJ); m = K.margins(q); print("IK residual %.3f mm %.3f deg; wrist deg %s -> %s; margins %s" % (ep, er, np.round(np.degrees(q0[3:]), 1).tolist(), np.round(np.degrees(q[3:]), 1).tolist(), np.round(m, 1).tolist()), flush=True)
    if ep > 0.05 or er > 0.05 or m[3:].min() < 3 or m[:2].min() < 3 or m[2] < 5: raise Guard("IK / margins")
    n = max(3, int(np.ceil(np.degrees(np.abs(q - q0)[[0, 1, 3, 4, 5]]).max() / 2.0))); print("dry-run checks passed,", n, "steps", flush=True)
    if a.send:
        for k in range(1, n + 1):
            mv.state_ok(); qk = q0 + (q - q0) * k / n; m_ = JointState(); m_.position = qk.tolist(); pub.publish(m_); t0 = time.time()
            while time.time() - t0 < 6.0:
                time.sleep(0.05)
                if np.degrees(np.abs(jsq() - qk)[[0, 1, 3, 4, 5]]).max() < 1.0 and abs(jsq()[2] - qk[2]) < 5e-4: break
            else: raise Guard(f"joints did not settle at {k}/{n}")
            time.sleep(0.1); log(dict(ev="step", k=k, q=jsq(), eff=eff()))
            if abs(eff()) < 0.12: raise Guard(f"jaw effort {eff():.3f}")
        time.sleep(0.6); b = io.frames(8); cv2.imwrite(a.out + "/left_rect.png", np.mean([z[0] for z in b], 0).round().astype(np.uint8)); cv2.imwrite(a.out + "/right_rect.png", np.mean([z[1] for z in b], 0).round().astype(np.uint8)); json.dump(dict(q=jsq().tolist(), jaw=jaw(), eff=eff()), open(a.out + "/state.json", "w")); print("done: wrist deg %s jaw %.3f eff %.3f" % (np.round(np.degrees(jsq()[3:]), 1).tolist(), jaw(), eff()), flush=True)
except BaseException as e:
    import traceback; traceback.print_exc(); print("STOPPED:", e, flush=True); log(dict(ev="stop", why=str(e)))
sys.stdout.flush(); os._exit(0)
