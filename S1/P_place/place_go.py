#!/usr/bin/env python3
"""Placing: joint-space move to the target needle pose given by the fixed needle-in-tool model, in stages with a stereo snapshot + overlay at each stage.
Dry run by default.  Guards: margins, ECM/SUJ freeze, joints must settle, jaw effort, clearance from the FS surface along the path."""
import argparse, json, os, sys, time
import numpy as np, cv2
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", R0 + "/P_place", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K
from rig import RigIO, Mover, load_kin, Guard, FX, CX, CY, FB
import plan3 as P3
from sensor_msgs.msg import JointState
ap = argparse.ArgumentParser(); ap.add_argument("--fit", default="w1/joint_fit.json"); ap.add_argument("--key", default="B const wrist offset"); ap.add_argument("--measure", default="m0/measure.json"); ap.add_argument("--fssnap", default="../G_place/now_1104a/snap")
ap.add_argument("--a", type=float, default=46.8); ap.add_argument("--dh", type=float, default=0.0); ap.add_argument("--h", type=float, default=12.0); ap.add_argument("--off", type=float, nargs=3, default=[0, 0, 0], help="camera mm added to the entry point")
ap.add_argument("--frac", type=float, nargs="+", default=[0.5, 1.0]); ap.add_argument("--min-clear", type=float, default=4.0); ap.add_argument("--out", required=True); ap.add_argument("--send", action="store_true"); ap.add_argument("--step", type=float, default=5.0); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); pad = json.load(open(R0 + "/D_pad/pad.json")); disp = np.load(a.fssnap + "/fs_disp.npy")
logf = open(a.out + "/log.jsonl", "a"); log = lambda r: (logf.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"), logf.flush())
io = RigIO(images=True); kin = load_kin(); time.sleep(2.5); mv = Mover(io, kin, log); mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = np.array(cal["frozen"]["suj_psm1"]); mv.pad = (np.array(pad["normal"]), pad["d"]); mv.state_ok()
pub = io.node.create_publisher(JointState, "/PSM1/move_jp", 1); time.sleep(0.5)
jsq = lambda: np.array(io.get("js")[1][:6]); eff = lambda: float(io.get("jaw_eff")[1][0]); jaw = lambda: float(io.get("jaw")[1][0])
def uvs(X): return np.c_[FX * X[:, 0] / X[:, 2] + CX, FX * X[:, 1] / X[:, 2] + CY]
def goto_q(q, tag):
    mv.state_ok(); q0 = jsq(); n = max(1, int(np.ceil(np.degrees(np.abs(q - q0)[[0, 1, 3, 4, 5]]).max() / a.step)))
    for k in range(1, n + 1):
        qk = q0 + (q - q0) * k / n; m = JointState(); m.position = qk.tolist(); pub.publish(m); t0 = time.time()
        while time.time() - t0 < 6.0:
            time.sleep(0.05)
            if np.degrees(np.abs(jsq() - qk)[[0, 1, 3, 4, 5]]).max() < 1.0 and abs(jsq()[2] - qk[2]) < 5e-4: break
        else: raise Guard(f"joints did not settle at {tag} sub-step {k}: err deg {np.round(np.degrees(jsq() - qk), 2).tolist()}")
        mv.state_ok(); log(dict(ev="q", tag=tag, k=k, q=jsq(), eff=eff()))
        if abs(eff()) < 0.12: raise Guard(f"jaw effort {eff():.3f}: needle may be loose")
    time.sleep(0.6)
def snap(tag, md, tgt):
    b = io.frames(8); L = np.mean([z[0] for z in b], 0).round().astype(np.uint8); R = np.mean([z[1] for z in b], 0).round().astype(np.uint8); d = f"{a.out}/{tag}"; os.makedirs(d, exist_ok=True)
    cv2.imwrite(d + "/left_rect.png", L); cv2.imwrite(d + "/right_rect.png", R); json.dump(dict(q=jsq().tolist(), cp=np.asarray(io.get("cp")[1]).tolist(), jaw=jaw(), eff=eff(), t=time.time()), open(d + "/state.json", "w"), indent=1)
    o = R.copy(); cv2.polylines(o, [np.int32(uvs(tgt))], False, (0, 255, 0), 2); cv2.circle(o, tuple(np.int32(uvs(tgt)[0])), 6, (0, 255, 0), 2); pr = uvs(md.cam(jsq())); cv2.polylines(o, [np.int32(pr)], False, (255, 0, 255), 2); cv2.circle(o, tuple(np.int32(pr[0])), 6, (255, 0, 255), 2); cv2.circle(o, tuple(np.int32(uvs(md.e[None])[0])), 8, (0, 0, 255), 2); cv2.imwrite(d + "/overlay.png", o)
    print(f"stage {tag}: wrist deg {np.round(np.degrees(jsq()[3:]), 1).tolist()} ins {jsq()[2] * 1e3:.1f} jaw {jaw():.3f} eff {eff():.3f}; model tip uv {np.round(pr[0]).tolist()} target tip uv {np.round(uvs(tgt)[0]).tolist()}", flush=True)
try:
    q_now = jsq(); md = P3.Model(a.fit, a.key, a.measure, q_now); tgt = md.target(a.a, a.dh, a.h, a.off); b = md.solve(tgt, seeds=[q_now, np.clip(q_now + np.r_[0, 0, 0, 0, 0, np.radians(-60)], K.QMIN + .06, K.QMAX - .06), np.clip(q_now + np.r_[0, 0, 0, np.radians(-30), 0, np.radians(-60)], K.QMIN + .06, K.QMAX - .06)])
    q = b[1]; print("now wrist deg", np.round(np.degrees(q_now[3:]), 1).tolist(), "jaw %.3f eff %.3f" % (jaw(), eff())); print("target a %.1f dh %.1f h %.1f: needle error max %.2f tip %.2f mm; wrist deg %s ins %.1f (now %.1f); margins %s" % (a.a, a.dh, a.h, b[0], b[2][0], np.round(np.degrees(q[3:]), 1).tolist(), q[2] * 1e3, q_now[2] * 1e3, np.round(K.margins(q), 1).tolist()), flush=True)
    if b[0] > 3.5: raise Guard("needle error > 3.5 mm: not sent")
    cl = [P3.clearance(md, q_now + (q - q_now) * f, disp) for f in np.linspace(0, 1, 21)]; print("clearance along the path jaw/needle (mm) min", min(c[0] for c in cl), min(c[1] for c in cl), "end", cl[-1], flush=True)
    if min(c[1] for c in cl) < a.min_clear or min(c[0] for c in cl) < a.min_clear: raise Guard("path clearance below %.1f" % a.min_clear)
    print("dry-run checks passed", flush=True)
    if a.send:
        for f in a.frac: goto_q(q_now + (q - q_now) * f, f"s{int(f * 100):03d}"); snap(f"s{int(f * 100):03d}", md, tgt)
        print("done", flush=True)
except BaseException as e:
    import traceback; traceback.print_exc(); print("STOPPED:", e, flush=True); log(dict(ev="stop", why=str(e)))
sys.stdout.flush(); os._exit(0)
