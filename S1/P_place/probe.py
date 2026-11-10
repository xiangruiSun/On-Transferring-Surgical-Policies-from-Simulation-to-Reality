#!/usr/bin/env python3
"""Placing, step 1: lift clear along the pad normal, then wrist-only poses in air, a stereo snapshot at each, to measure the needle-in-tool transform.
Default = dry run (prints every target, checks margins). --send moves.  Stops on: state/ECM/SUJ guard, joint not settling, jaw effort loss."""
import argparse, json, os, sys, time
import numpy as np, cv2
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K
from rig import RigIO, Mover, load_kin, Guard
from sensor_msgs.msg import JointState
ap = argparse.ArgumentParser(); ap.add_argument("--lift", type=float, default=20.0); ap.add_argument("--out", required=True); ap.add_argument("--send", action="store_true"); ap.add_argument("--no-lift", action="store_true"); ap.add_argument("--wrist", default=None); a = ap.parse_args()
assert 0 <= a.lift <= 30
WRIST = [(0, 0, 0), (25, 0, 0), (-25, 0, 0), (0, 20, 0), (0, -20, 0), (0, 0, 20), (0, 0, -20), (-20, -20, -25), (-35, -35, -45), (0, 0, 0)]   # roll, pitch, yaw offsets deg; the last-but-one leans toward the expected target
WRIST = json.loads(a.wrist) if a.wrist else WRIST
STEP_DEG = 6.0; SETTLE_DEG = 1.0; MIN_MARGIN = 3.0; EFF_MIN = 0.12
os.makedirs(a.out, exist_ok=True); cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); pad = json.load(open(R0 + "/D_pad/pad.json"))
logf = open(a.out + "/log.jsonl", "a"); log = lambda r: (logf.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"), logf.flush())
io = RigIO(images=True); kin = load_kin(); time.sleep(2.5); mv = Mover(io, kin, log)
mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = np.array(cal["frozen"]["suj_psm1"]); mv.pad = (np.array(pad["normal"]), pad["d"]); mv.state_ok()
pub = io.node.create_publisher(JointState, "/PSM1/move_jp", 1); time.sleep(0.5)
jsq = lambda: np.array(io.get("js")[1][:6]); eff = lambda: float(io.get("jaw_eff")[1][0]); jaw = lambda: float(io.get("jaw")[1][0])
def check_eff(tag):
    e = eff(); log(dict(ev="jaw", tag=tag, eff=e, jaw=jaw()))
    if abs(e) < EFF_MIN: raise Guard(f"jaw effort {e:.3f} below {EFF_MIN} at {tag}: needle may be loose")
def goto_q(q, tag):
    mv.state_ok(); q0 = jsq(); n = max(1, int(np.ceil(np.degrees(np.abs(q - q0)[[0, 1, 3, 4, 5]]).max() / STEP_DEG)))
    for k in range(1, n + 1):
        qk = q0 + (q - q0) * k / n; m = JointState(); m.position = qk.tolist(); pub.publish(m); t0 = time.time()
        while time.time() - t0 < 6.0:
            time.sleep(0.05)
            if np.degrees(np.abs(jsq() - qk)[[0, 1, 3, 4, 5]]).max() < SETTLE_DEG and abs(jsq()[2] - qk[2]) < 5e-4: break
        else: raise Guard(f"joints did not settle at {tag} sub-step {k}: err deg {np.round(np.degrees(jsq() - qk), 2).tolist()}")
        mv.state_ok(); log(dict(ev="q", tag=tag, k=k, q=jsq(), eff=eff()))
    time.sleep(0.5); check_eff(tag)
def snap(i):
    b = io.frames(8); L = np.mean([z[0] for z in b], 0).round().astype(np.uint8); R = np.mean([z[1] for z in b], 0).round().astype(np.uint8); d = f"{a.out}/v{i:02d}"; os.makedirs(d, exist_ok=True)
    cv2.imwrite(d + "/left_rect.png", L); cv2.imwrite(d + "/right_rect.png", R); json.dump(dict(q=jsq().tolist(), cp=np.asarray(io.get("cp")[1]).tolist(), jaw=jaw(), eff=eff(), t=time.time()), open(d + "/state.json", "w"), indent=1)
    print(f"view {i}: wrist deg {np.round(np.degrees(jsq()[3:]), 1).tolist()} ins {jsq()[2] * 1e3:.1f} jaw {jaw():.3f} eff {eff():.3f}", flush=True)
try:
    S = mv.setpoint(); T = S.copy(); T[:3, 3] = S[:3, 3] + mv.pad[0] * a.lift * 1e-3; qn = jsq()
    print("now wrist deg", np.round(np.degrees(qn[3:]), 1).tolist(), "margins", np.round(K.margins(qn), 0).tolist(), "jaw %.3f eff %.3f" % (jaw(), eff()), "tip above pad %.1f" % mv.pad_dist_mm(mv.tip(S)), flush=True)
    print("lift %.0f mm: cp mm %s -> %s" % (0 if a.no_lift else a.lift, np.round(S[:3, 3] * 1e3, 1).tolist(), np.round(T[:3, 3] * 1e3, 1).tolist()), flush=True)
    for f in np.linspace(0, 1, 21)[1:]:
        Tf = S.copy(); Tf[:3, 3] = S[:3, 3] + f * (T[:3, 3] - S[:3, 3]); mv.check_target(Tf, contact_ok="away")
    ok, q1, why = K.check_pose(T[:3, 3], T[:3, :3], qn, np.array(cal["frozen"]["suj_psm1"]))
    if not ok: raise Guard("lift target: " + why)
    tg = []
    for w in WRIST:
        q = q1 + np.r_[0, 0, 0, np.radians(w)]; mg = K.margins(q); print("  wrist offset", w, "-> wrist deg", np.round(np.degrees(q[3:]), 0).tolist(), "margins wrist", np.round(mg[3:], 0).tolist(), flush=True)
        if mg[3:].min() < MIN_MARGIN: raise Guard(f"margin at {w}")
        tg.append(q)
    print("dry-run checks passed", flush=True)
    if a.send:
        check_eff("start")
        if not a.no_lift: mv.move_to(T, tag="lift", contact_ok="away", lim=(2.0, 3.0)); check_eff("lift"); q1 = jsq(); tg = [q1 + np.r_[0, 0, 0, np.radians(w)] for w in WRIST]
        for i, q in enumerate(tg): goto_q(q, f"v{i}"); snap(i)
        print("done, commands", mv.n_cmd, flush=True)
except BaseException as e:
    import traceback; traceback.print_exc(); print("STOPPED:", e, flush=True); log(dict(ev="stop", why=str(e)))
sys.stdout.flush(); os._exit(0)
