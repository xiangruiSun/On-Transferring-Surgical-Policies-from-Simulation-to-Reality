#!/usr/bin/env python3
"""Reproduce the learned placing pose from wherever the user left PSM1.
 target tool pose = learned tool-in-site applied to the CURRENT site frame (needle fixed in the jaw).
 Phase 1: lift if needed, joint move to the target orientation with the tool hover-mm higher along n; snapshot for checking.
 Phase 2: straight down to the target in small steps, needle tip pixel check afterwards.
Dry run by default.  --phase 1 | 2.  Guards: ECM/SUJ freeze, joint settling, wrist margins, tool clearance from the FS surface, jaw effort (--min-eff), insertion effort in phase 2."""
import argparse, json, os, sys, time
import numpy as np, cv2
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", R0 + "/P_place", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K, needle3d as N3
from rig import RigIO, Mover, load_kin, Guard, FX, CX, CY, FB, TCP_MM
from calib import rotation_from_affine, cam2ecm_mm
from sensor_msgs.msg import JointState
ap = argparse.ArgumentParser(); ap.add_argument("--learned", default="learned.json"); ap.add_argument("--measure", default="m0/measure.json"); ap.add_argument("--fssnap", default="../G_place/hover_142828/snap"); ap.add_argument("--phase", type=int, default=1); ap.add_argument("--hover", type=float, default=15.0); ap.add_argument("--lift", type=float, default=0.0)
ap.add_argument("--grip", default=None, help="grip_fit.py output: new needle grip"); ap.add_argument("--extra-h", type=float, default=0.0, help="stop this many mm above the learned pose (phase 2)"); ap.add_argument("--min-clear", type=float, default=3.0); ap.add_argument("--min-eff", type=float, default=0.12); ap.add_argument("--ins-eff", type=float, default=0.25); ap.add_argument("--out", required=True); ap.add_argument("--send", action="store_true"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); L = json.load(open(a.learned)); cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); pad = json.load(open(R0 + "/D_pad/pad.json")); M = np.array(cal["M"]); Rc = rotation_from_affine(M); SUJ = np.array(cal["frozen"]["suj_psm1"]); disp = np.load(a.fssnap + "/fs_disp.npy")
s = N3.site(json.load(open(a.measure))["stickers"]); B = np.c_[s["u"], s["m"], s["n"]]; Rct = B @ np.array(L["R_site_tool"]); hinge = s["e"] + B @ np.array(L["hinge_site"]); Rt = Rc.T @ Rct; pt = cam2ecm_mm(hinge, M) * 1e-3; nE = Rc.T @ s["n"]
if a.grip:
    g = json.load(open(a.grip)); Rct = Rct @ np.array(g["R_delta"]); Rt = Rc.T @ Rct; so = json.load(open(a.learned))["site"]; So = np.c_[so["u"], np.cross(so["n"], so["u"]), so["n"]]; tip_site = So.T @ (np.array(L["tip_cam_rough"]) - np.array(so["e"])); tip_t = s["e"] + B @ tip_site; pt = cam2ecm_mm(tip_t, M) * 1e-3 - Rt @ (np.array(g["tip_tool_mm"]) * 1e-3); print("grip correction %.0f deg; tip target in site %s; tip target cam %s" % (g["angle"], np.round(tip_site, 1).tolist(), np.round(tip_t, 1).tolist()), flush=True); np.save(a.out + "/tip_target_cam.npy", tip_t)
logf = open(a.out + "/log.jsonl", "a"); log = lambda r: (logf.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"), logf.flush())
io = RigIO(images=True); kin = load_kin(); time.sleep(2.5); mv = Mover(io, kin, log); mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = SUJ; mv.pad = (np.array(pad["normal"]), pad["d"]); mv.state_ok()
pub = io.node.create_publisher(JointState, "/PSM1/move_jp", 1); time.sleep(0.5)
jsq = lambda: np.array(io.get("js")[1][:6]); eff = lambda: float(io.get("jaw_eff")[1][0]); jaw = lambda: float(io.get("jaw")[1][0]); ins = lambda: float(io.get("jse")[1][2]) if io.get("jse") else float("nan")
def solve(p, seed):
    now = jsq(); best = None
    for dr in (0, 60, -60, 120, -120, 180, -180, 240, -240):
        for sd in (seed, now):
            si = np.clip(sd + np.r_[0, 0, 0, np.radians(dr), 0, 0], K.QMIN + 1e-3, K.QMAX - 1e-3); q, ep, er = K.ik_ecm(p, Rt, si, SUJ); m = K.margins(q)
            if ep > 0.05 or er > 0.05 or m[3:].min() < 3 or m[:2].min() < 3 or m[2] < 5: continue
            cost = abs(q[3] - now[3])                                          # least roll travel: thread and needle sweep less
            if best is None or cost < best[0]: best = (cost, q)
    if best is None: raise Guard("no IK solution with margins")
    print("IK: roll travel %.0f deg" % np.degrees(best[0]), flush=True); return best[1]
def clear(q):
    T = K.fk_ecm(q, SUJ); best = 1e3
    for f in np.linspace(0, 1.15, 24):
        X = np.r_[(T[:3, 3] + T[:3, :3] @ (TCP_MM * 1e-3 * f)) * 1e3, 1] @ M; u, v = FX * X[0] / X[2] + CX, FX * X[1] / X[2] + CY
        if 0 <= u < disp.shape[1] and 0 <= v < disp.shape[0]: best = min(best, FB / disp[int(v), int(u)] - X[2])
    return float(best)
def path(q0, q1, n): return [q0 + (q1 - q0) * k / n for k in range(1, n + 1)]
def go(qs, tag, e0=None):
    for k, qk in enumerate(qs, 1):
        mv.state_ok(); m = JointState(); m.position = qk.tolist(); pub.publish(m); t0 = time.time()
        while time.time() - t0 < 6.0:
            time.sleep(0.05)
            if np.degrees(np.abs(jsq() - qk)[[0, 1, 3, 4, 5]]).max() < 1.0 and abs(jsq()[2] - qk[2]) < 5e-4: break
        else: raise Guard(f"joints did not settle at {tag} {k}/{len(qs)}")
        time.sleep(0.1); log(dict(ev="step", tag=tag, k=k, q=jsq(), eff=eff(), ins=ins()))
        if abs(eff()) < a.min_eff: raise Guard(f"jaw effort {eff():.3f}: needle may be loose")
        if e0 is not None:
            if k == 2: e0 = ins()
            if k > 2 and abs(ins() - e0) > a.ins_eff: raise Guard(f"insertion effort change {ins() - e0:.3f} at {k}/{len(qs)}")
    time.sleep(0.6)
def snap(tag):
    b = io.frames(8); Li = np.mean([z[0] for z in b], 0).round().astype(np.uint8); Ri = np.mean([z[1] for z in b], 0).round().astype(np.uint8); d = f"{a.out}/{tag}"; os.makedirs(d, exist_ok=True); cv2.imwrite(d + "/left_rect.png", Li); cv2.imwrite(d + "/right_rect.png", Ri); json.dump(dict(q=jsq().tolist(), cp=np.asarray(io.get("cp")[1]).tolist(), jaw=jaw(), eff=eff()), open(d + "/state.json", "w"), indent=1); print(f"{tag}: wrist deg {np.round(np.degrees(jsq()[3:]), 1).tolist()} ins {jsq()[2] * 1e3:.1f} jaw {jaw():.3f} eff {eff():.3f}", flush=True)
try:
    q0 = jsq(); T0 = K.fk_ecm(q0, SUJ); qd = np.array(L["q_demo"]); print("now wrist deg", np.round(np.degrees(q0[3:]), 1).tolist(), "ins %.1f; jaw %.3f eff %.3f; tool clearance %.1f" % (q0[2] * 1e3, jaw(), eff(), clear(q0)), flush=True)
    if a.phase == 1:
        ph = pt + nE * a.hover * 1e-3; qh = solve(ph, qd); print("hover %.0f mm: wrist deg %s ins %.1f margins %s; hinge move %.1f mm" % (a.hover, np.round(np.degrees(qh[3:]), 1).tolist(), qh[2] * 1e3, np.round(K.margins(qh), 1).tolist(), np.linalg.norm(ph - T0[:3, 3]) * 1e3), flush=True)
        qL = q0
        if a.lift > 0:
            pL = T0[:3, 3] + nE * a.lift * 1e-3; qL = solve_l = K.ik_ecm(pL, T0[:3, :3], q0, SUJ)[0]; print("lift %.0f first, clearance after %.1f" % (a.lift, clear(qL)), flush=True)
        n = max(2, int(np.ceil(np.degrees(np.abs(qh - qL)[[0, 1, 3, 4, 5]]).max() / 4.0))); P = path(qL, qh, n); c = min(clear(q) for q in P); print(f"joint path {n} steps, min tool clearance {c:.1f}; end {clear(qh):.1f}", flush=True)
        if c < a.min_clear: raise Guard(f"path clearance {c:.1f} < {a.min_clear}: use --lift")
        print("dry-run checks passed", flush=True)
        if a.send:
            if a.lift > 0: go(path(q0, qL, max(2, int(a.lift))), "lift")
            go(P, "to_hover"); snap("hover"); print("phase 1 done", flush=True)
    else:
        pg = pt + nE * a.extra_h * 1e-3; qg = solve(pg, q0); dist = np.linalg.norm(pg - T0[:3, 3]) * 1e3; print("to target: hinge move %.1f mm; wrist deg %s ins %.1f margins %s; clearance end %.1f" % (dist, np.round(np.degrees(qg[3:]), 1).tolist(), qg[2] * 1e3, np.round(K.margins(qg), 1).tolist(), clear(qg)), flush=True)
        n = max(4, int(np.ceil(dist / 0.5))); print("dry-run checks passed", flush=True)
        if a.send: go(path(q0, qg, n), "down", e0=ins()); snap("target"); print("phase 2 done", flush=True)
except BaseException as e:
    import traceback; traceback.print_exc(); print("STOPPED:", e, flush=True); log(dict(ev="stop", why=str(e)))
sys.stdout.flush(); os._exit(0)
