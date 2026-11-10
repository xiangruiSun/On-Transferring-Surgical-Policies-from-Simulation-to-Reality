#!/usr/bin/env python3
"""Read-only hardware-guard watchdog for an UNMODIFIED script that commands PSM1 (today: the R6 r6_calib.py copy).  Publishes nothing.

    python3 watchdog.py --pid <pid of the commanding script> --log wd.jsonl [--cube-mm 60]

Listens to /PSM1/move_cp (every command the script sends) and the robot state.  Trips (SIGINT to --pid, TRIP file, exit 3) on:
  PSM1 operating state != ENABLED; commanded pose IK-unreachable or joint margin < 3 deg / 5 mm; goal_reached not true within 10 s
  of a command; setpoint != command after reached (> 0.3 mm / 0.3 deg: silent IK clip); measured not within 1.0 mm / 1.5 deg of the
  command; joint tracking at rest > 0.5 mm insertion / 1 deg; measured_cp vs SUJ x FK > 0.5 mm; ECM / SUJ base moved; command tip
  outside +-cube_mm around the tip at start.  Exits by itself when --pid ends.
"""
import argparse, json, os, signal, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig import RigIO, load_kin, tip_of, geodesic_deg, T_of
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser(); ap.add_argument("--pid", type=int, required=True); ap.add_argument("--log", required=True); ap.add_argument("--cube-mm", type=float, default=60.0)
a = ap.parse_args()
io = RigIO(node_name="s1028_watchdog", images=False); kin = load_kin(); from geometry_msgs.msg import PoseStamped
cmds = []
def on_cmd(m):
    p, o = m.pose.position, m.pose.orientation
    cmds.append((time.monotonic(), T_of([p.x, p.y, p.z], Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix())))
io.node.create_subscription(PoseStamped, "/PSM1/move_cp", on_cmd, 50)
f = open(a.log, "a")
def log(**k): k["t"] = time.time(); f.write(json.dumps(k, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"); f.flush()
def trip(why):
    log(ev="TRIP", why=why); open(os.path.dirname(os.path.abspath(a.log)) + "/WATCHDOG_TRIP", "w").write(f"{time.strftime('%T')} {why}\n")
    try: os.kill(a.pid, signal.SIGINT)
    except ProcessLookupError: pass
    for _ in range(20):                    # 2026-11-05: background jobs inherit SIGINT ignored -> fall back to SIGTERM after 1 s
        time.sleep(0.05)
        if not alive(): break
    else:
        try: os.kill(a.pid, signal.SIGTERM); log(ev="TRIP_SIGTERM_FALLBACK")
        except ProcessLookupError: pass
    print("WATCHDOG TRIP:", why, flush=True); sys.stdout.flush(); os._exit(3)
def alive():
    try: os.kill(a.pid, 0); return True
    except ProcessLookupError: return False
t0 = time.time()
while time.time() - t0 < 8 and any(io.get(k) is None for k in ("cp", "sp", "suj", "js", "spj", "ecm")): time.sleep(0.1)
if any(io.get(k) is None for k in ("cp", "sp", "suj", "js", "spj", "ecm")): trip("watchdog: robot topics missing")
ecm0 = io.get("ecm")[1].copy(); suj0 = io.get("suj")[1].copy(); c0 = tip_of(io.get("cp")[1]); log(ev="start", ecm0=ecm0, suj0=suj0, tip0=c0)
print("watchdog armed", flush=True); open(os.path.dirname(os.path.abspath(a.log)) + "/WATCHDOG_ARMED", "w").write(time.strftime("%T")); seen = 0; pending = None; last_rest = time.monotonic()
while alive():
    time.sleep(0.05)
    op = io.get("opstate")
    if op is not None and op[1] != "ENABLED": trip(f"operating state {op[1]}")
    e = io.get("ecm")[1]; d = np.abs(e - ecm0)
    if np.degrees(np.r_[d[:2], d[3]]).max() > 0.1 or d[2] * 1e3 > 0.1: trip("ECM moved")
    s = io.get("suj")[1]
    if np.linalg.norm(s[:3, 3] - suj0[:3, 3]) * 1e3 > 0.3 or geodesic_deg(s[:3, :3], suj0[:3, :3]) > 0.1: trip("PSM1 SUJ base moved")
    js = io.get("js")[1]; cp = io.get("cp")[1]
    dfk = np.linalg.norm(kin.fk_ecm(js[:6], s)[:3, 3] - cp[:3, 3]) * 1e3
    if dfk > 0.5: trip(f"measured_cp vs SUJ x FK {dfk:.2f} mm")
    if pending is not None:
        tc, T, i = pending; r = io.get("reached")
        if r is not None and r[0] > tc and r[1] and time.monotonic() - r[0] > 0.25:      # r6_calib sends the next command 0.4 s after reached
            sp = io.get("sp")[1]; e_sp = np.linalg.norm(sp[:3, 3] - T[:3, 3]) * 1e3; r_sp = geodesic_deg(sp[:3, :3], T[:3, :3])
            e_m = np.linalg.norm(cp[:3, 3] - T[:3, 3]) * 1e3; r_m = geodesic_deg(cp[:3, :3], T[:3, :3]); jerr = js[:6] - io.get("spj")[1][:6]
            log(ev="arrived", i=i, sp_err_mm=e_sp, sp_err_deg=r_sp, meas_err_mm=e_m, meas_err_deg=r_m, joint_err=jerr)
            if e_sp > 0.3 or r_sp > 0.3: trip(f"command {i}: setpoint not at the command {e_sp:.2f} mm / {r_sp:.2f} deg (IK clip)")
            if e_m > 2.0 or r_m > 2.0: trip(f"command {i}: measured {e_m:.2f} mm / {r_m:.2f} deg from the command")
            if abs(jerr[2]) * 1e3 > 0.5 or np.degrees(np.abs(np.r_[jerr[:2], jerr[3:6]])).max() > 1.0: trip(f"command {i}: joint tracking error {np.round(jerr, 5).tolist()}")
            pending = None
        elif time.monotonic() - tc > 10.0 and len(cmds) == i: trip(f"command {i}: goal_reached not true within 10 s")
    while seen < len(cmds):
        tc, T = cmds[seen]; seen += 1
        ok, q, why = kin.check_pose(T[:3, 3], T[:3, :3], js[:6], s, min_deg=3.0, min_mm=5.0)
        tipc = tip_of(T); dc = np.abs(tipc - c0) * 1e3
        log(ev="cmd", i=seen, T=T, ik_ok=ok, why=why, tip_from_start_mm=dc)
        if not ok: trip(f"command {seen}: IK / margin: {why}")
        if np.any(dc > a.cube_mm): trip(f"command {seen}: tip {np.round(dc, 1).tolist()} mm from start (> {a.cube_mm})")
        pending = (tc, T, seen)
log(ev="end", n_cmds=len(cmds)); print("watchdog: commanding process ended,", len(cmds), "commands checked", flush=True); sys.stdout.flush(); os._exit(0)
