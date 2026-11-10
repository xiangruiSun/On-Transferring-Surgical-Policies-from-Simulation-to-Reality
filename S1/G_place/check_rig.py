#!/usr/bin/env python3
"""READ-ONLY rig check before any motion (sends nothing): PSM1 operating state, fresh joint / pose / camera data,
measured_cp vs SUJ x FK (10 reads), SUJ and ECM vs the values frozen at calibration (C_r6cal), arm at the base?

    python3 check_rig.py
"""
import json, os, sys, time
import numpy as np
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code")
from rig import RigIO, Mover, load_kin
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json"))
io = RigIO(images=True); kin = load_kin(); time.sleep(3.0); mv = Mover(io, kin, lambda r: None)
mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = np.array(cal["frozen"]["suj_psm1"])
bad = []
op = io.get("opstate"); print("PSM1 operating state:", None if op is None else op[1]);  bad += [] if op and op[1] == "ENABLED" else ["PSM1 not ENABLED (Power On + Home in the dVRK GUI)"]
for k in ("cp", "sp", "js", "suj", "ecm", "jaw"):
    v = io.get(k); age = None if v is None else time.monotonic() - v[0]; print(f"  {k:4s} age {age if age is None else round(age, 2)} s")
    if v is None or age > 1.0: bad.append(f"{k} missing / stale")
js = io.get("js"); print("  PSM1 joints", None if js is None else np.round(js[1][:6], 4).tolist())
ok = 0
for _ in range(10):
    try: mv.state_ok(); ok += 1
    except Exception as e: print("  state check:", e)
    time.sleep(0.2)
print(f"state checks passed {ok}/10 (measured_cp vs SUJ x FK, ECM / SUJ vs frozen, operating state)")
if ok < 10: bad.append("state checks failed (see above)")
suj = io.get("suj"); ecm = io.get("ecm")
if suj is not None: print("  SUJ PSM1 vs frozen: %.3f mm" % (np.linalg.norm(np.asarray(suj[1])[:3, 3] - mv.suj0[:3, 3]) * 1e3))
if ecm is not None: print("  ECM joints", np.round(ecm[1], 4).tolist(), "frozen", np.round(mv.ecm0, 4).tolist())
try:
    d = np.linalg.norm(mv.setpoint()[:3, 3] - np.array(cal["base_T"])[:3, 3]) * 1e3; print("  arm to calibration base: %.2f mm" % d)
    if d > 1.5: bad.append("arm is not at the base -> code/recover_to_base.py (set out=) after looking at the robot")
except Exception as e: bad.append(f"setpoint: {e}")
try: b = io.frames(2); print("  stereo frames ok", b[-1][1].shape)
except Exception as e: bad.append(f"camera: {e}")
print("\nRESULT:", "OK - ready" if not bad else "NOT READY:\n  - " + "\n  - ".join(bad))
sys.stdout.flush(); os._exit(0 if not bad else 1)
