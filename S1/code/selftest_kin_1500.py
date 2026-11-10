#!/usr/bin/env python3
"""FakeIO-only test of the 14:5x hover gate (no robot): same setup as session_driver.selftest up to the first full attempt, then a full
attempt whose hover checks after the move report a camera grasp point 10 mm off -> expect source 'kin' in every check and status completed."""
import json, os, shutil, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_driver as SD
import calib as C, attempt as A

kin = SD.load_kin(); tmp = tempfile.mkdtemp(prefix="s1028_kintest_"); os.makedirs(tmp + "/C_calib")
shutil.copytree(SD.ROOT + "/C_calib/scene0", tmp + "/C_calib/scene0")
fw = SD.FakeWorld(kin); orig = C.save_snap
def save_snap_fake(io, d, k=6, extra=None):
    r = orig(io, d, k, extra); np.save(d + "/fake_disp.npy", fw.disp_map()); return r
C.save_snap = save_snap_fake; A.save_snap = save_snap_fake
SD.fake_r6_cal(fw, tmp); os.makedirs(tmp + "/D_pad", exist_ok=True)
json.dump(dict(normal=fw.n_ecm.tolist(), d=fw.d, max_resid_mm=0.0, fake=True), open(tmp + "/D_pad/pad.json", "w"))
SD.stage_prereg(tmp)
real_track = A.Attempt.track10
def off_track(self, name, pred):
    tr = real_track(self, name, pred)
    if name != "hover_init" and "X_grasp" in tr: tr["X_grasp"] = np.asarray(tr["X_grasp"]) + np.array([0.0, 0.0, 10.0])
    return tr
A.Attempt.track10 = off_track
r = SD.stage_attempt(tmp, 8, fake=fw, gui=False)
hv = [json.loads(l) for l in open(r["dir"] + "/trial_log.jsonl") if '"name": "hover"' in l and '"ev": "phase"' in l][-1]
print("status", r["status"], "| last phase", r["last_phase"], "| abort", r.get("abort_reason"))
print("hover init", hv.get("init"), "| checks", [(c["it"], c["source"], None if c["cam_vs_kin_mm"] is None else round(c["cam_vs_kin_mm"], 1), round(c["err_norm"], 2)) for c in hv["corrections"]], "| stop", hv["stop_reason"])
assert r["status"] == "completed" and all(c["source"] == "kin" for c in hv["corrections"]), "kin-path test failed"
print("KIN-PATH TEST PASSED")
