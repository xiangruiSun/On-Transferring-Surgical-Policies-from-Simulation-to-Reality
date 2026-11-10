#!/usr/bin/env python3
"""Undo run2's blind climb (23 x 1.5 mm along the site normal while tipmeas was locked on the sticker edge): pure translation
(orientation unchanged) to the run2 start + --keep-mm along the same direction, then a needle measurement with the ROI on the
PREDICTED tip.  --send to move."""
import argparse, json, os, sys, time
import numpy as np
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path.insert(0, R0 + "/code"); sys.path.insert(0, R0 + "/G_place")
import cv2
from rig import RigIO, Mover, load_kin, TCP_MM
from calib import proj, cam2ecm_mm
import tipmeas
ap = argparse.ArgumentParser(); ap.add_argument("--keep-mm", type=float, default=10.0); ap.add_argument("--send", action="store_true"); a = ap.parse_args()
G = R0 + "/G_place"; out = G + "/v5_recover_" + time.strftime("%H%M%S"); os.makedirs(out, exist_ok=True); logf = open(out + "/log.jsonl", "a")
def log(r): logf.write(json.dumps(dict(t=time.time(), **r), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"); logf.flush()
S = [json.loads(l) for l in open(G + "/v5_run2/v5_log.jsonl")]; S = [r for r in S if r["ev"] == "step"]
p1, p2 = np.array(S[0]["target"])[:3, 3], np.array(S[1]["target"])[:3, 3]; d_e = (p2 - p1) / np.linalg.norm(p2 - p1); p_start = p1 - 1.5e-3 * d_e
plan = json.load(open(G + "/plan_manual1.json")); cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); pad = json.load(open(R0 + "/D_pad/pad.json"))
c, n = np.array(plan["site_cam"]["centre"]), np.array(plan["site_cam"]["normal"])
io = RigIO(images=True); kin = load_kin(); time.sleep(2.5); mv = Mover(io, kin, log)
mv.ecm0 = np.array(cal["frozen"]["ecm_js"]); mv.suj0 = np.array(cal["frozen"]["suj_psm1"]); mv.pad = (np.array(pad["normal"]), pad["d"]); mv.cube = None
mv.tip_off_tool = np.array(plan["tip_tool_mm"]) * 1e-3 - TCP_MM * 1e-3
sp = mv.setpoint(); tgt = p_start + a.keep_mm * 1e-3 * d_e; v = tgt - sp[:3, 3]
print(f"now {np.linalg.norm(sp[:3, 3] - p_start) * 1e3:.1f} mm from the run2 start; move {np.round(v * 1e3, 1).tolist()} mm ({np.linalg.norm(v) * 1e3:.1f} mm), "
      f"orientation change from run2 start {np.degrees(np.arccos(np.clip((np.trace(np.array(S[0]['target'])[:3, :3].T @ sp[:3, :3]) - 1) / 2, -1, 1))):.2f} deg", flush=True)
tip_pred_e = mv.tip(sp) + v; tip_pred_c = np.r_[tip_pred_e * 1e3, 1] @ M
print("predicted tip after the move (cam mm)", np.round(tip_pred_c, 1).tolist(), "uv", np.round(proj(tip_pred_c), 0).tolist(), "pad clearance of the tip %.1f mm" % mv.pad_dist_mm(tip_pred_e), flush=True)
if a.send:
    mv.translate(v, tag="recover_back", lim=(2.0, 3.0)); time.sleep(0.5)
b = io.frames(10); R = np.mean([x[1] for x in b], 0).round().astype(np.uint8); L = np.mean([x[0] for x in b], 0).round().astype(np.uint8)
cv2.imwrite(out + "/right_rect.png", R); cv2.imwrite(out + "/left_rect.png", L)
uv = proj(tip_pred_c if a.send else (np.r_[mv.tip(sp) * 1e3, 1] @ M))
m, why = tipmeas.measure(R, L, c[2], roi_uv=uv, roi_px=160, jaw_px=None)
if m is None: print("measure:", why)
else:
    e = m["tip_cam"] - c; h = float(e @ n); lat = float(np.linalg.norm(e - h * n)); ang = float(np.degrees(np.arccos(np.clip(m["dir_cam"] @ -n, -1, 1))))
    print(f"measured tip {np.round(m['tip_cam'], 1).tolist()} (pred {np.round(tip_pred_c, 1).tolist()}, diff {np.linalg.norm(m['tip_cam'] - tip_pred_c):.2f} mm); height {h:.2f} lateral {lat:.2f} dir {ang:.1f} deg")
    ov = R.copy(); cv2.polylines(ov, [np.round(m["poly"]).astype(np.int32)], False, (0, 255, 255), 2); cv2.circle(ov, tuple(np.round(m["tip"]["uv"]).astype(int)), 8, (255, 0, 255), 2)
    cv2.circle(ov, tuple(np.round(uv).astype(int)), 160, (255, 255, 0), 1); cv2.imwrite(out + "/overlay.png", ov)
print("jaw", np.round(io.get("jaw")[1], 3), "eff", np.round(io.get("jaw_eff")[1], 3), "out", out, flush=True); sys.stdout.flush(); os._exit(0)
