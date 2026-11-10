#!/usr/bin/env python3
"""READ-ONLY. Learn the demonstrated placing pose as: tool pose relative to the site frame (entry sticker origin; u entry->exit, m sideways, n up), plus the needle pixels for image checks."""
import json, os, sys
import numpy as np, cv2
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", R0 + "/P_place", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K, needle3d as N3
from calib import rotation_from_affine
from scipy.spatial.transform import Rotation as Ro
name, mj, tipR, tipL = sys.argv[1], sys.argv[2], list(map(float, sys.argv[3:5])), list(map(float, sys.argv[5:7]))
d = f"{R0}/G_place/{name}"; dem = json.load(open(d + "/demo.json")); cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); Rc = rotation_from_affine(M); s = N3.site(json.load(open(mj))["stickers"])
q = np.array(dem["js_mean"]); T = K.fk_ecm(q, np.array(dem["suj"])); hinge = np.r_[T[:3, 3] * 1e3, 1] @ M; Rct = Rc @ T[:3, :3]; B = np.c_[s["u"], s["m"], s["n"]]
L = dict(name=name, q_demo=q.tolist(), jaw=dem["jaw_mean"], jaw_eff=dem["jaw_eff"], suj=dem["suj"], ecm=dem["ecm"], hinge_site=(B.T @ (hinge - s["e"])).tolist(), R_site_tool=(B.T @ Rct).tolist(), tipR=tipR, tipL=tipL, tip_cam_rough=N3.backproj(tipR[0], tipR[1], tipR[0] - tipL[0]).tolist(), site=dict(e=s["e"].tolist(), x=s["x"].tolist(), n=s["n"].tolist(), u=s["u"].tolist(), bite=s["bite"]), M_source="C_r6cal_try2")
json.dump(L, open(R0 + "/P_place/learned.json", "w"), indent=1)
print("learned: hinge in site mm", np.round(L["hinge_site"], 1).tolist(), "wrist deg", np.round(np.degrees(q[3:]), 1).tolist(), "yaw margin %.1f" % K.margins(q)[5], "ins mm %.1f" % (q[2] * 1e3))
t = np.array(tipR); print("tip right px", tipR, "left px", tipL, "disp %.1f" % (tipR[0] - tipL[0]), "rough tip cam", np.round(L["tip_cam_rough"], 1).tolist(), "rel entry in site", np.round(B.T @ (np.array(L["tip_cam_rough"]) - s["e"]), 1).tolist())
for other, jf in (("my last pose (p2d)", "p2d/s1/state.json"),):
    o = json.load(open(jf)); To = K.fk_ecm(np.array(o["q"]), np.array(dem["suj"])); dR = T[:3, :3].T @ To[:3, :3]; rv = Ro.from_matrix(Rc @ T[:3, :3] @ dR @ (Rc @ T[:3, :3]).T).as_rotvec(); hc = np.r_[To[:3, 3] * 1e3, 1] @ M
    print(other, "vs demo: tool rotation %.0f deg, axis in site (u m n) %s; hinge offset site mm %s; wrist deg %s; jaw %.3f eff %.3f" % (np.degrees(np.linalg.norm(rv)), np.round(B.T @ rv / np.linalg.norm(rv), 2).tolist(), np.round(B.T @ (hc - hinge), 1).tolist(), np.round(np.degrees(o["q"][3:]), 1).tolist(), o["jaw"], o["eff"]))
