#!/usr/bin/env python3
"""DRY: fixed needle-in-tool transform (one stereo measurement + FK + C_r6cal M) -> tool pose for a target needle pose -> IK / margins / clearance.
Sends nothing."""
import json, os, sys
import numpy as np
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K
from calib import rotation_from_affine, cam2ecm_mm
from rig import FX, CX, CY, FB, TCP_MM
meas = json.load(open(sys.argv[1])); dem = json.load(open(sys.argv[2])); snap = sys.argv[3]
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); Rc = rotation_from_affine(M)
SUJ = np.array(dem["suj"]); q0 = np.array(dem["js_mean"]); T0 = K.fk_ecm(q0, SUJ)
print("FK vs cp at the snapshot: %.3f mm" % (np.linalg.norm(T0[:3, 3] - np.array(dem["cp"])[:3, 3]) * 1e3), "| SUJ vs frozen %.3f mm" % (np.linalg.norm(SUJ[:3, 3] - np.array(cal["frozen"]["suj_psm1"])[:3, 3]) * 1e3))
m = meas["m3"]; C, tip, tan = map(np.array, (m["centre"], m["tip_cam"], m["dir_cam"])); Rn = m["radius_mm"]; span = m["span_deg"]
x = (tip - C) / np.linalg.norm(tip - C); z = np.cross(x, tan); z /= np.linalg.norm(z); y = np.cross(z, x)
RcN = np.c_[x, y, z]
Rtn = T0[:3, :3].T @ Rc.T @ RcN; ptn = T0[:3, :3].T @ (cam2ecm_mm(C, M) * 1e-3 - T0[:3, 3])
tail = C + Rn * (np.cos(-np.radians(span)) * x + np.sin(-np.radians(span)) * y)
tail_tool = T0[:3, :3].T @ (cam2ecm_mm(tail, M) * 1e-3 - T0[:3, 3])
print("needle in tool: centre mm", np.round(ptn * 1e3, 1).tolist(), "normal", np.round(Rtn[:, 2], 3).tolist(), "tail mm", np.round(tail_tool * 1e3, 1).tolist(), "(TCP", TCP_MM.tolist(), ") span", round(span, 1))
print("tail vs TCP %.1f mm" % np.linalg.norm(tail_tool * 1e3 - TCP_MM))
S = meas["stickers"]; ent, ex = (S[1], S[0]) if S[1]["area"] < S[0]["area"] else (S[0], S[1])
e, xx = np.array(ent["cam"]), np.array(ex["cam"]); n = np.array(ent["n25"]); n = n if n[2] < 0 else -n
u = (xx - e) - ((xx - e) @ n) * n; bite = np.linalg.norm(u); u /= bite; side = np.cross(n, u)
print("entry", np.round(e, 1).tolist(), "exit", np.round(xx, 1).tolist(), "bite %.1f mm" % bite, "n", np.round(n, 3).tolist(), "u", np.round(u, 3).tolist())
disp = np.load(snap + "/fs_disp.npy")
def clearance(T):
    """lowest height of hinge -> jaw tip and of the shaft (first 40 mm) above the FS surface, along the camera ray depth."""
    out = []
    for name, s0, s1 in (("jaw", 0.0, 1.0), ("shaft", -3.0, 0.0)):
        best = 1e9
        for f in np.linspace(s0, s1, 25):
            p = T[:3, 3] + T[:3, :3] @ (TCP_MM * 1e-3 * f); X = np.r_[p * 1e3, 1] @ M; uu, vv = FX * X[0] / X[2] + CX, FX * X[1] / X[2] + CY
            if 0 <= uu < disp.shape[1] and 0 <= vv < disp.shape[0]:
                best = min(best, FB / disp[int(vv), int(uu)] - X[2])
        out.append(round(float(best), 1))
    return out
def tool_pose(Rstar, Cstar):
    R = Rc.T @ Rstar @ Rtn.T; p = cam2ecm_mm(Cstar, M) * 1e-3 - R @ ptn; return R, p
def solve(R, p):
    best = None
    lo, hi = K.QMIN + 1e-4, K.QMAX - 1e-4
    for dr in (0, 90, -90, 180, -180):
        for dp, dy in ((0, 0), (40, 40), (40, -40), (-40, 40), (-40, -40)):
            qi = np.clip(q0 + np.r_[0, 0, 0, np.radians([dr, dp, dy])], lo, hi)
            q, ep, er = K.ik_ecm(p, R, qi, SUJ)
            if ep > 0.05 or er > 0.05: continue
            mg = K.margins(q); score = np.min(np.r_[mg[:2], mg[2] / 2, mg[3:]])
            if best is None or score > best[0]: best = (score, q, mg)
    return best
def target(kind, a, dh, h):
    du = np.cos(np.radians(dh)) * u + np.sin(np.radians(dh)) * side; ta = np.radians(a)
    if kind == "def":   # tip tangent = -n tilted a toward u; concave toward the exit
        t = -np.cos(ta) * n + np.sin(ta) * du; w = np.sin(ta) * n + np.cos(ta) * du
    else:               # demo1 style: tip heading along u, tilted a up; centre above the tissue, jaw trailing away from the exit
        t = np.cos(ta) * du + np.sin(ta) * n; w = -np.sin(ta) * du + np.cos(ta) * n
    tipc = e + h * n; Cs = tipc + Rn * w; xs = -w; zs = np.cross(xs, t); zs /= np.linalg.norm(zs); return np.c_[xs, np.cross(zs, xs), zs], Cs
print("kind  a  dh  h | score | margins yaw pit ins roll wp wy | q deg | clear jaw shaft")
rows = []
for kind in ("def", "demo"):
    for a in (0, 15, 30, 45, 60, 75, 90):
        for dh in (0, -20, 20, -40, 40):
            Rs, Cs = target(kind, a, dh, 3.0); R, p = tool_pose(Rs, Cs); b = solve(R, p)
            if b is None: rows.append((kind, a, dh, None)); continue
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p; rows.append((kind, a, dh, b[0], np.round(b[2], 0).tolist(), np.round(np.degrees(b[1][[3, 4, 5]]), 0).tolist(), round(b[1][2] * 1e3), clearance(T)))
for r in rows:
    if r[3] is None: print(r[0], r[1], r[2], "IK none")
    else: print(r[0], r[1], r[2], "score %.1f" % r[3], r[4], "roll/p/y", r[5], "ins", r[6], "clear", r[7])
