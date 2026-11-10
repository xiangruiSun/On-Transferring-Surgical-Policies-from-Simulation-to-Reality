#!/usr/bin/env python3
"""READ-ONLY. Per-view stereo circle -> per-view needle-in-tool transform; spread over views = how rigid / how reliable FK is."""
import json, os, sys
import numpy as np, cv2
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", R0 + "/P_place", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K, arc_fit as AF
from calib import rotation_from_affine, cam2ecm_mm
from rig import FX, CX, CY, FB, TCP_MM
from scipy.spatial.transform import Rotation as Ro
d, hints = sys.argv[1], json.load(open(sys.argv[2])); z = float(sys.argv[3]) if len(sys.argv) > 3 else 97.0
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); Rc = rotation_from_affine(M); SUJ = np.array(cal["frozen"]["suj_psm1"]); out = {}
for k, (tip, jaw) in hints.items():
    v = f"{d}/{k}"; R = cv2.imread(v + "/right_rect.png"); L = cv2.imread(v + "/left_rect.png"); st = json.load(open(v + "/state.json")); tip, jaw = np.array(tip, float), np.array(jaw, float)
    mid = (tip + jaw) / 2; pxr = np.linalg.norm(tip - jaw) / 2
    m, why = AF.fit(R, L, tuple(mid), tuple(jaw), z=z, verbose=False, cut=tuple(mid), cut_r=1.12 * pxr)
    if m is None: print(k, "FAILED", why); continue
    T = K.fk_ecm(np.array(st["q"]), SUJ); C, tp, tan = m["centre"], m["tip_cam"], m["dir_cam"]; x = (tp - C) / np.linalg.norm(tp - C); zz = np.cross(x, tan); zz /= np.linalg.norm(zz)
    Rtn = T[:3, :3].T @ Rc.T @ np.c_[x, np.cross(zz, x), zz]; p = T[:3, :3].T @ (cam2ecm_mm(C, M) * 1e-3 - T[:3, 3]); tp_t = T[:3, :3].T @ (cam2ecm_mm(tp, M) * 1e-3 - T[:3, 3]); tl_t = T[:3, :3].T @ (cam2ecm_mm(m["tail_cam"], M) * 1e-3 - T[:3, 3])
    out[k] = dict(Rtn=Rtn.tolist(), p=p.tolist(), tip_tool=tp_t.tolist(), tail_tool=tl_t.tolist(), normal_cam=m["normal"].tolist(), tip_cam=tp.tolist(), tail_cam=m["tail_cam"].tolist(), centre=C.tolist(), dir_cam=tan.tolist(), span=m["span_deg"], cost=m["cost_px"], end_err=m["end_err_px"], eff=st["eff"], q=st["q"])
    print(k, "wrist", np.round(np.degrees(st["q"][3:]), 0).tolist(), "cost %.2f end %.1f span %.0f | tool mm: tip %s tail %s | normal tool %s | eff %.3f" % (m["cost_px"], m["end_err_px"], m["span_deg"], np.round(tp_t * 1e3, 1).tolist(), np.round(tl_t * 1e3, 1).tolist(), np.round(Rtn[:, 2], 2).tolist(), st["eff"]), flush=True)
    P = m["arc"]; uR = np.c_[FX * P[:, 0] / P[:, 2] + CX, FX * P[:, 1] / P[:, 2] + CY]; o = R.copy(); cv2.polylines(o, [np.int32(uR)], False, (0, 255, 0), 1); cv2.polylines(o, [np.int32(m["cR"])], False, (255, 0, 0), 1); cv2.imwrite(v + "/arc.png", o)
ks = list(out); Rs = np.array([out[k]["Rtn"] for k in ks]); Rm = Ro.from_matrix(Rs).mean().as_matrix(); ang = [np.degrees(Ro.from_matrix(Rm.T @ r).magnitude()) for r in Rs]; ps = np.array([out[k]["p"] for k in ks]) * 1e3
print("rotation spread about the mean, deg:", dict(zip(ks, np.round(ang, 1).tolist()))); print("centre mean mm", np.round(ps.mean(0), 1).tolist(), "std", np.round(ps.std(0), 1).tolist())
json.dump(out, open(d + "/views.json", "w"), indent=1)
