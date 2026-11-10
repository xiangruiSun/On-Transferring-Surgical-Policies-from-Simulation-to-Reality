#!/usr/bin/env python3
"""Offline replay (no robot) of the hover tracker's left-image match: R6 match(rad=25) vs calib.match_band (|dv| <= 2 px, band 25).

Per saved hover check (first frame only; online 5-10): right match as online (pred from the logged pose), left match both ways,
X_grasp = X_feat + delta (the attempt's template), compared with the kinematic reference = model TCP at the logged setpoint + c_off
(calibration held-out error 0.7-0.9 mm).  Templates: the online ones (tpl_R.png / tpl_L.png, disp0 and delta from the log).
Out: ../F_attempts/replay_track_1450.json"""
import glob, json, os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig import tip_of
from calib import match, match_band, tri, proj

R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1")
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); A3 = M[:3]


def replay(d):
    ev = [json.loads(l) for l in open(d + "/trial_log.jsonl")]
    tpl = next((r for r in ev if r.get("ev") == "phase" and r.get("name") == "template"), None)
    if tpl is None or not os.path.exists(d + "/tpl_R.png"): return None
    TR = cv2.imread(d + "/tpl_R.png", 0).astype(np.float32); TL = cv2.imread(d + "/tpl_L.png", 0).astype(np.float32)
    disp0, delta, coff = tpl["disp0"], np.array(tpl["delta_cam"]), np.array(tpl["c_off_cam"])
    T_last, X_prev, rows = None, None, []
    for r in ev:
        if r.get("ev") == "step": T_last = np.array(r["target"])
        if r.get("ev") != "track10" or T_last is None or not os.path.exists(f"{d}/{r['name']}/right_rect.png"): continue
        Xk = np.r_[tip_of(T_last) * 1e3, 1] @ M + coff                                     # kinematic reference of the grasp point
        if r["name"] == "hover_init" or X_prev is None: X_feat_ref = Xk - delta
        pred = proj(Xk - delta)
        Rg = cv2.imread(f"{d}/{r['name']}/right_rect.png", 0).astype(np.float32); Lg = cv2.imread(f"{d}/{r['name']}/left_rect.png", 0).astype(np.float32)
        m1, u, v = match(Rg, TR, pred[0], pred[1], 60)
        out = dict(attempt=os.path.basename(os.path.normpath(d)), name=r["name"], ncc_R=m1, online_n_valid=r["n_valid"],
                   online_X_grasp=r.get("X_grasp"), X_kin=Xk.tolist())
        for key, (m2, uL, vL) in (("rad25", match(Lg, TL, u - disp0, v, 25)), ("band", match_band(Lg, TL, u - disp0, v, 25))):
            ok = m1 >= 0.5 and m2 >= 0.5 and u - uL > 1
            Xg = tri(u, v, uL) + delta if u - uL > 1 else np.full(3, np.nan)
            out[key] = dict(ncc_L=m2, dv=vL - v, valid=bool(ok), X_grasp=Xg.tolist(), err_vs_kin_mm=float(np.linalg.norm(Xg - Xk)))
        if r.get("X_grasp") is not None: out["online_err_vs_kin_mm"] = float(np.linalg.norm(np.array(r["X_grasp"]) - Xk))
        rows.append(out); X_prev = Xk
    return rows


if __name__ == "__main__":
    res = [x for d in sorted(glob.glob(R0 + "/F_attempts/A*_*/")) for x in (replay(d) or [])]
    json.dump(res, open(R0 + "/F_attempts/replay_track_1450.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    f = lambda v: "  nan" if v is None or not np.isfinite(v) else f"{v:5.1f}"
    print("attempt  check      nccR | online err/kin n_valid | rad25 dv nccL err/kin valid | band dv nccL err/kin valid")
    for x in res:
        a, b = x["rad25"], x["band"]
        print(f"{x['attempt']:9s} {x['name']:10s} {x['ncc_R']:.2f} | {f(x.get('online_err_vs_kin_mm'))} {x['online_n_valid']:2d} | "
              f"{a['dv']:6.1f} {a['ncc_L']:.2f} {f(a['err_vs_kin_mm'])} {a['valid']!s:5s} | {b['dv']:5.1f} {b['ncc_L']:.2f} {f(b['err_vs_kin_mm'])} {b['valid']}")
