#!/usr/bin/env python3
"""READ-ONLY. Rough needle points in the tool frame for two grips (NCC stereo + FK + M).  Hand-seeded right-image arc points."""
import json, os, sys
import numpy as np, cv2
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code", R0 + "/P_place", os.path.expanduser("~/rl_live/sim2real_20261025/step1d_calib")]
import kin as K
from calib import cam2ecm_mm
from ncc_disp import disp, FB
FX, CX, CY = 1860.724932, 690.538345, 576.803638; cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); SUJ = np.array(cal["frozen"]["suj_psm1"])
def run(name, d, q, seeds, z):
    R = cv2.imread(d + "/right_rect.png"); L = cv2.imread(d + "/left_rect.png"); T = K.fk_ecm(np.array(q), SUJ); out = []
    print(name, "wrist deg", np.round(np.degrees(np.array(q)[3:]), 0).tolist())
    for a, b in seeds:
        r = max((disp(R, L, a, b, hw=h, d0=FB / z, rng=28) for h in (7, 9)), key=lambda x: x[0]); Zv = FB / r[1]; X = np.array([(a - CX) * Zv / FX, (b - CY) * Zv / FX, Zv]); pt = T[:3, :3].T @ (cam2ecm_mm(X, M) * 1e-3 - T[:3, 3]) * 1e3; out.append(pt); print("   ", (a, b), "ncc %.2f d %.1f Z %.1f tool mm %s" % (r[0], r[1], Zv, np.round(pt, 1).tolist()))
    return np.array(out)
dem = json.load(open("../G_place/correct_143942/demo.json"))
A = run("DEMO grip", "../G_place/correct_143942/snap", dem["js_mean"], [(722, 570), (696, 625), (715, 668), (753, 685), (800, 693), (850, 690), (890, 682)], 108)
st = json.load(open("set2c/state.json")); seeds = json.loads(sys.argv[1])
Bq = run("NEW grip", "set2c", st["q"], seeds, 112)
