#!/usr/bin/env python3
"""Offline replay of the 14:2x autotpl change (gate + fallback + solid-metal feature) on the saved attempts A03-A14 — no robot.

Per attempt: open25 images + logged template inputs -> new auto_template (with the grasp prior, as attempt.py now does) ->
the template stereo / patch step of attempt.open_and_template -> the track10 matching on the saved hover frames
(hover_init at the template pose; hover_c<k> after the logged hover translations, pred = X_feat + d).  Only the first frame of each
hover check was saved, so the replay matches 1 frame per check (online: 5-10).
Out: ../F_attempts/replay_tpl_1428.json + one overlay per attempt (replay_tpl_1428.png in the attempt folder)."""
import glob, json, os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig import FB, TCP_MM
from calib import match, tri, proj
from autotpl import auto_template, overlay
from attempt import OFF_TOOL_PRIOR_MM

R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1")
cal = json.load(open(R0 + "/C_r6cal/calibration_tcp.json")); M = np.array(cal["M"]); A3 = M[:3]


def events(d):
    return [json.loads(l) for l in open(d + "/trial_log.jsonl")]


def replay(d):
    ev = events(d); tpl = next((r for r in ev if r.get("ev") == "phase" and r.get("name") == "template"), None)
    if tpl is None or not os.path.exists(d + "/open25/right_rect.png"): return dict(attempt=os.path.basename(os.path.normpath(d)), skipped="no template phase / open25")
    sp = np.array(json.load(open(d + "/open25/state.json"))["sp"]); Rnow = sp[:3, :3]
    Xt = np.array(tpl["tip_model_cam"], float); c = proj(Xt); dg = FB / Xt[2]
    cpx = proj(np.r_[sp[:3, 3] * 1e3, 1] @ M)
    Xp = Xt + (Rnow @ OFF_TOOL_PRIOR_MM) @ A3; gpR = proj(Xp); prior = dict(R=gpR.tolist(), L=[float(gpR[0] - FB / Xp[2]), float(gpR[1])])
    R = cv2.imread(d + "/open25/right_rect.png"); L = cv2.imread(d + "/open25/left_rect.png")
    old = json.load(open(d + "/template_clicks.json"))
    out = dict(attempt=os.path.basename(os.path.normpath(d)), old_feat_R=old["feat_R"], old_c_off_cam=tpl["c_off_cam"])
    if OLD:                                                       # replay check: the template that ran online
        ck = dict(old, mode="online", fallback_why=""); ck.setdefault("debug", dict(tips_R=[], feat_metal_frac=None)); ck["debug"].setdefault("feat_metal_frac", None)
    else:
        try: ck = auto_template(R, L, cpx, c, dg, grasp_prior=prior)
        except ValueError as e: out.update(template="FAILED (quality stop)", why=str(e)); return out
        cv2.imwrite(d + "/replay_tpl_1428.png", overlay(R, ck))
    tips = np.array(ck["debug"].get("tips_R", []), float); out.update(mode=ck["mode"], why=ck["fallback_why"], feat_R=ck["feat_R"], feat_metal_frac=ck["debug"].get("feat_metal_frac"),
                                                    tip_sep_px=float(np.linalg.norm(tips[0] - tips[1])) if len(tips) == 2 else None)
    # attempt.open_and_template: patch + stereo (unchanged code path)
    Rg = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY).astype(np.float32); Lg = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hf = 16; fu, fv = ck["feat_R"]; TR = Rg[fv - hf:fv + hf + 1, fu - hf:fu + hf + 1]
    dgc = ck["grasp_R"][0] - ck["grasp_L"][0]; best = None
    for dd in np.arange(dgc - 6, dgc + 6.01, 1):
        x = int(round(fu - dd)); W = Lg[fv - hf - 3:fv + hf + 4, x - hf - 3:x + hf + 4]
        if W.shape[0] < TR.shape[0] or W.shape[1] < TR.shape[1]: continue
        r = cv2.matchTemplate(W, TR, cv2.TM_CCOEFF_NORMED); _, mx, _, (lx, ly) = cv2.minMaxLoc(r)
        if best is None or mx > best[0]: best = (mx, x - 3 + lx, fv - 3 + ly)
    mx, xl, yl = best; disp0 = fu - xl; TL = Lg[yl - hf:yl + hf + 1, xl - hf:xl + hf + 1]
    Xf = tri(fu, fv, xl); Xg = tri(ck["grasp_R"][0], ck["grasp_R"][1], ck["grasp_L"][0]); c_off = Xg - Xt
    off_tool = Rnow.T @ (np.asarray(c_off) @ np.linalg.inv(A3))
    out.update(stereo_ncc=float(mx), c_off_cam=c_off.tolist(), off_tool_mm=off_tool.tolist())
    # hover checks: positions from the logged step targets (setpoint before hover_init = template pose)
    p_last, p_tpl, checks = None, None, []
    for r in ev:
        if r.get("ev") == "step": p_last = np.array(r["target"])[:3, 3]
        if r.get("ev") == "track10":
            name = r["name"]
            if name == "hover_init": p_tpl = p_last
            if p_tpl is None or not os.path.exists(f"{d}/{name}/right_rect.png"): continue
            dcam = ((p_last - p_tpl) * 1e3) @ A3; pred = proj(Xf + dcam)
            Ri = cv2.cvtColor(cv2.imread(f"{d}/{name}/right_rect.png"), cv2.COLOR_BGR2GRAY).astype(np.float32)
            Li = cv2.cvtColor(cv2.imread(f"{d}/{name}/left_rect.png"), cv2.COLOR_BGR2GRAY).astype(np.float32)
            m1, u, v = match(Ri, TR, pred[0], pred[1], 60); m2, uL, vL = match(Li, TL, u - disp0, v, 25)
            ok = m1 >= 0.5 and m2 >= 0.5 and u - uL > 1
            err_px = float(np.linalg.norm(np.array([u, v]) - pred))
            checks.append(dict(name=name, moved_mm=float(np.linalg.norm(p_last - p_tpl) * 1e3), ncc=[float(m1), float(m2)], valid=bool(ok),
                               match_minus_pred_px=err_px, online_n_valid=r["n_valid"], online_n_frames=r["n_frames"]))
    out["checks"] = checks; return out


OLD = "--old" in sys.argv          # replay the online templates instead (checks that the 1-frame replay reproduces the online outcome)

if __name__ == "__main__":
    res = [replay(d) for d in sorted(glob.glob(R0 + "/F_attempts/A*_*/"))]
    json.dump(res, open(R0 + "/F_attempts/replay_tpl_1428%s.json" % ("_old" if OLD else ""), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    f = lambda v: None if v is None else np.round(np.asarray(v, float), 2).tolist()
    for r in res:
        if "skipped" in r or "template" in r: print(r["attempt"], r.get("skipped") or r["template"], r.get("why", "")); continue
        print(r["attempt"], r["mode"], "sep", f(r["tip_sep_px"]), "feat", r["feat_R"], "(old", r["old_feat_R"], ") metal", f(r["feat_metal_frac"]), "ncc", f(r["stereo_ncc"]),
              "off_tool", f(r["off_tool_mm"]), "|", r["why"][:60])
        for c in r["checks"]: print("    ", c["name"], "moved", f(c["moved_mm"]), "ncc", f(c["ncc"]), "valid", c["valid"], "dpx", f(c["match_minus_pred_px"]), "| online", c["online_n_valid"], "/", c["online_n_frames"])
