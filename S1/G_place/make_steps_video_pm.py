#!/usr/bin/env python3
"""Step video of the 2026-10-30 afternoon placing (14:00-14:33) from the saved right-camera snapshots (no live recording existed)."""
import glob, json, os, subprocess
import cv2, numpy as np
G = os.path.expanduser("~/rl_live/sim2real_20261106/S1/G_place"); OUT = G + "/final_1030pm"
W = 960
steps = [("now_135356/snap/right_rect.png", "13:53 start: needle C-shaped, far from the site", 2.0),
         ("demo1/snap/right_rect.png", "Recorded demo1 (desired pose, before phantom moved)", 2.5),
         ("goto_demo_run2/right_after.png", "14:06 demo1 joints + phantom shift + 20 mm lift: tool out of view", 1.5),
         ("mc1/right_rect.png", "14:08 translate 12 mm parallel to tissue", 1.2),
         ("mc2/right_rect.png", "14:09 translate again: needle face-on (plane ~parallel to tissue)", 1.5),
         ("mc3/right_rect.png", "14:11 lift 9 mm", 1.2)]
for k in range(7): steps.append((f"wrist4/p{k:02d}/right_rect.png", f"14:1x wrist probe +-15 deg ({['base','roll+','roll-','pitch+','pitch-','yaw+','yaw-'][k]})", 0.6))
steps.append(("mc4/right_rect.png", "14:19 lift 15 mm (hinge ~43 mm above tissue)", 1.2))
for k in range(8): steps.append((f"wrist5/p{k:02d}/right_rect.png", f"wrist probe large offsets ({['p+45','p+70','p-43','y+43','y-45','y-75','p45y43','p45y-45'][k]})", 0.6))
for k in range(7): steps.append((f"wrist6/p{k:02d}/right_rect.png", f"roll sweep, jaw ~along shaft (roll {[0,-45,-90,45,90,135,-135][k]:+d}): stays face-on", 0.6))
for k in range(25):
    q = np.degrees(json.load(open(f"{G}/wrist7/p{k:02d}/state.json"))["q_cmd"][3:])
    steps.append((f"wrist7/p{k:02d}/right_rect.png", f"pitch x yaw grid: pitch {q[1]:.0f} yaw {q[2]:.0f}" + ("  <- edge-on" if k == 23 else ""), 1.5 if k == 23 else 0.35))
for k in range(4): steps.append((f"wrist8/p{k:02d}/right_rect.png", f"14:24 yaw steps at pitch 68: edge-on line turns toward bite ({k})", 0.9))
for k in range(3): steps.append((f"wrist9/p{k:02d}/right_rect.png", f"roll probe ({['-20','+20','-40'][k]})", 0.8))
steps += [("mc5/right_rect.png", "14:26 roll -20: edge-on, heading ~-27 deg", 1.5),
          ("mo1/right_rect.png", "14:28 outer-joint translation: needle back in view", 1.5),
          ("mo2/right_rect.png", "14:29 descend 15 mm: heading -30 deg = entry->exit", 1.5),
          ("mo3/right_rect.png", "14:29 descend 27 mm: tip over the entry, ~8-10 mm high", 1.8)]
for k in range(4): steps.append((f"wrist11/p{k:02d}/right_rect.png", f"tilt probe (pitch/yaw at limit): {['base','roll+10','pitch+4','yaw+1'][k]}", 0.7))
steps += [("mo5/right_rect.png", "14:32 final descent 5.5 mm", 1.5),
          ("final_1030pm/overlay.png", "FINAL: tip ~2.7 mm above entry, edge-on along entry->exit, not tip-down", 4.0)]
fps = 10; H = None; frames = 0
tmp = OUT + "/steps_raw.mp4"; vw = None
for i, (p, cap, dur) in enumerate(steps):
    im = cv2.imread(G + "/" + p)
    if im is None: print("missing", p); continue
    im = cv2.resize(im, (W, int(im.shape[0] * W / im.shape[1])))
    if vw is None: H = im.shape[0]; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + 50))
    canvas = np.zeros((H + 50, W, 3), np.uint8); canvas[50:] = im
    cv2.putText(canvas, f"{i + 1}/{len(steps)}  {cap}", (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for _ in range(max(1, int(round(dur * fps)))): vw.write(canvas); frames += 1
vw.release()
out = OUT + "/placing_steps_20261030_pm_h264.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out], check=True); os.remove(tmp)
print(f"{len(steps)} steps, {frames / fps:.0f} s -> {out}")
