#!/usr/bin/env python3
"""READ-ONLY. Averaged stereo frames + zoomed crops around an expected tip pixel, for hand-checking the tip."""
import json, os, sys, time
import numpy as np, cv2
R0 = os.path.expanduser("~/rl_live/sim2real_20261106/S1"); sys.path[:0] = [R0 + "/code"]
from rig import RigIO, FB
out, u, v, d = sys.argv[1], *map(float, sys.argv[2:5]); r = int(sys.argv[5]) if len(sys.argv) > 5 else 70; f = 5; os.makedirs(out, exist_ok=True)
io = RigIO(images=True); time.sleep(2.5); b = io.frames(8); L = np.mean([z[0] for z in b], 0).round().astype(np.uint8); R = np.mean([z[1] for z in b], 0).round().astype(np.uint8); cv2.imwrite(out + "/left_rect.png", L); cv2.imwrite(out + "/right_rect.png", R)
json.dump(dict(q=np.array(io.get("js")[1][:6]).tolist(), cp=np.asarray(io.get("cp")[1]).tolist(), jaw=float(io.get("jaw")[1][0]), eff=float(io.get("jaw_eff")[1][0])), open(out + "/state.json", "w"), indent=1)
def crop(I, x, y):
    x, y = int(max(r, min(I.shape[1] - r, x))), int(max(r, min(I.shape[0] - r, y))); im = cv2.resize(I[y - r:y + r, x - r:x + r], None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    for k in range(0, 2 * r, 10): cv2.line(im, (k * f, 0), (k * f, 6 if k % 50 else 16), (255, 255, 255), 1); cv2.line(im, (0, k * f), (6 if k % 50 else 16, k * f), (255, 255, 255), 1)
    cv2.putText(im, f"x0={x - r} y0={y - r} tick=10px", (25, 25), 0, .6, (255, 255, 0), 2); return im
cv2.imwrite(out + "/crop.png", np.hstack([crop(L, u - d, v + 2), crop(R, u, v)])); print("jaw %.3f eff %.3f wrist %s" % (float(io.get("jaw")[1][0]), float(io.get("jaw_eff")[1][0]), np.round(np.degrees(np.array(io.get("js")[1][3:6])), 1).tolist()), flush=True); sys.stdout.flush(); os._exit(0)
