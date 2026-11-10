"""READ-ONLY: tip height above / lateral offset from the entry sticker for hand-read stereo tip pixels. usage: tip_metric.py measure.json uR vR uL"""
import sys, json, numpy as np
FX, CX, CY = 1860.724932, 690.538345, 576.803638; FB = FX * 5.4476353950418295
m = json.load(open(sys.argv[1])); en = min(m["stickers"], key=lambda s: s["area"]); e = np.array(en["cam"]); n = np.array(en["n25"]); n = n if n[2] < 0 else -n
uR, vR, uL = map(float, sys.argv[2:5])
for du in (0, -1, 1):
    Z = FB / (uR - uL - du); X = np.array([(uR - CX) * Z / FX, (vR - CY) * Z / FX, Z]); d = X - e
    print("disp %.1f: Z %.1f, height above entry %.1f mm, lateral %.1f mm" % (uR - uL - du, Z, d @ n, np.linalg.norm(d - (d @ n) * n)))
