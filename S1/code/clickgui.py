#!/usr/bin/env python3
"""Click windows for the two human inputs of an attempt (OpenCV/Qt, DISPLAY of the console, window placed on DP-2).

needle:   physical-left image (topic right) with the AUTOMATIC polyline (needle_detect) drawn in green.
          'a' = accept the green line (explicit; nothing is accepted by default).  Or click points TAIL (thread end) -> TIP, ENTER.
          u = undo, r = reset, ESC = abort the attempt.
template: 2x zoom around the jaw.  Click 1 = a distinct jaw feature (wrist/jaw texture), click 2 = the grasp point (between the
          jaw tips) -> ENTER; then the other view: click the same grasp point -> ENTER.  u = undo, ESC = abort.
"""
import numpy as np
import cv2

WIN_XY = (60, 40)


def _loop(name, draw, on_click, n_needed=None):
    clicks = []
    def cb(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN: clicks.append(on_click(x, y))
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE); cv2.moveWindow(name, *WIN_XY); cv2.setMouseCallback(name, cb)
    while True:
        cv2.imshow(name, draw(clicks)); k = cv2.waitKey(30) & 0xFF
        if k == 27: cv2.destroyWindow(name); cv2.waitKey(1); return None
        if k in (ord("u"),) and clicks: clicks.pop()
        if k == ord("r"): clicks.clear()
        if k == ord("a") and n_needed is None and not clicks: cv2.destroyWindow(name); cv2.waitKey(1); return "ACCEPT_AUTO"
        if k in (13, 10):
            if (n_needed is None and len(clicks) >= 4) or (n_needed is not None and len(clicks) == n_needed): cv2.destroyWindow(name); cv2.waitKey(1); return clicks
        if n_needed is not None and len(clicks) > n_needed: clicks.pop()


def click_needle(img, auto_poly=None, title="needle"):
    s = 1.0
    def draw(c):
        v = img.copy()
        if auto_poly is not None and not c: cv2.polylines(v, [np.round(auto_poly).astype(np.int32)], False, (0, 255, 0), 2); cv2.circle(v, tuple(np.round(auto_poly[-1]).astype(int)), 7, (0, 0, 255), 2)
        for i, p in enumerate(c):
            cv2.circle(v, (int(p[0]), int(p[1])), 4, (0, 255, 255), -1)
            if i: cv2.line(v, (int(c[i - 1][0]), int(c[i - 1][1])), (int(p[0]), int(p[1])), (0, 255, 255), 1)
        txt = "'a' = accept green auto line (red circle = tip), or click TAIL -> TIP" if (auto_poly is not None and not c) else "click TAIL(thread end) -> TIP (>= 4 points), ENTER when done"
        cv2.putText(v, txt + "   u undo  r reset  ESC abort", (10, 30), 0, 0.8, (255, 255, 255), 2); return v
    c = _loop(title, draw, lambda x, y: [x / s, y / s])
    if c is None: return None, "aborted"
    if c == "ACCEPT_AUTO":
        return (None, "no auto line") if auto_poly is None else (np.asarray(auto_poly, float).tolist(), "auto accepted by user ('a')")
    return c, "manual clicks"


def _zoom(img, centre, z=2.0, size=(1300, 1024)):
    w, h = int(size[0] / z), int(size[1] / z)
    x0 = int(np.clip(centre[0] - w / 2, 0, img.shape[1] - w)); y0 = int(np.clip(centre[1] - h / 2, 0, img.shape[0] - h))
    return cv2.resize(img[y0:y0 + h, x0:x0 + w], size, interpolation=cv2.INTER_CUBIC), (x0, y0, z)


def click_template(imgR, imgL, centre_R, disp_guess):
    zr, (x0, y0, z) = _zoom(imgR, centre_R)
    def drawR(c):
        v = zr.copy()
        for i, p in enumerate(c):
            q = (int((p[0] - x0) * z), int((p[1] - y0) * z)); cv2.circle(v, q, 6, (0, 255, 255) if i == 0 else (0, 0, 255), 2)
        cv2.putText(v, "view A: click 1 = jaw FEATURE, click 2 = GRASP point (between jaw tips), ENTER   u undo ESC abort", (10, 30), 0, 0.7, (255, 255, 255), 2); return v
    c = _loop("template A (physical left)", drawR, lambda x, y: [x0 + x / z, y0 + y / z], n_needed=2)
    if c is None: return None
    zl, (x1, y1, z1) = _zoom(imgL, (c[1][0] - disp_guess, c[1][1]))
    def drawL(c2):
        v = zl.copy(); cv2.line(v, (0, int((c[1][1] - y1) * z1)), (v.shape[1], int((c[1][1] - y1) * z1)), (80, 80, 80), 1)
        for p in c2: cv2.circle(v, (int((p[0] - x1) * z1), int((p[1] - y1) * z1)), 6, (0, 0, 255), 2)
        cv2.putText(v, "view B: click the SAME grasp point (grey line = same row), ENTER   u undo ESC abort", (10, 30), 0, 0.7, (255, 255, 255), 2); return v
    c2 = _loop("template B (physical right)", drawL, lambda x, y: [x1 + x / z1, y1 + y / z1], n_needed=1)
    if c2 is None: return None
    return dict(feat_R=[int(round(c[0][0])), int(round(c[0][1]))], grasp_R=[float(c[1][0]), float(c[1][1])], grasp_L=[float(c2[0][0]), float(c2[0][1])], half=16)
