#!/usr/bin/env python3
"""1d stage 0b: bring the tracked jaw feature to the calibration base (operator authorised 2026-10-25, "dvrk所有配件你都可以动").

Same mechanics as r6_calib.py (266e422f...): fixed orientation (current setpoint), move_cp sub-steps <= 3 mm, per-view temporal
template tracking (phys-left = topic right, left located with FS disparity), NCC gates 0.6 at start / 0.5 after moves.
  1. probe: +2 mm along cp x, y, z (each returns) -> J (cam mm per cp mm)
  2. goto : closed loop, step = J^-1 (X_target - X) clipped to 3 mm, until |X_target - X| < 1.5 mm (max 25 iterations)
Target: FS pad point at TARGET_UV (pre3) + TARGET_H mm along the pad normal (towards the camera).
Stops (no retry) on: NCC below gate, move not reached, predicted or measured feature height < MIN_H, iteration cap.
"""
import os, sys
if 'AMENT_PREFIX_PATH' not in os.environ:
    import shlex
    os.execvp('bash', ['bash', '-c', 'source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash && export ROS_AUTOMATIC_DISCOVERY_RANGE=SYSTEM_DEFAULT FASTDDS_BUILTIN_TRANSPORTS=UDPv4 && exec python3 '
                       + ' '.join(shlex.quote(x) for x in [os.path.abspath(__file__)] + sys.argv[1:])])
import traceback; sys.excepthook = lambda t, v, tb: (traceback.print_exception(t, v, tb), sys.stdout.flush(), sys.stderr.flush(), os._exit(1))
import json, time, threading, numpy as np, cv2, rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import message_filters as mf

ROOT = os.path.expanduser('~/rl_live/sim2real_20261025/step1d_calib/calib')
PRE = ROOT + '/pre3_high'
FX, CX, CY = 1860.724932, 690.538345, 576.803638; FB = FX * 5.4476353950418295
FEAT_UV = (455, 485); HALF = 35
TARGET_UV = (700, 720); TARGET_H = 15.0
STEP = 3.0; MIN_H = 8.0; NCC0, NCC1 = 0.6, 0.5; TOL = 1.5; MAX_IT = 25

def cam_points(disp):
    Z = FB / np.clip(disp, 1, None); H, W = disp.shape; v, u = np.mgrid[0:H, 0:W]; return np.dstack([(u - CX) * Z / FX, (v - CY) * Z / FX, Z])

P = cam_points(np.load(PRE + '/fs_disp.npy'))
Q = P[640:900, 520:820].reshape(-1, 3)[::7]; rng = np.random.default_rng(0); best = None
for _ in range(400):
    s = Q[rng.choice(len(Q), 3, replace=False)]; nn = np.cross(s[1] - s[0], s[2] - s[0])
    if np.linalg.norm(nn) < 1e-9: continue
    nn /= np.linalg.norm(nn); k = (np.abs((Q - s[0]) @ nn) < 0.5).sum()
    if best is None or k > best[0]: best = (k, nn, s[0])
I = Q[np.abs((Q - best[2]) @ best[1]) < 0.5]; c_pad = I.mean(0); n_up = np.linalg.svd(I - c_pad)[2][2]; n_up = n_up if n_up[2] < 0 else -n_up
X_pad_t = np.median(P[TARGET_UV[1] - 3:TARGET_UV[1] + 4, TARGET_UV[0] - 3:TARGET_UV[0] + 4].reshape(-1, 3), 0)
X_t = X_pad_t + TARGET_H * n_up

LOG = open(ROOT + '/goto_base_log.jsonl', 'a')
def rec(**k): k['t'] = time.time(); LOG.write(json.dumps(k, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n'); LOG.flush()
rec(ev='plane', c_pad=c_pad, n_up=n_up, inliers=int(len(I)), of=int(len(Q)), X_pad_target=X_pad_t, X_target=X_t)
print(f'pad plane inliers {len(I)}/{len(Q)} normal {n_up.round(3)}; target feature {X_t.round(1)} mm (cam)', flush=True)

rclpy.init(); node = rclpy.create_node('goto_base_1d'); br = CvBridge(); L = {}; grab = {'buf': None, 'n': 0}
subs = [mf.Subscriber(node, Image, f'/jhu_daVinci/{c}/image_rect', qos_profile=qos_profile_sensor_data) for c in ('left', 'right')]
sy = mf.ApproximateTimeSynchronizer(subs, 10, 0.02)
sy.registerCallback(lambda l, r: grab['buf'].append((br.imgmsg_to_cv2(l, 'bgr8').astype(np.float32), br.imgmsg_to_cv2(r, 'bgr8').astype(np.float32))) if grab['buf'] is not None and len(grab['buf']) < grab['n'] else None)
for k, t in (('cp', '/PSM1/measured_cp'), ('sp', '/PSM1/setpoint_cp')):
    node.create_subscription(PoseStamped, t, lambda msg, k=k: L.__setitem__(k, msg), 10)
node.create_subscription(Bool, '/PSM1/goal_reached', lambda msg: L.__setitem__('reached', msg.data), 10)
pub = node.create_publisher(PoseStamped, '/PSM1/move_cp', 1)
threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start(); t0 = time.time()
while not all(k in L for k in ('cp', 'sp')) and time.time() - t0 < 10: time.sleep(0.05)
assert all(k in L for k in ('cp', 'sp')), 'no PSM1 state'
Pm = lambda msg: np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]) * 1000
o = L['sp'].pose.orientation; qref = [o.x, o.y, o.z, o.w]

def move_abs(cp_mm):
    cur = Pm(L['sp']); d = np.asarray(cp_mm) - cur; k = max(1, int(np.ceil(np.linalg.norm(d) / STEP)))
    for j in range(1, k + 1):
        tgt = cur + d * j / k; msg = PoseStamped(); msg.header.frame_id = L['sp'].header.frame_id
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, tgt / 1000)
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = qref
        L.pop('reached', None); pub.publish(msg); t = time.time()
        while 'reached' not in L and time.time() - t < 10: time.sleep(0.02)
        time.sleep(0.4)
        if not L.get('reached'): rec(ev='move_not_reached', tgt=tgt); return False
    time.sleep(0.6); return True

def capture(k=6):
    grab['n'] = k; grab['buf'] = []; t = time.time()
    while len(grab['buf']) < k and time.time() - t < 10: time.sleep(0.02)
    b = grab['buf']; grab['buf'] = None; return np.mean([x[0] for x in b], 0), np.mean([x[1] for x in b], 0)

B0 = cv2.cvtColor(cv2.imread(PRE + '/right_rect.png'), cv2.COLOR_BGR2GRAY).astype(np.float32)
B0L = cv2.cvtColor(cv2.imread(PRE + '/left_rect.png'), cv2.COLOR_BGR2GRAY).astype(np.float32)
U0, V0 = FEAT_UV; T0 = B0[V0 - HALF:V0 + HALF + 1, U0 - HALF:U0 + HALF + 1]
D0 = float(np.median(np.load(PRE + '/fs_disp.npy')[V0 - 2:V0 + 3, U0 - 2:U0 + 3])); UL0 = int(round(U0 - D0))
T0L = B0L[V0 - HALF:V0 + HALF + 1, UL0 - HALF:UL0 + HALF + 1]
def sub1(r, j):
    if 0 < j < len(r) - 1:
        p, q, s = r[j - 1], r[j], r[j + 1]; den = p - 2 * q + s; return j + (0.5 * (p - s) / den if den < 0 else 0)
    return float(j)
def match(img, T, cu, cv_, rad=120):
    best = None
    for s in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15):
        Ts = cv2.resize(T, None, fx=s, fy=s); th, tw = Ts.shape
        x0, y0 = int(max(cu - rad - tw / 2, 0)), int(max(cv_ - rad - th / 2, 0)); x1, y1 = int(min(cu + rad + tw / 2, 1300)), int(min(cv_ + rad + th / 2, 1024))
        r = cv2.matchTemplate(img[y0:y1, x0:x1], Ts, cv2.TM_CCOEFF_NORMED); _, mx, _, (lx, ly) = cv2.minMaxLoc(r)
        if best is None or mx > best[0]: best = (mx, x0 + sub1(r[ly], lx) + tw / 2 - 0.5, y0 + sub1(r[:, lx], ly) + th / 2 - 0.5)
    return best
def track(name, pred_uv, pred_disp=None):
    Lm, Rm = capture(); d = f'{ROOT}/goto_pts/{name}'; os.makedirs(d, exist_ok=True)
    cv2.imwrite(d + '/right_rect.png', Rm.round().astype(np.uint8)); cv2.imwrite(d + '/left_rect.png', Lm.round().astype(np.uint8))
    R = cv2.cvtColor(Rm.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32); Lg = cv2.cvtColor(Lm.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    mx, u, v = match(R, T0, *pred_uv); pd = pred_disp if pred_disp is not None else D0
    m2, uL, vL = match(Lg, T0L, u - pd, v, rad=60); dsp = u - uL; Z = FB / dsp
    X = np.array([(u - CX) * Z / FX, (v - CY) * Z / FX, Z]); h = float((X - c_pad) @ n_up)
    rec(ev='track', name=name, uv=[u, v], uvL=[uL, vL], dv=v - vL, ncc=mx, ncc_d=m2, disp=dsp, X=X, h_pad=h, cp=Pm(L['cp']), sp=Pm(L['sp']))
    return X, mx, m2, h, (u, v), dsp
proj = lambda X: (FX * X[0] / X[2] + CX, FX * X[1] / X[2] + CY)
def stop(msg):
    rec(ev='stop', reason=msg); print('STOP:', msg, flush=True); sys.stdout.flush(); os._exit(3)

base = Pm(L['sp']); rec(ev='start', base_cp_mm=base, qref=qref, feat=FEAT_UV, D0=D0)
X0, c1, c2, h0, uv0, d0 = track('probe_base', FEAT_UV)
print(f'base feature {X0.round(1)} ncc {c1:.2f}/{c2:.2f} h {h0:.1f} mm disp {d0:.1f} (FS {D0:.1f})', flush=True)
if c1 < NCC0 or c2 < NCC0: stop(f'low NCC at base {c1:.2f}/{c2:.2f}')
J = np.zeros((3, 3))
for i, dv in enumerate(([2, 0, 0], [0, 2, 0], [0, 0, 2])):
    if not move_abs(base + np.array(dv, float)): stop('probe move not reached')
    X, c1, c2, h, uv, dsp = track(f'probe_{i}', proj(X0), d0); J[:, i] = (X - X0) / 2.0
    print(f'probe {i} {dv}: dX {(X - X0).round(2)} ncc {c1:.2f}/{c2:.2f} h {h:.1f}', flush=True)
    if c1 < NCC1 or c2 < NCC1: stop(f'low NCC on probe {i}')
    if not move_abs(base): stop('probe return not reached')
sv = np.linalg.svd(J)[1]; rec(ev='probe_J', J=J, sv=sv); print('J', J.round(3).tolist(), 'sv', sv.round(3), flush=True)
if sv.min() < 0.5 or sv.max() > 2.0: stop(f'Jacobian not near-rigid, sv {sv.round(3)}')
Ji = np.linalg.inv(J); X, dsp, uv = X0, d0, uv0
for it in range(MAX_IT):
    e = X_t - X
    if np.linalg.norm(e) < TOL: break
    dcp = Ji @ e; nrm = np.linalg.norm(dcp)
    if nrm > STEP: dcp *= STEP / nrm
    Xp = X + J @ dcp; hp = float((Xp - c_pad) @ n_up)
    if hp < MIN_H: stop(f'predicted feature height {hp:.1f} < {MIN_H}')
    if not move_abs(Pm(L['sp']) + dcp): stop('goto move not reached')
    X, c1, c2, h, uv, dsp = track(f'goto_{it:02d}', proj(Xp), dsp)
    print(f'goto {it:02d}: |e| {np.linalg.norm(X_t - X):5.2f} mm h {h:5.1f} ncc {c1:.2f}/{c2:.2f} uv {np.round(uv, 0)}', flush=True)
    if c1 < NCC1 or c2 < NCC1: stop(f'low NCC at goto {it}')
    if h < MIN_H: stop(f'measured feature height {h:.1f} < {MIN_H}')
else:
    stop('iteration cap reached')
rec(ev='done', cp_mm=Pm(L['cp']), sp_mm=Pm(L['sp']), X=X, uv=uv, err=float(np.linalg.norm(X_t - X)))
json.dump({'feature_uv_now': [float(uv[0]), float(uv[1])], 'disp_now': float(dsp), 'X_now': X.tolist(), 'base_cp_mm': Pm(L['sp']).tolist(),
           'qref': qref, 'J': J.tolist(), 'target': X_t.tolist()}, open(ROOT + '/goto_base_result.json', 'w'), indent=1)
print(f'done: feature at {X.round(1)} (target {X_t.round(1)}), uv {np.round(uv, 1)}', flush=True)
sys.stdout.flush(); os._exit(0)
