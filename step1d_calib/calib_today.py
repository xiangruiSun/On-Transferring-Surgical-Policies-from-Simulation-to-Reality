#!/usr/bin/env python3
"""!!! DO NOT RUN AS IS (2026-10-25 end of day): the stereo depth of the thin jaw from template NCC (D-a, even epipolar)
locks onto the pad texture behind the jaw.  goto_base2.py (same tracker family) drove the jaw ~20 mm deeper than intended
(kinematics: feature from 16.8 mm to about -3..+2 mm above the pad; effort showed no contact).  Next version must take the
jaw depth from FS disparity at the tracked (u, v) and cross-check every move against probe-J x commanded displacement (> 3 mm -> stop).

1d (2026-10-25): on-site cam <- PSM1 calibration + pad plane from 4 tip touches.

Derived from the frozen R6 r6_calib.py (sha256 266e422f...) -- same probe / 12-point pad-frame grid / affine fit / 3 held-out
points / 3 mm gate / 3 mm sub-steps / fixed orientation / NCC gates -- with these logged deviations:
  D-a  left-image match restricted to the epipolar band (same row +-2 px, |dv| <= 1.5 px else stop); R6 searched 2-D +-60 px
       and drifted to dv -8 px today (depth jumps of +-5 mm)
  D-b  every sub-step IK-checked against the joint limits (kin.py) and a setpoint that stops advancing is a stop
       (dVRK reports goal_reached while IK clips at a limit)
  D-c  height gate relative to the LOCAL surface (pad + ridge) under the grid, not the flat pad plane
  D-d  feature near the upper-jaw tip, template half 20 px (R6: 'jaw centre near the tip', half 35)
  D-e  stage 5 (new): jaw closed (0 rad) for the touches, so the touching point is the jaw tip = TCP (R2)
Phases (run in order with --phase all; each also runs alone): lift pre manifest probe grid fit pad4 plane
Contact (stage 5) exactly as r6_grasp.py touch (R6_REPRODUCTION b7aac5f :372-381): baseline effort mean over 1 s at hover;
0.5 mm steps along the jaw axis; hit if |d ins effort| > 0.15 or max|d j0,j1 effort| > 0.05 or |ins tracking err| > 0.3 mm.
"""
import os, sys
if 'AMENT_PREFIX_PATH' not in os.environ:
    import shlex
    os.execvp('bash', ['bash', '-c', 'source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash && export ROS_AUTOMATIC_DISCOVERY_RANGE=SYSTEM_DEFAULT FASTDDS_BUILTIN_TRANSPORTS=UDPv4 && exec python3 -u '
                       + ' '.join(shlex.quote(x) for x in [os.path.abspath(__file__)] + sys.argv[1:])])
import traceback; sys.excepthook = lambda t, v, tb: (traceback.print_exception(t, v, tb), sys.stdout.flush(), sys.stderr.flush(), os._exit(1))
import argparse, json, subprocess, time, threading, numpy as np, cv2, rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as Rot
import message_filters as mf

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import kin
ap = argparse.ArgumentParser(); ap.add_argument('--phase', required=True, choices=['all', 'lift', 'pre', 'manifest', 'probe', 'grid', 'fit', 'pad4', 'plane']); a = ap.parse_args()
ROOT = HERE + '/calib3'; os.makedirs(ROOT, exist_ok=True); MAN = ROOT + '/manifest.json'
PREV = HERE + '/calib2'                           # goto_base2 output (template + J) used only for 'lift' and 'pre'
DEPLOY = os.path.expanduser('~/Desktop/surgicai_grasp_integration/surgic_ai_deployment_unpublished/deploy')
VENV = os.path.expanduser('~/surgical-rl-only/deploy/.venv-deploy/bin/python')
FX, CX, CY = 1860.724932, 690.538345, 576.803638; FB = FX * 5.4476353950418295
TCP = np.array([-2.70, 1.08, 23.81])            # R2-E ProGrasp 400093, tool frame mm
FEAT_OFFSET_PX = (-30, 20); HALF = 20           # D-d, from the goto-tracked point to near the upper-jaw tip
LIFT_MM = 5.0; STEP = 3.0; MIN_CLEAR = 8.0; NCC0, NCC1 = 0.6, 0.5; DV_MAX = 1.5; BAND = 2
GATE_TEST_MM = 3.0; PAD_SIDE_MM = (12.0, 15.0, 9.0, 18.0); NEEDLE_CLEAR_MM = 5.0; NEEDLE_PICK_MM = 10.0
HOVER_TIP_MM = 8.0; TOUCH_STEP = 0.5

LOG = open(ROOT + '/calib_log.jsonl', 'a')
def rec(**k): k['t'] = time.time(); k['date'] = time.strftime('%Y-%m-%dT%H:%M:%S%z'); LOG.write(json.dumps(k, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n'); LOG.flush()
def stop(msg): rec(ev='STOP', phase=a.phase, reason=msg); print('STOP:', msg, flush=True); sys.stdout.flush(); os._exit(3)
def jload(p): return json.load(open(p))
def jsave(p, d): json.dump(d, open(p, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
def cam_points(disp):
    Z = FB / np.clip(disp, 1, None); H, W = disp.shape; v, u = np.mgrid[0:H, 0:W]; return np.dstack([(u - CX) * Z / FX, (v - CY) * Z / FX, Z])
proj = lambda X: (FX * X[0] / X[2] + CX, FX * X[1] / X[2] + CY)
def ransac_plane(Q, tol=0.5, it=400):
    rng = np.random.default_rng(0); best = None
    for _ in range(it):
        s = Q[rng.choice(len(Q), 3, replace=False)]; n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9: continue
        n /= np.linalg.norm(n); k = (np.abs((Q - s[0]) @ n) < tol).sum()
        if best is None or k > best[0]: best = (k, n, s[0])
    I = Q[np.abs((Q - best[2]) @ best[1]) < tol]; c = I.mean(0); n = np.linalg.svd(I - c)[2][2]; n = n if n[2] < 0 else -n
    return c, n, len(I), len(Q)

# ---------------------------------------------------------------- ROS (not needed for 'fit' and 'plane')
NEED_ROS = a.phase not in ('fit', 'plane')
if NEED_ROS:
    rclpy.init(); node = rclpy.create_node('calib_today_1d'); br = CvBridge(); L = {}; grab = {'buf': None, 'n': 0}
    subs = [mf.Subscriber(node, Image, f'/jhu_daVinci/{c}/image_rect', qos_profile=qos_profile_sensor_data) for c in ('left', 'right')]
    sy = mf.ApproximateTimeSynchronizer(subs, 10, 0.02)
    sy.registerCallback(lambda l, r: grab['buf'].append((br.imgmsg_to_cv2(l, 'bgr8').astype(np.float32), br.imgmsg_to_cv2(r, 'bgr8').astype(np.float32))) if grab['buf'] is not None and len(grab['buf']) < grab['n'] else None)
    for k, t in (('cp', '/PSM1/measured_cp'), ('sp', '/PSM1/setpoint_cp'), ('suj', '/SUJ/PSM1/measured_cp')):
        node.create_subscription(PoseStamped, t, lambda msg, k=k: L.__setitem__(k, msg), qos_profile_sensor_data)
    for k, t in (('js', '/PSM1/measured_js'), ('spj', '/PSM1/setpoint_js'), ('jaw', '/PSM1/jaw/measured_js')):
        node.create_subscription(JointState, t, lambda msg, k=k: L.__setitem__(k, msg), qos_profile_sensor_data)
    node.create_subscription(Bool, '/PSM1/goal_reached', lambda msg: L.__setitem__('reached', msg.data), 10)
    pub = node.create_publisher(PoseStamped, '/PSM1/move_cp', 1); pjaw = node.create_publisher(JointState, '/PSM1/jaw/move_jp', 1)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start(); t0 = time.time()
    while not all(k in L for k in ('cp', 'sp', 'suj', 'js', 'spj', 'jaw')) and time.time() - t0 < 10: time.sleep(0.05)
    if not all(k in L for k in ('cp', 'sp', 'suj', 'js', 'spj', 'jaw')): stop(f'missing topics {set(["cp","sp","suj","js","spj","jaw"]) - set(L)}')
Pm = lambda msg: np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]) * 1000
Qm = lambda msg: [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]

def precheck(tgt_mm, qref, min_deg=0.3):
    return kin.check_pose(tgt_mm / 1000, Rot.from_quat(qref).as_matrix(), np.array(L['js'].position[:6]), kin.K.pose_mat(L['suj']), min_deg=min_deg, min_mm=5.0)
def send_cp(tgt, qref):
    msg = PoseStamped(); msg.header.frame_id = L['sp'].header.frame_id
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, tgt / 1000)
    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = qref
    L.pop('reached', None); pub.publish(msg); t = time.time()
    while 'reached' not in L and time.time() - t < 10: time.sleep(0.02)
    time.sleep(0.4); return bool(L.get('reached'))
def move_abs(cp_mm, qref):
    """sub-stepped absolute move (<= STEP mm), fixed orientation; IK precheck + stall check per sub-step (D-b)"""
    cur = Pm(L['sp']); d = np.asarray(cp_mm) - cur; k = max(1, int(np.ceil(np.linalg.norm(d) / STEP)))
    for j in range(1, k + 1):
        tgt = cur + d * j / k
        if np.linalg.norm(tgt - Pm(L['sp'])) > STEP + 1e-6: stop(f'sub-step {np.linalg.norm(tgt - Pm(L["sp"])):.2f} mm > {STEP}')
        ok, _, why = precheck(tgt, qref)
        if not ok: stop('joint-limit precheck: ' + why)
        before = Pm(L['sp'])
        if not send_cp(tgt, qref): stop(f'move not reached {tgt.round(2)}')
        adv, want = np.linalg.norm(Pm(L['sp']) - before), np.linalg.norm(tgt - before)
        if want > 0.2 and adv < 0.7 * want: stop(f'setpoint stalled ({adv:.2f}/{want:.2f} mm)')
    time.sleep(0.6); return True
def capture(k=6):
    grab['n'] = k; grab['buf'] = []; t = time.time()
    while len(grab['buf']) < k and time.time() - t < 10: time.sleep(0.02)
    b = grab['buf']; grab['buf'] = None
    if len(b) < k: stop('camera frames missing')
    return np.mean([x[0] for x in b], 0), np.mean([x[1] for x in b], 0)
def gray(x): return cv2.cvtColor(x.round().astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
def sub1(r, j):
    if 0 < j < len(r) - 1:
        p, q, s = r[j - 1], r[j], r[j + 1]; den = p - 2 * q + s; return j + (0.5 * (p - s) / den if den < 0 else 0)
    return float(j)
def match(img, T, cu, cv_, rad=120):            # as r6_calib.match (right view)
    best = None
    for s in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15):
        Ts = cv2.resize(T, None, fx=s, fy=s); th, tw = Ts.shape
        x0, y0 = int(max(cu - rad - tw / 2, 0)), int(max(cv_ - rad - th / 2, 0)); x1, y1 = int(min(cu + rad + tw / 2, 1300)), int(min(cv_ + rad + th / 2, 1024))
        r = cv2.matchTemplate(img[y0:y1, x0:x1], Ts, cv2.TM_CCOEFF_NORMED); _, mx, _, (lx, ly) = cv2.minMaxLoc(r)
        if best is None or mx > best[0]: best = (mx, x0 + sub1(r[ly], lx) + tw / 2 - 0.5, y0 + sub1(r[:, lx], ly) + th / 2 - 0.5, s)
    return best
def match_epi(img, T, u_pred, v, s, rad=40):    # D-a: epipolar band
    Ts = cv2.resize(T, None, fx=s, fy=s); th, tw = Ts.shape
    y0 = int(round(v - th / 2 - BAND)); x0 = int(max(u_pred - rad - tw / 2, 0)); x1 = int(min(u_pred + rad + tw / 2, 1300))
    r = cv2.matchTemplate(img[y0:y0 + th + 2 * BAND, x0:x1], Ts, cv2.TM_CCOEFF_NORMED); _, mx, _, (lx, ly) = cv2.minMaxLoc(r)
    return mx, x0 + sub1(r[ly], lx) + tw / 2 - 0.5, y0 + sub1(r[:, lx], ly) + th / 2 - 0.5

class Tracker:
    def __init__(s, TR, TL, d0): s.TR, s.TL, s.d = TR, TL, d0
    def track(s, name, pred_uv, out_dir, surf=None):
        Lm, Rm = capture(); d = f'{out_dir}/{name}'; os.makedirs(d, exist_ok=True)
        cv2.imwrite(d + '/right_rect.png', Rm.round().astype(np.uint8)); cv2.imwrite(d + '/left_rect.png', Lm.round().astype(np.uint8))
        mx, u, v, sc = match(gray(Rm), s.TR, *pred_uv)
        m2, uL, vL = match_epi(gray(Lm), s.TL, u - s.d, v, sc); dsp = u - uL; dv = v - vL; Z = FB / dsp
        X = np.array([(u - CX) * Z / FX, (v - CY) * Z / FX, Z]); h = surf(X) if surf else float('nan')
        rec(ev='track', name=name, uv=[u, v], uvL=[uL, vL], dv=dv, scale=sc, ncc=mx, ncc_d=m2, disp=dsp, X=X, h=h, cp=Pm(L['cp']), sp=Pm(L['sp']), js=list(L['js'].position[:6]))
        if abs(dv) > DV_MAX: stop(f'{name}: vertical disparity {dv:+.2f} px > {DV_MAX} (stereo match anomaly)')
        s.d = dsp
        return X, mx, m2, h

def surface_fn(man):
    """height of X above the local surface: flat pad plane, raised by the ridge max under the grid footprint (D-c)"""
    c, n, rmax = np.array(man['pad_plane']['centre_cam_mm']), np.array(man['pad_plane']['normal_toward_cam']), man['pad_plane']['local_surface_max_mm']
    return lambda X: float((np.asarray(X) - c) @ n) - rmax

# ================================================================ phases
def ph_lift():
    G = [json.loads(l) for l in open(PREV + '/goto_base_log.jsonl')]; J = np.array([g for g in G if g.get('ev') == 'probe_J'][-1]['J'])
    pl = [g for g in G if g.get('ev') == 'plane'][-1]; n_up = np.array(pl['n_up'])
    qref = Qm(L['sp']); dcp = np.linalg.inv(J) @ (LIFT_MM * n_up); tgt = Pm(L['sp']) + dcp
    rec(ev='lift_plan', from_sp=Pm(L['sp']), dcp=dcp, qref=qref); print(f'lift {LIFT_MM} mm along the pad normal: dcp {dcp.round(2)} mm', flush=True)
    move_abs(tgt, qref); print('lifted; margins', dict(zip(kin.NAMES, kin.margins(np.array(L['js'].position[:6])).round(1))), flush=True)

def ph_pre():
    d = ROOT + '/pre'; os.makedirs(d, exist_ok=True)
    G = [json.loads(l) for l in open(PREV + '/goto_base_log.jsonl')]; last = [g for g in G if g.get('ev') == 'track'][-1]
    # locate the goto feature in the new view with the goto template (right view only), then cut the near-tip template
    res = jload(PREV + '/goto_base_result.json'); look = cv2.imread(PREV + '/look/right_rect.png'); g0 = gray(look.astype(np.float32))
    U0, V0 = 530, 160; T_old = g0[V0 - 35:V0 + 36, U0 - 35:U0 + 36]
    Lm, Rm = capture(10); cv2.imwrite(d + '/right_rect.png', Rm.round().astype(np.uint8)); cv2.imwrite(d + '/left_rect.png', Lm.round().astype(np.uint8))
    st = {'cp': list(Pm(L['cp']) / 1000) + Qm(L['cp']), 'sp': list(Pm(L['sp']) / 1000) + Qm(L['sp']), 'js': list(L['js'].position[:6]),
          'jaw': list(L['jaw'].position), 'suj_base': kin.K.pose_mat(L['suj']).tolist(), 'date': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    jsave(d + '/state.json', st)
    mx, u, v, sc = match(gray(Rm), T_old, *res['feature_uv_now'], rad=90)
    if mx < NCC1: stop(f'cannot re-find the goto feature at the new base (NCC {mx:.2f})')
    fu, fv = int(round(u + FEAT_OFFSET_PX[0])), int(round(v + FEAT_OFFSET_PX[1]))
    r = subprocess.run(['bash', DEPLOY + '/tools/remote_fs.sh', d], cwd=DEPLOY, capture_output=True, text=True,
                       env={**os.environ, 'FS_REMOTE': 'FS_USER@FS_HOST', 'FS_RUN': 'cd ~/FoundationStereo && PYTHONPATH=~/FoundationStereo ~/miniconda3/envs/foundation_stereo/bin/python'})
    if not os.path.exists(d + '/fs_disp.npy'): stop('FS failed: ' + r.stderr[-400:])
    nd = subprocess.run([VENV, DEPLOY + '/tools/needle_detect.py', '--snap', d, '--out', d + '/clicks.json'], cwd=DEPLOY, capture_output=True, text=True)
    # 1d only needs the polyline (distance of touch points to the needle); tail/tip orientation is not used here
    if not os.path.exists(d + '/clicks.json') or len(jload(d + '/clicks.json').get('polyline_px', [])) < 5 or jload(d + '/clicks.json').get('arc_span_deg_image', 0) < 90:
        stop('needle_detect failed: ' + (nd.stdout + nd.stderr)[-400:])
    rec(ev='pre', feature_goto_uv=[u, v], ncc=mx, feature_uv=[fu, fv], half=HALF, needle=jload(d + '/clicks.json')['polyline_px'])
    print(f'pre: goto feature re-found at ({u:.0f},{v:.0f}) NCC {mx:.2f}; near-tip feature ({fu},{fv}) half {HALF}; needle polyline {len(jload(d + "/clicks.json")["polyline_px"])} pts', flush=True)

def ph_manifest():
    d = ROOT + '/pre'; st = jload(d + '/state.json'); disp = np.load(d + '/fs_disp.npy'); P = cam_points(disp)
    fu, fv = [r for r in map(json.loads, open(ROOT + '/calib_log.jsonl')) if r.get('ev') == 'pre'][-1]['feature_uv']
    Q = np.r_[P[330:520, 820:1000].reshape(-1, 3)[::5], P[350:520, 450:560].reshape(-1, 3)[::5], P[600:800, 300:700].reshape(-1, 3)[::7]]
    c, n, ni, nq = ransac_plane(Q)
    gR = gray(cv2.imread(d + '/right_rect.png').astype(np.float32)); gL = gray(cv2.imread(d + '/left_rect.png').astype(np.float32))
    TR = gR[fv - HALF:fv + HALF + 1, fu - HALF:fu + HALF + 1]
    D0 = float(np.median(disp[fv - 2:fv + 3, fu - 2:fu + 3]))
    m2, uL, vL = match_epi(gL, TR, fu - D0, fv, 1.0); D1 = fu - uL
    TL = gL[int(round(vL)) - HALF:int(round(vL)) + HALF + 1, int(round(uL)) - HALF:int(round(uL)) + HALF + 1]
    Z = FB / D1; feat = np.array([(fu - CX) * Z / FX, (fv - CY) * Z / FX, Z])
    e1 = np.cross(n, [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    # local surface max under the grid footprint: pad points within 14 mm (in-plane) of the feature's foot point
    foot = feat - ((feat - c) @ n) * n; Hh = (P.reshape(-1, 3) - c) @ n; inpl = P.reshape(-1, 3) - foot; inpl -= np.outer(inpl @ n, n)
    sel = (np.linalg.norm(inpl, axis=1) < 14.0) & (Hh > -3) & (Hh < 10)
    rmax = float(np.percentile(Hh[sel], 99)) if sel.sum() > 100 else 0.0
    rg = np.random.default_rng(20261016); pts = np.c_[rg.uniform(-8, 8, (12, 2)), rg.uniform(0, 8, 12)]; test = sorted(rg.choice(12, 3, replace=False).tolist())
    m = {'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'derived_from': 'r6_calib.py sha256 266e422f4b596897e557db3423d49bb2a7ad27022f070f0c1bc3082819286293',
         'deviations': ['D-a epipolar left match', 'D-b IK precheck + stall stop', 'D-c local-surface height gate', 'D-d near-tip feature half 20', 'D-e jaw closed for touches'],
         'ref_orientation_xyzw': st['sp'][3:], 'base_cp_m': st['sp'][:3], 'base_js': st['js'], 'suj_base': st['suj_base'],
         'feature': {'uv_phys_left': [fu, fv], 'half': HALF, 'disp_fs': D0, 'disp_epi': D1, 'ncc_lr': m2, 'cam_mm': feat.tolist(), 'template_src': 'calib3/pre/right_rect.png'},
         'pad_plane': {'normal_toward_cam': n.tolist(), 'centre_cam_mm': c.tolist(), 'inliers': [ni, nq], 'local_surface_max_mm': rmax, 'feature_height_mm': float((feat - c) @ n)},
         'pad_frame': {'e1': e1.tolist(), 'e2': e2.tolist(), 'up': n.tolist()},
         'probe_cp_mm': [[2, 0, 0], [0, 2, 0], [0, 0, 2]], 'grid_pad_mm': pts.round(2).tolist(), 'test_idx': test, 'seed': 20261016,
         'rules': {'max_step_mm': STEP, 'min_clearance_local_mm': MIN_CLEAR, 'gate_test_err_mm': GATE_TEST_MM, 'ncc': [NCC0, NCC1], 'dv_max_px': DV_MAX}}
    if abs(D1 - D0) > 3: stop(f'feature disparity: FS {D0:.1f} vs epipolar NCC {D1:.1f} px')
    if m2 < NCC0: stop(f'left/right template NCC {m2:.2f} < {NCC0}')
    jsave(MAN, m); np.save(ROOT + '/TR.npy', TR); np.save(ROOT + '/TL.npy', TL)
    print(json.dumps({'feature': m['feature'], 'pad_plane': m['pad_plane'], 'grid_pad_mm': m['grid_pad_mm'], 'test_idx': test}, indent=0), flush=True)

def load_tracker(m):
    return Tracker(np.load(ROOT + '/TR.npy'), np.load(ROOT + '/TL.npy'), m['feature']['disp_epi'])

def ph_probe():
    m = jload(MAN); base = np.array(m['base_cp_m']) * 1000; qref = m['ref_orientation_xyzw']; surf = surface_fn(m); tr = load_tracker(m)
    if np.linalg.norm(Pm(L['sp']) - base) > 1.0: stop(f'not at base ({np.linalg.norm(Pm(L["sp"]) - base):.2f} mm)')
    X0, c1, c2, h0 = tr.track('probe_base', m['feature']['uv_phys_left'], ROOT + '/pts', surf)
    print(f'base X {X0.round(2)} ncc {c1:.2f}/{c2:.2f} clearance {h0:.1f}', flush=True)
    if c1 < NCC0 or c2 < NCC0: stop(f'low NCC at base {c1:.2f}/{c2:.2f}')
    J = np.zeros((3, 3))
    for i, dv in enumerate(m['probe_cp_mm']):
        move_abs(base + dv, qref); X, c1, c2, h = tr.track(f'probe_{i}', proj(X0), ROOT + '/pts', surf); J[:, i] = (X - X0) / 2.0
        print(f'probe {i} {dv}: dX {(X - X0).round(2)} ncc {c1:.2f}/{c2:.2f} clearance {h:.1f}', flush=True)
        if c1 < NCC1 or c2 < NCC1: stop(f'low NCC probe {i}')
        move_abs(base, qref)
    sv = np.linalg.svd(J)[1]; rec(ev='probe_J', J=J, X0=X0, sv=sv); jsave(ROOT + '/probe_J.json', {'J': J, 'X0': X0})
    print('J', J.round(3).tolist(), 'sv', sv.round(3), flush=True)
    if sv.min() < 0.5 or sv.max() > 2.0: stop(f'probe Jacobian not near-rigid (sv {sv.round(3)})')

def ph_grid():
    m = jload(MAN); base = np.array(m['base_cp_m']) * 1000; qref = m['ref_orientation_xyzw']; surf = surface_fn(m); tr = load_tracker(m)
    pj = jload(ROOT + '/probe_J.json'); J = np.array(pj['J']); X0 = np.array(pj['X0']); Ji = np.linalg.inv(J)
    E = np.c_[m['pad_frame']['e1'], m['pad_frame']['e2'], m['pad_frame']['up']]
    plan = []
    for i, p in enumerate(m['grid_pad_mm']):         # pre-check every point before any motion
        dcam = E @ np.array(p); Xp = X0 + dcam; hp = surf(Xp); tgt = base + Ji @ dcam
        ok, q, why = precheck(tgt, qref, min_deg=3.0)
        plan.append((i, p, dcam, Xp, hp, tgt, ok and hp >= MIN_CLEAR, why if not ok else (f'clearance {hp:.1f} < {MIN_CLEAR}' if hp < MIN_CLEAR else '')))
        print(f'plan g{i:02d} pad {np.round(p, 1)} pred clearance {hp:5.1f} {"OK" if plan[-1][6] else "SKIP " + plan[-1][7]}', flush=True)
    rec(ev='grid_plan', plan=[(x[0], x[1], x[4], x[6], x[7]) for x in plan])
    if any(not x[6] for x in plan if x[0] in m['test_idx']): stop('a held-out point is not reachable/clear -> grid would have < 3 test points')
    for i, p, dcam, Xp, hp, tgt, ok, why in plan:
        if not ok: rec(ev='grid_skip', i=i, reason=why); continue
        move_abs(tgt, qref)
        X, c1, c2, h = tr.track(f'g{i:02d}', proj(Xp), ROOT + '/pts', surf); good = bool(c1 >= NCC1 and c2 >= NCC1)
        rec(ev='grid_point', i=i, pad=p, cp_mm=Pm(L['cp']), sp_mm=Pm(L['sp']), X=X, pred_X=Xp, ncc=[c1, c2], h=h, ok=good, test=i in m['test_idx'])
        print(f'g{i:02d} pad {np.round(p, 1)} X {X.round(2)} pred_err {np.linalg.norm(X - Xp):.2f} ncc {c1:.2f}/{c2:.2f} clearance {h:.1f}', flush=True)
        if not good: stop(f'g{i:02d}: NCC {c1:.2f}/{c2:.2f} below {NCC1}')
        if h < MIN_CLEAR - 2: stop(f'g{i:02d}: measured clearance {h:.1f} mm')
    move_abs(base, qref); X, c1, c2, h = tr.track('grid_end_base', proj(X0), ROOT + '/pts', surf)
    print(f'back at base X {X.round(2)} drift {np.linalg.norm(X - X0):.2f} mm', flush=True); rec(ev='grid_end', drift=float(np.linalg.norm(X - X0)))

def polar(A):
    U, s, Vt = np.linalg.svd(A); Rm = U @ Vt
    if np.linalg.det(Rm) < 0: U[:, -1] *= -1; Rm = U @ Vt
    return Rm, s
def ph_fit():
    m = jload(MAN); R = [json.loads(l) for l in open(ROOT + '/calib_log.jsonl')]; G = {g['i']: g for g in R if g.get('ev') == 'grid_point' and g.get('ok')}
    tr_ = [g for i, g in G.items() if i not in m['test_idx']]; te = [g for i, g in G.items() if i in m['test_idx']]
    X = np.array([np.r_[g['cp_mm'], 1] for g in tr_]); Y = np.array([g['X'] for g in tr_]); M = np.linalg.lstsq(X, Y, rcond=None)[0]
    res = lambda gs: [float(np.linalg.norm(np.r_[g['cp_mm'], 1] @ M - g['X'])) for g in gs]
    A = M[:3].T; Rp, sv = polar(A); s_iso = float(sv.mean())
    # rigid (+ isotropic scale) Procrustes on the same training points, for comparison
    P = np.array([g['cp_mm'] for g in tr_]); pc, yc = P.mean(0), Y.mean(0); H = (P - pc).T @ (Y - yc); U, S, Vt = np.linalg.svd(H)
    Rr = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T; sr = float(S.sum() / ((P - pc) ** 2).sum()); tr_t = yc - sr * Rr @ pc
    rig = lambda gs: [float(np.linalg.norm(sr * Rr @ np.array(g['cp_mm']) + tr_t - np.array(g['X']))) for g in gs]
    out = {'n_train': len(tr_), 'n_test': len(te), 'M': M.tolist(), 'A': A.tolist(), 'singular_values': sv.tolist(),
           'polar_R': Rp.tolist(), 'polar_iso_scale': s_iso, 'A_minus_sR_fro': float(np.linalg.norm(A - s_iso * Rp)),
           'A_minus_sR_max_abs': float(np.abs(A - s_iso * Rp).max()),
           'polar_vs_procrustes_deg': float(np.degrees(Rot.from_matrix(Rp.T @ Rr).magnitude())),
           'train_rms_mm': float(np.sqrt(np.mean(np.square(res(tr_))))), 'train_err_mm': dict(zip([g['i'] for g in tr_], res(tr_))),
           'test_err_mm': dict(zip([g['i'] for g in te], res(te))),
           'rigid': {'R': Rr.tolist(), 's': sr, 't': tr_t.tolist(), 'train_rms_mm': float(np.sqrt(np.mean(np.square(rig(tr_))))), 'test_err_mm': dict(zip([g['i'] for g in te], rig(te)))},
           'gate_pass': bool(len(te) == 3 and max(res(te)) <= GATE_TEST_MM), 'convention': 'X_cam(feature ~ tip) = [cp_mm, 1] @ M at the reference orientation (R6)'}
    jsave(ROOT + '/calibration_today.json', out)
    print(json.dumps({k: v for k, v in out.items() if k not in ('M', 'A', 'polar_R', 'rigid', 'train_err_mm')}, indent=1), flush=True)
    print('rigid+scale: train rms %.2f, test %s' % (out['rigid']['train_rms_mm'], {k: round(v, 2) for k, v in out['rigid']['test_err_mm'].items()}), flush=True)
    if not out['gate_pass']: stop(f'held-out gate failed: {out["test_err_mm"]}')

def ph_pad4():
    m = jload(MAN); cal = jload(ROOT + '/calibration_today.json'); M = np.array(cal['M']); A3 = M[:3]; b3 = M[3]
    base = np.array(m['base_cp_m']) * 1000; qref = m['ref_orientation_xyzw']; Rref = Rot.from_quat(qref).as_matrix()
    c, n = np.array(m['pad_plane']['centre_cam_mm']), np.array(m['pad_plane']['normal_toward_cam']); e1, e2 = np.array(m['pad_frame']['e1']), np.array(m['pad_frame']['e2'])
    P = cam_points(np.load(ROOT + '/pre/fs_disp.npy')); poly = np.array(jload(ROOT + '/pre/clicks.json')['polyline_px'], float)
    def ray_plane(u, v):
        r = np.array([(u - CX) / FX, (v - CY) / FX, 1.0]); return r * ((c @ n) / (r @ n))
    N3 = np.array([ray_plane(u, v) for u, v in poly]); Nd = np.r_[[N3[0]], [N3[i] + (N3[i + 1] - N3[i]) * t for i in range(len(N3) - 1) for t in np.linspace(0, 1, 10)]]
    cen = N3.mean(0); dist_needle = lambda X: float(np.min(np.linalg.norm(Nd - X, axis=1)))
    cam2cp = lambda Xc: np.linalg.solve(A3.T, Xc - b3)                    # cp (hinge, mm) whose tip-feature sits at Xc (ref orientation)
    zc = Rref[:, 2]; zc_cam = A3.T @ zc; down = -float(zc_cam @ n) > 0
    if not down: zc = -zc; zc_cam = -zc_cam
    cosang = -float(zc_cam @ n) / np.linalg.norm(zc_cam)
    rec(ev='pad4_geom', needle_centre_cam=cen, axis_cam=zc_cam, cos_axis_normal=cosang)
    def flat_ok(X):
        u, v = proj(X); u, v = int(round(u)), int(round(v))
        if not (20 < u < 1280 and 20 < v < 1004): return False, 'outside image'
        Hh = (P[v - 6:v + 7, u - 6:u + 7].reshape(-1, 3) - c) @ n
        return (bool(np.abs(Hh).max() < 1.0), f'local height {Hh.min():+.1f}..{Hh.max():+.1f} mm')
    # candidates: square corners around the needle centre in the pad plane
    cands = []
    for side in PAD_SIDE_MM:
        for sa, sb in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            X = cen + side * (sa * e1 + sb * e2); X = X - ((X - c) @ n) * n
            okf, why = flat_ok(X); dn = dist_needle(X)
            hover_feat = X + HOVER_TIP_MM * n; cp_h = cam2cp(hover_feat); cp_hh = cam2cp(X + 20.0 * n)
            ok_ik, q, whyk = precheck(cp_h, qref, min_deg=3.0)
            cands.append({'side': side, 'X': X, 'd_needle': dn, 'flat': okf, 'flat_why': why, 'ik': ok_ik, 'ik_why': whyk, 'cp_hover': cp_h, 'cp_hover_high': cp_hh})
    good = [k for k in cands if k['flat'] and k['ik'] and k['d_needle'] >= NEEDLE_PICK_MM]
    pick = []
    for k in sorted(good, key=lambda k: (k['side'] != PAD_SIDE_MM[0], k['side'])):
        if all(np.linalg.norm(k['X'] - p['X']) > 8.0 for p in pick): pick.append(k)
        if len(pick) == 4: break
    for k in cands: print(f"cand side {k['side']:4.1f} needle {k['d_needle']:5.1f} mm flat {k['flat']} ({k['flat_why']}) ik {k['ik']} {k['ik_why']}", flush=True)
    rec(ev='pad4_candidates', cands=cands, picked=[p['X'] for p in pick])
    if len(pick) < 4: stop(f'only {len(pick)} touch points pass flatness / reach / needle clearance')
    # close the jaw (D-e)
    msg = JointState(); msg.position = [0.0]; pjaw.publish(msg); time.sleep(1.8); rec(ev='jaw_closed', jaw=list(L['jaw'].position))
    def eff(dt):
        E, Pe = [], []; t0_ = time.time()
        while time.time() - t0_ < dt: E.append(list(L['js'].effort)[:6]); Pe.append(np.array(L['js'].position[:6]) - np.array(L['spj'].position[:6])); time.sleep(0.01)
        return np.mean(E, 0), np.mean(Pe, 0)
    touches = []
    for j, k in enumerate(pick):
        move_abs(k['cp_hover_high'], qref); move_abs(k['cp_hover'], qref)
        tip_feat_now = np.r_[Pm(L['sp']), 1] @ M; h_now = float((tip_feat_now - c) @ n)
        max_mm = h_now / max(cosang, 0.3) + 3.0; nsteps = int(np.ceil(max_mm / TOUCH_STEP))
        e0, _ = eff(1.0); rows = []; contact = None
        print(f'touch {j}: pad point needle-dist {k["d_needle"]:.1f} mm, predicted tip height {h_now:.1f} mm, max descent {max_mm:.1f} mm', flush=True)
        for s in range(nsteps):
            tgt = Pm(L['sp']) + TOUCH_STEP * zc
            ok, _, why = precheck(tgt, qref)
            if not ok: stop('touch precheck: ' + why)
            before = Pm(L['sp']); send_cp(tgt, qref); time.sleep(0.4)
            e, pe = eff(0.3); de = e - e0
            hit = {'ins_effort': abs(de[2]) > 0.15, 'j01_effort': max(abs(de[0]), abs(de[1])) > 0.05, 'ins_track_mm': abs(pe[2]) * 1000 > 0.3}
            rows.append({'step': s + 1, 'effort': e, 'd_effort': de, 'track_err': pe, 'hit': hit, 'sp': Pm(L['sp'])})
            print(f'  step {s + 1:2d} (+{TOUCH_STEP * (s + 1):4.1f} mm): d_ins {de[2]:+.3f} d_j0 {de[0]:+.3f} d_j1 {de[1]:+.3f} ins_err {pe[2] * 1000:+.2f} mm {[x for x, v in hit.items() if v]}', flush=True)
            if any(hit.values()): contact = s + 1; break
        q = np.array(L['js'].position[:6]); T = kin.fk_ecm(q, kin.K.pose_mat(L['suj'])); tip_ecm = T[:3, 3] * 1000 + T[:3, :3] @ TCP
        tip_cam = np.r_[Pm(L['cp']), 1] @ M + ((T[:3, :3] - Rref) @ TCP) @ A3; dn = dist_needle(tip_cam)
        touches.append({'i': j, 'pad_X_cam': k['X'], 'contact_step': contact, 'baseline_effort': e0, 'rows': rows, 'q': q, 'tip_ecm_mm': tip_ecm,
                        'cp_mm': Pm(L['cp']), 'tip_cam_pred': tip_cam, 'needle_dist_mm': dn})
        rec(ev='touch', **touches[-1])
        # retract back up the jaw axis to the hover point, then to 20 mm, before any check that may stop
        move_abs(k['cp_hover'], qref); move_abs(k['cp_hover_high'], qref)
        if contact is None: stop(f'touch {j}: no contact within {max_mm:.1f} mm (predicted height + 3 mm)')
        if dn < NEEDLE_CLEAR_MM: stop(f'touch {j}: contact point {dn:.1f} mm from the needle polyline (< {NEEDLE_CLEAR_MM})')
        print(f'  contact at step {contact}; tip ECM {tip_ecm.round(2)} mm; needle distance {dn:.1f} mm', flush=True)
    move_abs(np.array(jload(MAN)['base_cp_m']) * 1000, qref)
    msg = JointState(); msg.position = [float(np.deg2rad(60.0))]; pjaw.publish(msg); time.sleep(1.8)
    jsave(ROOT + '/pad4_touches.json', touches)

def ph_plane():
    m = jload(MAN); cal = jload(ROOT + '/calibration_today.json'); T = jload(ROOT + '/pad4_touches.json')
    Ptip = np.array([t['tip_ecm_mm'] for t in T]); c = Ptip.mean(0); U, S, Vt = np.linalg.svd(Ptip - c); n = Vt[2]
    Rp = np.array(cal['polar_R']); A3 = np.array(cal['M'])[:3]; b3 = np.array(cal['M'])[3]
    cam_origin_ecm = np.linalg.solve(A3.T, -b3)                       # ECM point mapped to the camera origin
    if n @ (cam_origin_ecm - c) < 0: n = -n
    res = (Ptip - c) @ n
    n_fs_cam = np.array(m['pad_plane']['normal_toward_cam']); c_fs_cam = np.array(m['pad_plane']['centre_cam_mm'])
    n_fs_ecm = Rp.T @ n_fs_cam; n_fs_ecm /= np.linalg.norm(n_fs_ecm); Rref = Rot.from_quat(m['ref_orientation_xyzw']).as_matrix(); c_fs_ecm = np.linalg.solve(A3.T, c_fs_cam - b3) + Rref @ TCP   # tip-equivalent ECM point (feature ~ TCP, R6)
    out = {'tip_points_ecm_mm': Ptip.tolist(), 'centre_ecm_mm': c.tolist(), 'normal_toward_cam_ecm': n.tolist(), 'residuals_mm': res.tolist(),
           'rms_mm': float(np.sqrt(np.mean(res ** 2))), 'fs_normal_in_ecm': n_fs_ecm.tolist(),
           'angle_vs_fs_deg': float(np.degrees(np.arccos(np.clip(abs(n @ n_fs_ecm), -1, 1)))),
           'fs_plane_offset_along_n_mm': float((c_fs_ecm - c) @ n),
           'safety_rule': 'signed distance of tip and hinge to this plane (towards camera +) >= 3 mm empty-handed'}
    jsave(ROOT + '/pad_plane_ecm.json', out); print(json.dumps(out, indent=1), flush=True)

PH = {'lift': ph_lift, 'pre': ph_pre, 'manifest': ph_manifest, 'probe': ph_probe, 'grid': ph_grid, 'fit': ph_fit, 'pad4': ph_pad4, 'plane': ph_plane}
for p in (['lift', 'pre', 'manifest', 'probe', 'grid', 'fit', 'pad4', 'plane'] if a.phase == 'all' else [a.phase]):
    print(f'===== {p}', flush=True); a.phase = p; PH[p](); rec(ev='phase_done', phase=p)
sys.stdout.flush(); os._exit(0)
