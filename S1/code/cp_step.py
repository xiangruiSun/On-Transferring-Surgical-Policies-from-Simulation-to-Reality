"""2026-11-05 S1 helper: translate PSM1 by an ECM-frame vector (mm), fixed orientation, sub-steps <= 0.5 mm; refuses > 10 mm.
Guard: stops (holds last reached step) if insertion-joint effort rises by more than INS_MAX (pushing in) or falls by more than INS_NEG_MAX, or a step is not reached."""
import sys, time, threading, os, json, numpy as np, rclpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
d = np.array([float(x) for x in sys.argv[1:4]]) * 1e-3; assert np.linalg.norm(d) <= 0.010, "refused > 10 mm"
INS_MAX, INS_NEG_MAX = 0.35, 1.5   # + = pushing in (contact); - = pulling out (insertion stiction seen 11-05, harmless)
rclpy.init(); n = rclpy.create_node('cp_step'); L = {}
n.create_subscription(PoseStamped, '/PSM1/setpoint_cp', lambda m: L.__setitem__('sp', m), 10)
n.create_subscription(JointState, '/PSM1/measured_js', lambda m: L.__setitem__('js', m), 10)
n.create_subscription(Bool, '/PSM1/goal_reached', lambda m: L.__setitem__('r', m.data), 10)
pub = n.create_publisher(PoseStamped, '/PSM1/move_cp', 1); threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()
t0 = time.time()
while ('sp' not in L or 'js' not in L) and time.time() - t0 < 5: time.sleep(0.05)
time.sleep(3.0); s = L["sp"]; p0 = np.array([s.pose.position.x, s.pose.position.y, s.pose.position.z]); e0 = L['js'].effort[2]
k = max(3, int(np.ceil(np.linalg.norm(d) * 1e3 / 0.5))); print('ECM delta mm', np.round(d * 1e3, 2).tolist(), 'steps', k, 'ins eff0 %.3f' % e0, flush=True)
for j in range(1, k + 1):
    m = PoseStamped(); m.header.frame_id = s.header.frame_id; p = p0 + d * j / k
    m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, p); m.pose.orientation = s.pose.orientation
    L.pop('r', None); pub.publish(m); t = time.time()
    while 'r' not in L and time.time() - t < 8: time.sleep(0.02)
    time.sleep(0.3); e = L['js'].effort[2]
    if j == 2: eb = e                      # baseline after friction direction has settled (insertion stiction flips sign on reversal)
    de = e - (eb if j > 2 else e0); lim = INS_MAX if j > 2 else 0.8
    if not L.get('r'): print('STOP: not reached at step', j, flush=True); os._exit(1)
    if de > lim or de < -INS_NEG_MAX: print('STOP: insertion effort change %.3f at step %d' % (de, j), flush=True); os._exit(2)
print('done; ins eff change %.3f' % (L['js'].effort[2] - e0), flush=True); os._exit(0)
