"""translate PSM1 by a CAMERA-frame vector (mm) using an affine A3 (rough), fixed orientation; sub-steps <= 2 mm; refuses > 15 mm."""
import sys, json, time, threading, os, numpy as np, rclpy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
v_cam = np.array([float(x) for x in sys.argv[1:4]]); cal = sys.argv[4]; assert np.linalg.norm(v_cam) <= 15
A3 = np.array(json.load(open(cal))["M"])[:3]; d_ecm = v_cam @ np.linalg.inv(A3) * 1e-3
rclpy.init(); n = rclpy.create_node('nudge_cam'); L = {}
n.create_subscription(PoseStamped, '/PSM1/setpoint_cp', lambda m: L.__setitem__('sp', m), 10)
n.create_subscription(Bool, '/PSM1/goal_reached', lambda m: L.__setitem__('r', m.data), 10)
pub = n.create_publisher(PoseStamped, '/PSM1/move_cp', 1); threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()
t0 = time.time()
while 'sp' not in L and time.time() - t0 < 5: time.sleep(0.05)
s = L['sp']; p0 = np.array([s.pose.position.x, s.pose.position.y, s.pose.position.z]); k = max(1, int(np.ceil(np.linalg.norm(d_ecm) * 1e3 / 2.0)))
print('ECM delta mm', np.round(d_ecm * 1e3, 2).tolist(), 'steps', k, flush=True)
for j in range(1, k + 1):
    m = PoseStamped(); m.header.frame_id = s.header.frame_id; p = p0 + d_ecm * j / k
    m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, p); m.pose.orientation = s.pose.orientation
    L.pop('r', None); pub.publish(m); t = time.time()
    while 'r' not in L and time.time() - t < 10: time.sleep(0.02)
    time.sleep(0.4)
    if not L.get('r'): print('not reached at step', j); os._exit(1)
print('done', flush=True); os._exit(0)
