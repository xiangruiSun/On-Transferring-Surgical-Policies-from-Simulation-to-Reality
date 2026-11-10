#!/usr/bin/env python3
"""Safety retract: move PSM1 back along its own jaw axis (-tool z) by N mm in 3 mm move_cp steps, fixed orientation; log effort."""
import os, sys, time, json, threading, numpy as np, rclpy
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from scipy.spatial.transform import Rotation as Rot
N = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
rclpy.init(); n = rclpy.create_node('retract_1d'); L = {}
n.create_subscription(PoseStamped, '/PSM1/setpoint_cp', lambda m: L.__setitem__('sp', m), qos_profile_sensor_data)
n.create_subscription(JointState, '/PSM1/measured_js', lambda m: L.__setitem__('js', m), qos_profile_sensor_data)
n.create_subscription(Bool, '/PSM1/goal_reached', lambda m: L.__setitem__('r', m.data), 10)
pub = n.create_publisher(PoseStamped, '/PSM1/move_cp', 1); threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()
t = time.time()
while not all(k in L for k in ('sp', 'js')) and time.time() - t < 8: time.sleep(0.05)
sp = L['sp']; p0 = np.array([sp.pose.position.x, sp.pose.position.y, sp.pose.position.z]); q = [sp.pose.orientation.x, sp.pose.orientation.y, sp.pose.orientation.z, sp.pose.orientation.w]
z = Rot.from_quat(q).as_matrix()[:, 2]; log = []
print('start effort', np.round(L['js'].effort[:6], 3).tolist(), 'insertion', round(L['js'].position[2], 4), flush=True)
k = int(np.ceil(N / 3.0))
for j in range(1, k + 1):
    tg = p0 - z * (N * j / k) / 1000; m = PoseStamped(); m.header.frame_id = sp.header.frame_id
    m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, tg); m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
    L.pop('r', None); pub.publish(m); t = time.time()
    while 'r' not in L and time.time() - t < 10: time.sleep(0.02)
    time.sleep(0.5); e = list(L['js'].effort[:6]); log.append({'mm': N * j / k, 'effort': e, 'ins': L['js'].position[2]})
    print(f'retract {N * j / k:4.1f} mm: insertion {L["js"].position[2]:.4f} effort ins {e[2]:+.3f} j0 {e[0]:+.3f} j1 {e[1]:+.3f}', flush=True)
json.dump(log, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calib3', 'retract_log.json'), 'w'), indent=1)
sys.stdout.flush(); os._exit(0)
