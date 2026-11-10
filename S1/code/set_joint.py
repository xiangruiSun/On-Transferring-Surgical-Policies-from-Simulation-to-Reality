"""set ONE PSM1 joint (index, value rad/m) by move_jp, the others = current setpoint; refuses |change| > 2.0 rad / 20 mm."""
import sys, time, threading, os, numpy as np, rclpy
from sensor_msgs.msg import JointState
i, val = int(sys.argv[1]), float(sys.argv[2])
rclpy.init(); n = rclpy.create_node('set_joint'); L = {}
n.create_subscription(JointState, '/PSM1/setpoint_js', lambda m: L.__setitem__('sp', m), 10)
n.create_subscription(JointState, '/PSM1/measured_js', lambda m: L.__setitem__('js', m), 10)
pub = n.create_publisher(JointState, '/PSM1/move_jp', 1); threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()
t0 = time.time()
while ('sp' not in L or 'js' not in L) and time.time() - t0 < 5: time.sleep(0.05)
q = np.array(L['sp'].position[:6]); dq = val - q[i]; assert abs(dq) <= (0.02 if i == 2 else 2.0), f"refused {dq}"
q[i] = val; m = JointState(); m.position = q.tolist(); pub.publish(m); time.sleep(2.0 + 2.0 * abs(dq))
print('joint', i, 'measured', round(L['js'].position[i], 4), flush=True); os._exit(0)
