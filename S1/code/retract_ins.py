"""retract the PSM1 insertion joint by --mm (move_jp, other joints unchanged): always away from the tissue along the shaft."""
import sys, time, threading, numpy as np, rclpy
from sensor_msgs.msg import JointState
mm = float(sys.argv[1]); assert 0 < mm <= 25
rclpy.init(); n = rclpy.create_node('retract_ins'); L = {}
n.create_subscription(JointState, '/PSM1/measured_js', lambda m: L.__setitem__('js', m), 10)
n.create_subscription(JointState, '/PSM1/setpoint_js', lambda m: L.__setitem__('sp', m), 10)
pub = n.create_publisher(JointState, '/PSM1/move_jp', 1); threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()
t0 = time.time()
while ('sp' not in L or 'js' not in L) and time.time() - t0 < 5: time.sleep(0.05)
q = np.array(L['sp'].position[:6]); q2 = q.copy(); q2[2] -= mm * 1e-3
print('insertion', round(q[2], 4), '->', round(q2[2], 4), flush=True)
m = JointState(); m.position = q2.tolist(); pub.publish(m); time.sleep(3.0)
print('measured insertion', round(L['js'].position[2], 4), flush=True); import os; os._exit(0)
