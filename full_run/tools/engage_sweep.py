# emulate the dVRK tool-engage sweep (tool_engage_position lower/upper of the tool file) with the arm where it is; jaw stays closed
import time, sys, rclpy, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
rclpy.init(); n = Node('engage_sweep'); st = {}
n.create_subscription(JointState, '/PSM1/measured_js', lambda m: st.__setitem__('js', m), 10)
n.create_subscription(JointState, '/PSM1/setpoint_js', lambda m: st.__setitem__('sp', m), 10)
pub = n.create_publisher(JointState, '/PSM1/move_jp', 10)
def spin(s):
    t = time.time()
    while time.time() - t < s: rclpy.spin_once(n, timeout_sec=0.02)
spin(4); q0 = np.array(st['sp'].position[:6]); print('start', np.round(q0, 4))
def go(w):
    q = q0.copy(); q[3:] = w
    for k in range(4):
        m = JointState(); m.position = q.tolist(); m.header.stamp = n.get_clock().now().to_msg(); pub.publish(m)
        t = time.time(); peak = np.zeros(3)
        while time.time() - t < 14:
            rclpy.spin_once(n, timeout_sec=0.02); peak = np.maximum(peak, np.abs(st['js'].effort[3:6]))
            if np.abs(np.array(st['js'].position[3:6]) - w).max() < 0.01 and time.time() - t > 1: break
        ok = np.abs(np.array(st['js'].position[3:6]) - w).max() < 0.02
        print('target', w, 'meas', np.round(st['js'].position[3:6], 3), 'peak |eff|', np.round(peak, 3), 'ok' if ok else 'RESEND', flush=True)
        if ok: return
    sys.exit('not reached')
lo, up = [-4.5, -0.1745, -0.1745], [4.5, 0.1745, 0.1745]
for w in ([0, 0, 0], lo, up, lo, up, [0, 0, 0], [1.555, -0.076, 0.125]): go(w)
