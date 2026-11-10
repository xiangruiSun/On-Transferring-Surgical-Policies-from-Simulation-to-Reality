"""set the PSM1 jaw angle (rad) with one /PSM1/jaw/move_jp; refuses outside [-0.25, 0.6]; prints measured before/after."""
import sys, time, threading, numpy as np, rclpy
from sensor_msgs.msg import JointState
tgt = float(sys.argv[1]); assert -0.25 <= tgt <= 0.6, "refused"
rclpy.init(); n = rclpy.create_node('jaw_set'); L = {}
n.create_subscription(JointState, '/PSM1/jaw/measured_js', lambda m: L.__setitem__('jaw', m), 10)
pub = n.create_publisher(JointState, '/PSM1/jaw/move_jp', 1); threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()
t0 = time.time()
while 'jaw' not in L and time.time() - t0 < 5: time.sleep(0.05)
print('before', np.round(L['jaw'].position, 4), 'effort', np.round(L['jaw'].effort, 3), flush=True)
time.sleep(1.0); m = JointState(); m.position = [tgt]; pub.publish(m); time.sleep(2.5)
print('after', np.round(L['jaw'].position, 4), 'effort', np.round(L['jaw'].effort, 3), flush=True)
import os; os._exit(0)
