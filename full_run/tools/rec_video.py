"""read-only: record /jhu_daVinci/left/image_raw (endoscope view) to an mp4 through ffmpeg until SIGTERM / SIGINT."""
import sys, signal, subprocess, time, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
out = sys.argv[1]; topic = sys.argv[2] if len(sys.argv) > 2 else '/jhu_daVinci/left/image_raw'
rclpy.init(); n = Node('demo_video_rec'); st = dict(p=None, k=0, t0=None, run=True)
def cb(m):
    if st['p'] is None:
        pix = 'rgb24' if m.encoding.startswith('rgb') else 'bgr24'
        st['p'] = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', pix, '-s', f'{m.width}x{m.height}', '-use_wallclock_as_timestamps', '1', '-i', '-', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-fps_mode', 'cfr', '-r', '30', out], stdin=subprocess.PIPE)
        st['t0'] = time.time()
    try: st['p'].stdin.write(bytes(m.data)); st['k'] += 1
    except Exception as e: print('write failed', e, flush=True); st['run'] = False
n.create_subscription(Image, topic, cb, qos_profile_sensor_data)
signal.signal(signal.SIGTERM, lambda *a: st.__setitem__('run', False)); signal.signal(signal.SIGINT, lambda *a: st.__setitem__('run', False))
while st['run']: rclpy.spin_once(n, timeout_sec=0.1)
if st['p']: st['p'].stdin.close(); st['p'].wait()
print('frames', st['k'], 'seconds %.1f' % (time.time() - (st['t0'] or time.time())), flush=True)
