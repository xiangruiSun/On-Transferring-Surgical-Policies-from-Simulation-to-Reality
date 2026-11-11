#!/usr/bin/env python3
"""2026-11-06: collect roll-model data with today's calibration (not main code).
Sequence: lift LIFT_MM along the pad normal -> straighten wrist (q5=q6=0) about the tip (like Attempt.wrist) ->
for each roll offset: rotate about the tip (<= 6 deg / 3 mm per step), open jaw to 0.6 rad (as attempt open25), save 10 frames + state,
close jaw (ramp) -> back to roll 0 -> back to the calibration base pose.  Without --send: only IK / margin / pad checks of every target."""
import os, sys, json, time, argparse
if 'AMENT_PREFIX_PATH' not in os.environ:
    import shlex
    os.execvp('bash', ['bash', '-c', 'source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash && export ROS_AUTOMATIC_DISCOVERY_RANGE=SYSTEM_DEFAULT FASTDDS_BUILTIN_TRANSPORTS=UDPv4 && exec python3 '
                       + ' '.join(shlex.quote(x) for x in [os.path.abspath(__file__)] + sys.argv[1:])])
import numpy as np, cv2
ROOT = os.path.expanduser('~/rl_live/sim2real_20261106/S1'); sys.path.insert(0, ROOT + '/code')
from rig import RigIO, Mover, Guard, load_kin, T_of, tip_of, TCP_MM
from attempt import rodrigues, R6_WRIST_PITCH, R6_WRIST_ROLL, OPEN_RAD
ap = argparse.ArgumentParser(); ap.add_argument('--send', action='store_true'); ap.add_argument('--rolls', type=float, nargs='+', default=[-75, -50, -25, 0, 25, 50])
ap.add_argument('--lift', type=float, default=12.0); ap.add_argument('--out', default=ROOT + '/A_check/roll_data'); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); LOG = open(a.out + '/log.jsonl', 'a')
def log(r): r['t'] = time.time(); LOG.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n'); LOG.flush()
cal = json.load(open(ROOT + '/C_r6cal/calibration_tcp.json')); pad = json.load(open(ROOT + '/D_pad/pad.json'))
n_pad = np.array(pad['normal']); Tbase = np.array(cal['base_T'])
kin = load_kin(); io = RigIO(node_name='collect_roll'); t0 = time.time()
while time.time() - t0 < 6 and any(io.get(k) is None for k in ('cp', 'sp', 'suj', 'js', 'spj', 'jaw', 'ecm')): time.sleep(0.1)
mv = Mover(io, kin, log); mv.pad = (n_pad, float(pad['d']))
S0 = mv.setpoint(); tip0 = mv.tip(S0)
print('start: tip pad height %.1f mm, q %s' % (mv.pad_dist_mm(tip0), np.round(io.get('js')[1][:6], 3).tolist()), flush=True)

def check(T, name):
    q = mv.check_target(T); print(f'  ok {name}: tip pad height {mv.pad_dist_mm(mv.tip(T)):.1f} mm, hinge {mv.pad_dist_mm(T[:3, 3]):.1f} mm, q {np.round(q, 3).tolist() if q is not None else None}', flush=True)

# planned targets (dry-run uses the same math as rotate_about_tip with the kinematic tip)
T_lift = S0.copy(); T_lift[:3, 3] += n_pad * a.lift * 1e-3
q = np.array(io.get('js')[1][:6]); qs = q.copy(); qs[4] = qs[5] = 0.0
Rs = kin.fk_ecm(qs, io.get('suj')[1])[:3, :3]; tipL = tip_of(T_lift)
plan = [('lift', T_lift)]
plan.append(('straight', T_of(tipL - Rs @ (TCP_MM * 1e-3), Rs)))
for r in a.rolls:
    Rg = rodrigues(Rs[:, 2], np.radians(r)) @ Rs; plan.append((f'roll{r:+.0f}', T_of(tipL - Rg @ (TCP_MM * 1e-3), Rg)))
plan.append(('base', S0))
for nm, T in plan:
    try: check(T, nm)
    except Guard as e: print(f'  GUARD {nm}: {e}', flush=True); sys.exit(2)
if not a.send: print('dry-run only; nothing sent'); io.shutdown(); os._exit(0)

def snap(name):
    d = f'{a.out}/{name}'; os.makedirs(d, exist_ok=True); b = io.frames(10)
    cv2.imwrite(d + '/right_rect.png', np.mean([x[1] for x in b], 0).round().astype(np.uint8)); cv2.imwrite(d + '/left_rect.png', np.mean([x[0] for x in b], 0).round().astype(np.uint8))
    st = {k: (None if io.get(k) is None else np.asarray(io.get(k)[1]).tolist()) for k in ('cp', 'sp', 'js', 'jse', 'jaw', 'jaw_eff', 'ecm')}; json.dump(st, open(d + '/state.json', 'w'), indent=1)
try:
    mv.move_to(T_lift, tag='lift', lim=(3.0, 4.0))
    for it in range(2):
        q = np.array(io.get('js')[1][:6]); qs = q.copy(); qs[4] = qs[5] = 0.0
        mv.rotate_about_tip(kin.fk_ecm(qs, io.get('suj')[1])[:3, :3], tag=f'straighten{it}', lim=R6_WRIST_PITCH)
    Rs = mv.setpoint()[:3, :3].copy(); json.dump(dict(Rs=Rs.tolist()), open(a.out + '/straight.json', 'w'))
    for r in a.rolls:
        mv.rotate_about_tip(rodrigues(Rs[:, 2], np.radians(r)) @ Rs, tag=f'roll{r:+.0f}', lim=R6_WRIST_ROLL)
        mv.jaw_to(OPEN_RAD, n_steps=1, dt=1.8, tag='open'); snap(f'roll{r:+.0f}'); mv.jaw_to(0.0, n_steps=6, dt=0.3, tag='close')
        print(f'roll {r:+.0f}: saved, q {np.round(io.get("js")[1][:6], 3).tolist()}', flush=True)
    mv.rotate_about_tip(Rs, tag='roll0', lim=R6_WRIST_ROLL)
    mv.move_to(S0, tag='base', lim=(3.0, 4.0)); print('back at base, %d commands' % mv.n_cmd, flush=True)
except Guard as e:
    print('GUARD (stopped, holding):', e, flush=True); log(dict(ev='guard', why=str(e)))
io.shutdown(); os._exit(0)
