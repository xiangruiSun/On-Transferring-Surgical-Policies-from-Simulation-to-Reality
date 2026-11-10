"""per-trial timing from the frozen runner's own logs (start.json, trial_log.jsonl phase times, FS fs_host/fs_meta). usage: timing.py <trial number>... ; appends to trials/timings.csv"""
import json, sys, os, csv
R = os.path.expanduser('~/rl_live/sim2real_20261106/S1')
COLS = ['trial', 'attempt', 'total_s', 'perception_s', 'fs_wall_s', 'fs_infer_s', 'perception_other_s', 'approach_s', 'wrist_s', 'template_s', 'hover_s', 'descend_s', 'close_lift_check_s', 'hold_s', 'status']
out = R + '/trials/timings.csv'; new = not os.path.exists(out); done = set()
if not new: done = {r['trial'] for r in csv.DictReader(open(out))}
w = csv.writer(open(out, 'a')); new and w.writerow(COLS)
for n in map(int, sys.argv[1:]):
    T, A = f'T{n:02d}', f'A{100 + n}_Prep'; d = f'{R}/F_attempts/{A}'
    if T in done: print(T, 'already in timings.csv'); continue
    t0 = json.load(open(d + '/start.json'))['t']; P = {}
    for l in open(d + '/trial_log.jsonl'):
        x = json.loads(l)
        if x.get('ev') == 'phase': P.setdefault(x['name'], x['t'])
    fs = json.load(open(d + '/snap0/fs_host.json'))['wall_s'] if os.path.exists(d + '/snap0/fs_host.json') else None
    inf = sum(f['infer_s'] for f in json.load(open(d + '/snap0/fs_meta.json'))['frames']) if os.path.exists(d + '/snap0/fs_meta.json') else None
    seq = ['perception', 'approach', 'wrist', 'template', 'hover', 'descend', 'close_lift', 'hold']
    dur = {}; prev = t0
    for k in seq:
        if k in P: dur[k] = round(P[k] - prev, 1); prev = P[k]
    st = json.load(open(d + '/trial_state.json')).get('status')
    tot = round(max(P.values()) - t0, 1) if P else None
    row = [T, A, tot, dur.get('perception'), fs, inf, round(dur['perception'] - fs, 1) if fs and 'perception' in dur else None] + [dur.get(k) for k in seq[1:]] + [st]
    w.writerow(row); print(dict(zip(COLS, row)))
