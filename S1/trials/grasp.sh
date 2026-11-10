#!/bin/bash
# one grasp trial with the frozen code (no RL).  usage: grasp.sh <trial number>   e.g. grasp.sh 3  -> trials/T03, F_attempts/A103_Prep
# arm must be at base with an empty jaw (run release.sh first if it holds a needle).  Ctrl-C stops it (no further command is sent).
set -e; N=$1; [ -n "$N" ] || { echo "usage: grasp.sh <trial number>"; exit 2; }
source ~/rl_live/sim2real_20261106/S1/trials/env.sh
T=$(printf T%02d $N); A=$((100+N)); [ -e $R/F_attempts/A${A}_Prep ] && { echo "A${A}_Prep exists: pick another number"; exit 2; }
mkdir -p $R/trials/$T; cd $R/G_place; echo "SKIP_SHIFT_STOP=${SKIP_SHIFT_STOP:-0}" | tee $R/trials/$T/flags.txt
python3 ../code/r6_local/r6_snap.py $R/trials/$T/top >/dev/null 2>&1 && cp $R/trials/$T/top/right_rect.png $R/trials/$T/${T}_top.png; date +%T > $R/trials/$T/start_time.txt
RIG=$(timeout 40 python3 check_rig.py 2>&1 | tail -1); echo "rig: $RIG"
case "$RIG" in *"OK - ready"*) ;; *) echo "STOP: rig check failed (run release.sh?)"; exit 4;; esac
set +e; timeout 600 python3 grasp_hold_norl.py --n $A 2>&1 | tee $R/F_attempts/A${A}_run.log | grep -E --line-buffered "^\[[0-9]|QUALITY|ABORT|GUARD" | cut -c1-150
ln -sfn $R/F_attempts/A${A}_Prep $R/trials/$T/data
python3 - $R/F_attempts/A${A}_Prep <<'PY'
import json,sys
s=json.load(open(sys.argv[1]+'/trial_state.json'));cl=s.get('close_lift',{});gc=cl.get('grasp_check') or {}
print('== status',s.get('status'),'| abort',str(s.get('abort_reason'))[:150],'| contact 1mm-step',s.get('descend_contact_step_1mm'),'| touch contact',(s.get('descend') or {}).get('contact_step'))
print('== jaw after close',cl.get('jaw_after_close'),'| grasp_check',gc.get('grasped'),'| hold',gc.get('jaw_hold'))
PY
python3 $R/trials/timing.py $N | tail -1
echo "frames: $R/F_attempts/A${A}_Prep/{pre_close,closed,lifted_2s}/right_rect.png"
