#!/bin/bash
# 2026-11-04 user "放开发送（放针，连测 3 次）": release -> attempt, three times. Stops on any hardware abort or rig check failure. Holds after the last grasp.
R=~/rl_live/sim2real_20261106/S1; F=$R/F_attempts
source /opt/ros/jazzy/setup.bash; source ~/ros2_ws/install/setup.bash
export VERTICAL_DESCENT=1 ROS_AUTOMATIC_DISCOVERY_RANGE=SYSTEM_DEFAULT FASTDDS_BUILTIN_TRANSPORTS=UDPv4 DISPLAY=:1 XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority FS_REMOTE=FS_USER@FS_HOST
export FS_RUN="cd ~/FoundationStereo && PYTHONPATH=~/FoundationStereo ~/miniconda3/envs/foundation_stereo/bin/python"
cd $R/G_place
for n in "$@"; do
  echo "=== $(date +%T) before A$n"
  J=$(timeout 15 python3 ./state.py 2>/dev/null | grep PSM1_jaw | tr -dc '0-9.\-')
  B=$(timeout 40 python3 check_rig.py 2>&1 | grep "arm to calibration base" | grep -oE "[0-9.]+ mm" | cut -d' ' -f1)
  echo "jaw $J, to base ${B:-unknown} mm"
  if [ -z "$B" ] || awk "BEGIN{exit !($B>1.0)}"; then timeout 150 python3 release_and_home.py 2>&1 | tail -3 | cut -c1-160; fi
  R1=$(timeout 40 python3 check_rig.py 2>&1 | tail -1); echo "rig: $R1"
  case "$R1" in *"OK - ready"*) ;; *) echo "STOP: rig check failed"; exit 4;; esac
  timeout 600 python3 grasp_hold_norl.py --n $n > $F/A$(printf %02d $n)_run.log 2>&1; echo "A$n rc $?"
  python3 - "$F" "$n" <<'PY' || { echo "STOP: hardware class abort"; exit 3; }
import json,sys
L=[json.loads(l) for l in open(sys.argv[1]+'/ledger.jsonl')];r=[x for x in L if x.get('n')==int(sys.argv[2])]
r=r[-1] if r else {};print('ledger',{k:(str(r.get(k))[:130]) for k in('n','status','abort_class','abort_reason','last_phase','n_commands')})
sys.exit(0 if r.get('status')=='completed' or r.get('abort_class')=='quality' else 1)
PY
done
echo "=== done $(date +%T)"
