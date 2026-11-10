#!/bin/bash
# consecutive attempts on the same needle; stops at the first hardware-safety abort (status aborted).
# The first attempt gets --user-ok (user asked to continue after the previous stop).
set -o pipefail
cd ~/rl_live/sim2real_20261106/S1/code
source /opt/ros/jazzy/setup.bash; source ~/ros2_ws/install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=SYSTEM_DEFAULT FASTDDS_BUILTIN_TRANSPORTS=UDPv4 S1028_TTY=/dev/pts/3 DISPLAY=:1 XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority
OK="--user-ok"
for n in "$@"; do
  echo "=== A$n $(date +%T)"
  python3 session_driver.py attempt --n $n $OK --where "${WHERE:-same needle as A03 (not re-placed), s=0.20, contact ins 0.10}" > ../F_attempts/A$(printf %02d $n)_run.log 2>&1
  OK=""
  tail -1 ../F_attempts/ledger.jsonl | python3 -c "import sys,json; r=json.loads(sys.stdin.read()); print(r['n'], r['status'], r['last_phase'], r.get('abort_reason'), 'contact', r.get('contact_step'))"
  st=$(tail -1 ../F_attempts/ledger.jsonl | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['status'])")
  [ "$st" == "aborted" ] && { echo "hardware abort -> stop series"; break; }
  [ "$st" == "completed" ] && [ "${STOP_ON_COMPLETED:-1}" == "1" ] && { echo "completed -> stop series (user judges from images)"; break; }
done
