#!/bin/bash
# one r6_calib phase with the read-only watchdog alongside (motion phases); output -> ../phase_<name>.log
set -o pipefail
P=$1; shift; EXTRA="$*"; R=~/rl_live/sim2real_20261106/S1/C_r6cal; H=$R/code
source /opt/ros/jazzy/setup.bash; source ~/ros2_ws/install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=SYSTEM_DEFAULT FASTDDS_BUILTIN_TRANSPORTS=UDPv4
rm -f $R/WATCHDOG_TRIP $R/WATCHDOG_ARMED
if [ "$P" == "probe" ] || [ "$P" == "grid" ]; then
  # r6_calib waits until the watchdog is armed (exec keeps the PID the watchdog watches)
  bash -c "for i in \$(seq 1 150); do [ -f $R/WATCHDOG_ARMED ] && exec python3 $H/r6_calib.py --phase $P $EXTRA; sleep 0.2; done; echo 'watchdog never armed: nothing sent'; exit 9" > $R/phase_$P.log 2>&1 &
  CP=$!
  python3 ~/rl_live/sim2real_20261106/S1/code/watchdog.py --pid $CP --log $R/watchdog_$P.jsonl > $R/watchdog_$P.out 2>&1 &
  WP=$!
  wait $CP; RC=$?
  sleep 1; kill -INT $WP 2>/dev/null; wait $WP 2>/dev/null
  cat $R/phase_$P.log; echo "--- watchdog:"; cat $R/watchdog_$P.out
  [ -f $R/WATCHDOG_TRIP ] && { echo "WATCHDOG TRIPPED: $(cat $R/WATCHDOG_TRIP)"; exit 3; }
  exit $RC
else
  python3 $H/r6_calib.py --phase $P 2>&1 | tee $R/phase_$P.log
fi
