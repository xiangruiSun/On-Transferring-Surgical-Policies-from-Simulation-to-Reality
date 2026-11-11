#!/bin/bash
# final demo v2 (19:30): ONE fully automatic trial with the endoscope view recorded.  The video is ALWAYS kept (named by the outcome); nothing is released automatically.
S1=~/rl_live/sim2real_20261106/S1; source $S1/trials/env.sh; SP=$(dirname $0); OUT=~/Desktop/final_demo_20261110; cd ~/rl_live/full_run_1109
(cd $S1/G_place; for i in 1 2 3; do timeout 40 python3 check_rig.py 2>&1 | tail -1 | grep -q "OK - ready" && break; sleep 2; done)
T=$(date +%H%M%S); V=$OUT/endoscope_left_$T.mp4; python3 $SP/rec_video.py $V > $SP/rec_$T.log 2>&1 & RP=$!; sleep 6
timeout 1500 python3 trial_auto.py > $SP/demo_run_$T.log 2>&1; RC=$?
sleep 3; kill -TERM $RP; wait $RP; cat $SP/rec_$T.log; tail -3 $SP/demo_run_$T.log | cut -c1-300
R=$([ $RC -eq 0 ] && echo success || echo stopped); mv $V ${V%.mp4}_$R.mp4; cp $SP/demo_run_$T.log $OUT/trial_log_${T}_$R.txt; echo "video kept: ${V%.mp4}_$R.mp4 (trial exit $RC)"
