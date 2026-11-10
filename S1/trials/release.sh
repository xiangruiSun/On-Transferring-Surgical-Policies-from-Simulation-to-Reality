#!/bin/bash
# open the jaw where the arm is (needle drops), then go back to the calibration base; then rig check.  usage: release.sh
source ~/rl_live/sim2real_20261106/S1/trials/env.sh; cd $R/G_place
timeout 150 python3 release_and_home.py 2>&1 | tail -3
timeout 40 python3 check_rig.py 2>&1 | tail -1
