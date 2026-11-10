# source this: environment for the frozen grasp runner (2026-11-06 S1)
source /opt/ros/jazzy/setup.bash; source ~/ros2_ws/install/setup.bash
export VERTICAL_DESCENT=1 ROS_AUTOMATIC_DISCOVERY_RANGE=SYSTEM_DEFAULT FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export DISPLAY=:1 XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority FS_REMOTE=FS_USER@FS_HOST
export FS_RUN="cd ~/FoundationStereo && PYTHONPATH=~/FoundationStereo ~/miniconda3/envs/foundation_stereo/bin/python"
R=~/rl_live/sim2real_20261106/S1
