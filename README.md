# Fully automatic reach -> grasp -> place on the dVRK (sim-to-real suturing policy)

Code of the automation pipeline used on the dVRK (PSM1 + ECM stereo endoscope), state of 2026-11-10.
One command runs a whole trial with no operator clicks:

```
cd full_run && python3 trial_auto.py            # needle lying on the pad, in view; a grasped needle is kept in the jaw at the end
python3 trial_auto.py --release                 # open the jaw + back to the base at the end / at a stop
python3 trial_auto.py --resume <run dir>        # continue placing with a needle that is already held
```

Stages: rig check -> background snapshot -> RL Reach + hover + descend + close + lift (S1 grasp code) ->
held-needle detection -> grip fit + best reachable placing pose -> rotations about the needle tip
(the tip is detected again after every step) -> stepwise approach when the tip is more than 18 mm beside the entry ->
tip to 15 / 8 / 3.5 mm above the entry sticker -> machine verdict. The needle stays in the jaw unless `--release` is given.

## Layout
| path | content |
|---|---|
| `full_run/trial_auto.py` | the automatic trial (detectors instead of clicks, all stop conditions) |
| `full_run/run.py` | the individual stages (`grasp`, `snap`, `shift`, `needle`, `rot`, `down`, `metric`, `release`) |
| `full_run/trial.py` | the earlier operator-click version of the same sequence |
| `full_run/tools/held_needle.py` | detector for the needle held in the jaw (FoundationStereo disparity vs. background) |
| `full_run/tools/grasp_rl_roll.sh`, `grasp_hold_roll.py` | grasp wrapper with the roll model of the current calibration |
| `full_run/tools/measure_site.py` | entry (blue sticker) / exit measurement |
| `full_run/tools/rot_tip_tol.py`, `phase2_tol.py` | copies of the S1 rotation / descent moves with a 2.5 deg settle tolerance for the wrist (a clamped needle holds the wrist yaw ~1 deg off its setpoint); selected by `PLACE_TOL=1` |
| `full_run/tools/engage_sweep.py` | engages the instrument disks by a wrist sweep when dVRK skipped the engage (tool tip outside the cannula at start-up) |
| `full_run/tools/rec_video.py`, `demo_driver.sh` | endoscope video recorder and the recorded-demo wrapper |
| `S1/A_check/collect_roll.py`, `fit_roll_1110pm3.py` | roll-model data collection and fit, to be redone after every calibration |
| `S1/C_r6cal/r6_calib.py`, `run_phase.sh`, `scene_inputs.json` | calibration phases (manifest / probe / grid / fit) and the scene inputs of the last calibration |
| `full_run/setting/` | example setting of 2026-11-10 (pad plane, site, roll model) |
| `S1/code`, `S1/G_place`, `S1/P_place`, `S1/trials` | the robot code the stages call (calibration, guarded motion, grasp attempt, placing moves) |
| `S1/C_r6cal/calibration_tcp.json`, `S1/D_pad/pad.json` | example hand-eye calibration and pad plane (2026-11-10) |
| `step1d_calib/` | kinematics + calibration helpers imported by the code above |
| `deploy_tools/remote_fs.sh` | runs FoundationStereo on the GPU machine |

## Notes
* Paths are those of the lab machine: the code expects `~/rl_live/full_run_1109` (= `full_run/`),
  `~/rl_live/sim2real_20261106/S1` (= `S1/`) and `~/rl_live/sim2real_20261025/step1d_calib`; ROS 2 Jazzy + dVRK;
  FoundationStereo on a second machine: set `FS_REMOTE` (placeholder `FS_USER@FS_HOST`) in `S1/trials/env.sh`.
* Recorded data (images, disparities, attempts) is not part of this repository.
* Status 2026-11-10 evening: first fully automatic success `T02` (grasp, 65 deg rotation in two steps, stepwise approach,
  descent; final tip 2.3 mm above the entry, 0.6 mm lateral, machine verdict). The camera / needle layout must keep the needle tip
  visible at the placing pose (instrument not between camera and needle); calibration, pad plane, site and roll model are
  redone after every ECM / setup-joint move (recipe in `full_run/README.md`).
* Known limits: the held-needle detector needs the jaw inside the image (otherwise the tip predicted by the move is used, flagged
  in the log); at ~65 mm camera distance the pixel thresholds of the grasp code do not fit (use ~90-100 mm).
