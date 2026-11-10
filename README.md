# Fully automatic reach -> grasp -> place on the dVRK (sim-to-real suturing policy)

Code of the automation pipeline used on the dVRK (PSM1 + ECM stereo endoscope), state of 2026-11-10.
One command runs a whole trial with no operator clicks:

```
cd full_run && python3 trial_auto.py            # needle lying on the pad, in view
python3 trial_auto.py --hold                    # keep holding the needle at the end
python3 trial_auto.py --resume <run dir>        # continue placing with a needle that is already held
```

Stages: rig check -> background snapshot -> RL Reach + hover + descend + close + lift (S1 grasp code) ->
held-needle detection -> grip fit + best reachable placing pose -> rotations about the needle tip
(the tip is detected again after every step) -> tip to 15 / 8 / 3.5 mm above the entry sticker ->
machine verdict -> release + back to the base.

## Layout
| path | content |
|---|---|
| `full_run/trial_auto.py` | the automatic trial (detectors instead of clicks, all stop conditions) |
| `full_run/run.py` | the individual stages (`grasp`, `snap`, `shift`, `needle`, `rot`, `down`, `metric`, `release`) |
| `full_run/trial.py` | the earlier operator-click version of the same sequence |
| `full_run/tools/held_needle.py` | detector for the needle held in the jaw (FoundationStereo disparity vs. background) |
| `full_run/tools/grasp_rl_roll.sh`, `grasp_hold_roll.py` | grasp wrapper with the roll model of the current calibration |
| `full_run/tools/measure_site.py` | entry (blue sticker) / exit measurement |
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
* Status 2026-11-10: reach and grasp run automatically and hold the needle. Placing reached 8.9 mm above the entry
  with 0.9 mm lateral offset in the best automatic run (A179); the final move to 3.5 mm has not completed yet.
  In the current phantom placement the placing pose sits at the wrist joint limits.
