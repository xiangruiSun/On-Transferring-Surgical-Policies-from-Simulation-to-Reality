# Reach -> grasp -> place, one trial (2026-11-09 setting)

## Run a trial (operator, one command)
```
cd ~/rl_live/full_run_1109 && python3 trial.py
```
`trial.py` runs the T01 sequence (the first success) and asks for clicks in pop-up windows
(left click = point, u = undo, ENTER = done, f = back to the full image, ESC = abort; the arm then stays where it is):
1. needle TAIL (end where the thread leaves) -> reach + grasp (RL Reach, frozen S1 code)
2. needle points TAIL (at the jaw) -> TIP, >= 6 points -> grip fit + placing plan
3. before each move: needle TIP in the right image, then the same tip in the left image (yellow line = same row);
   for the rotations also a needle point ~2 mm behind the tip
4. two rotations about the tip, then the tip to 15 / 8 / 3.5 mm above the sticker entry
5. final tip clicks -> height / lateral offset; answer y/n (success -> trials/T<nn>), then optional release + back to base
Stops on its own if the rig is not ready (ECM/setup joints moved -> recalibrate), the grasp fails, or any guard trips.
Calibration: S1/C_r6cal of 2026-11-10 11:11 (new setup: ECM rotated ~90 deg, PSM1 SUJ moved; test err 0.36/0.05/0.25 mm).
Setting (pad, sticker): setting/ (2026-11-10 11:12; exit = 13.7 mm along image -y, i.e. up, across the slit). 11-09 evening: setting_1109b/, S1/C_r6cal_1109b.

## New setup (ECM / SUJ / phantom moved), ~15 min
1. base pose: jaw 0, wrist roll 1.555 / pitch -0.076 / yaw 0.125 (code/set_joint.py), code/cp_step.py until the jaw tip is ~9 mm above the pad
2. archive S1/C_r6cal + D_pad, new C_r6cal: pre3 snapshot + FS, scene_inputs.json (feat = pin = jaw pivot screw, pad box),
   run_phase.sh manifest / probe / (probe_J -> polar rotation) / grid / fit, session_driver.py convert
3. pad: code/pad_from_snap_1109.py C_r6cal/pre3 --write; if the normal maps to camera +Z, raise base_T 7 mm (11-08 fix) and cp_step the arm; check_rig.py
4. setting/: base_snap (+FS), pad.json = S1/D_pad/pad.json, `tools/measure_site.py setting/base_snap setting/site 13.7 <ux,uy>` (exit direction in the image)
If only the phantom moved: steps 3 (pad only) and 4.


Entry point: `run.py` (stages in its docstring). Outputs: `trials/<TRIAL>/` (log.txt has every command + output).
Robot code = frozen S1 code (`~/rl_live/sim2real_20261106/S1`), called unchanged; all its guards apply.

## Stage-by-stage (run.py), and the earlier setting (setting_old/, 19:03, before the ECM/SUJ move)
- `setting/base_snap/`: base stereo snapshot + FoundationStereo disparity.
- `setting/pad.json`: pad plane from that snapshot (rms 0.27 mm, base tip 5.5 mm above the pad, normal -> camera -Z);
  also written to `S1/D_pad/pad.json` (old one kept as `pad_superseded_190317.json`).
- `setting/site/measure.json` + `measure.png`: entry = blue sticker centre (uv 580,232; 85.6 mm from the camera);
  exit = virtual point 13.7 mm along image +x on the pad plane (across the slit). Made by `tools/measure_site.py`.
- Calibration: `S1/C_r6cal` (11-09, rig check OK: ECM/SUJ unchanged). Redo setting/ if the phantom moves.

## One trial (T = trial number, e.g. 63)
```
export TRIAL=T63
python3 run.py grasp 63                      # RL Reach -> grasp -> lift (needle placed by hand, flat, in view)
python3 run.py snap s0 --fs                  # read the needle tail->tip points on trials/T63/s0_grid_R.jpg
python3 run.py shift s1 0 -8 0               # only if the needle tip is out of view
python3 run.py needle s0 u,v u,v ...         # grip fit + best reachable placing pose -> rotation needed
python3 run.py rot r1 30 uR,vR uB,vB Z       # repeat (<= 30 deg each) until the planned rotation is done
python3 run.py down rc1 15 uR,vR uL,vL       # re-centre 15 mm above the entry
python3 run.py down c8 8 uR,vR uL,vL
python3 run.py down c35 3.5 uR,vR uL,vL      # final
python3 run.py metric uR,vR uL,vL            # tip read on the final snapshot -> height / lateral
python3 run.py release                       # after the verdict
```
Tip pixels (right uR,vR; left uL,vL; a needle point 'body' near the tip) are read by eye on the gridded views
written after every move. Each moving stage runs dry first and sends only if the dry run passes.
