#!/usr/bin/env bash
# FoundationStereo on the GPU workstation for one or more R6 snapshot dirs (batch mode).
#
# The dVRK host cannot run FoundationStereo.  This copies each snapshot's
# rectified pair to the workstation, runs r6/r6_fs_infer.py there, and copies
# the disparity back as <snap>/fs_disp.npy (+ fs_meta.json, fs_host.json) --
# exactly what r6_perceive.py reads.  It replaces R6's r6_fs_remote.sh, which
# was wired to a laptop over a reverse tunnel.
#
#   tools/remote_fs.sh <trial>/snap0 <trial>/snap1
#
# Configure once (defaults in brackets):
#   FS_REMOTE      ssh target                          [FS_USER@FS_HOST]
#   FS_REMOTE_DIR  scratch dir on the workstation      [r6_fs_xfer]  (relative to its $HOME)
#   FS_RUN         how to start python in the FS env, run on the workstation; the
#                  script path and its arguments are appended.  Examples:
#                    "cd ~/FoundationStereo && PYTHONPATH=~/FoundationStereo ~/miniconda3/envs/foundation_stereo/bin/python"
#                    "docker exec -w /workspace/FoundationStereo foundationstereo env PYTHONPATH=. python"
#                  [cd ~/FoundationStereo && PYTHONPATH=~/FoundationStereo python3]
#   FS_CKPT        checkpoint path as FS_RUN sees it    [~/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth]
#
# Image convention (R6): files use the TOPIC names; physical-left = topic RIGHT,
# so FS "left" gets right_rect.png and FS "right" gets left_rect.png.
set -euo pipefail

REMOTE=${FS_REMOTE:-FS_USER@FS_HOST}
RDIR=${FS_REMOTE_DIR:-r6_fs_xfer}
RUN=${FS_RUN:-"cd ~/FoundationStereo && PYTHONPATH=~/FoundationStereo python3"}
CKPT=${FS_CKPT:-"~/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"}
HERE=$(cd "$(dirname "$0")/.." && pwd)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=8"

[ $# -ge 1 ] || { echo "usage: $0 <snapshot_dir> [...]"; exit 2; }
$SSH "$REMOTE" "mkdir -p $RDIR/scripts"
$SCP "$HERE/r6/r6_fs_infer.py" "$REMOTE:$RDIR/scripts/r6_fs_infer.py"

for D in "$@"; do
  [ -f "$D/left_rect.png" ] && [ -f "$D/right_rect.png" ] || { echo "$D: no left_rect.png/right_rect.png"; exit 2; }
  TAG="$(basename "$(dirname "$(readlink -f "$D")")")_$(basename "$D")_$(date +%H%M%S)"
  rm -f "$D/fs_disp.npy" "$D/fs_meta.json" "$D/fs_host.json"   # never reuse an old result
  $SSH "$REMOTE" "rm -rf $RDIR/$TAG && mkdir -p $RDIR/$TAG/in/left $RDIR/$TAG/in/right"
  $SCP "$D/right_rect.png" "$REMOTE:$RDIR/$TAG/in/left/f.png"
  $SCP "$D/left_rect.png" "$REMOTE:$RDIR/$TAG/in/right/f.png"
  echo "[$TAG] FoundationStereo on $REMOTE ..."
  T0=$(date +%s)
  $SSH "$REMOTE" "$RUN \$HOME/$RDIR/scripts/r6_fs_infer.py --in_dir \$HOME/$RDIR/$TAG/in --out_dir \$HOME/$RDIR/$TAG/out --ckpt $CKPT" \
      | grep -E '"stem"|Error|error' || true
  $SCP "$REMOTE:$RDIR/$TAG/out/disp/f.npy" "$D/fs_disp.npy"
  $SCP "$REMOTE:$RDIR/$TAG/out/meta.json" "$D/fs_meta.json"
  printf '{"host": "%s", "tag": "%s", "fetched": "%s", "wall_s": %d}\n' \
      "$REMOTE" "$TAG" "$(date -Iseconds)" "$(( $(date +%s) - T0 ))" > "$D/fs_host.json"
  echo "[$TAG] -> $D/fs_disp.npy"
done
