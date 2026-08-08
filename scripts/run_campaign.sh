#!/usr/bin/env bash
# End-to-end collection campaign, in the order it must be run.
#
# This is a checklist you drive, not a fire-and-forget script: it stops at each stage
# that needs a human decision (attaching or removing the tool, confirming a dry run).
#
#   ./scripts/run_campaign.sh 172.16.0.2
#
# Prerequisites, both enforced by the tools rather than assumed here:
#   * config/workspace.yaml holds MEASURED wall planes (fpi fit-plane), and
#   * config/tool.yaml holds the tool's scale mass and bounding box.
set -euo pipefail

IP="${1:-172.16.0.2}"
DATA="${FPI_DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data}"
RAW="$DATA/raw"
RESULTS="$DATA/results"
IMAGE="${FPI_ROBOT_IMAGE:-fpi-robot:0.9.2}"
mkdir -p "$RAW" "$RESULTS"

robot() {
  docker run --rm -it --network=host --cap-add=SYS_NICE \
    --ulimit rtprio=99 --ulimit rttime=-1 --ulimit memlock=-1 \
    -v "$DATA:/data" "$IMAGE" "$@"
}

pause() { echo; read -rp ">>> $1  [Enter to continue, Ctrl-C to abort] "; echo; }

echo "=== 0. communication quality ==="
robot fpi_check --ip "$IP" --seconds 10 --out /data/raw/check
echo "=== 0b. frame-convention gate ==="
echo "    (requires the Desk end-effector transform to be identity)"
fpi verify-fk --log "$RAW/check"

echo "=== 1. design ==="
fpi poses generate --out assets/static_poses.csv
fpi poses export   --out "$RAW/poses.csv"
fpi traj check
fpi traj export    --out "$RAW/excite.csv"
fpi traj view      --out "$RESULTS/trajectory.png"
pause "Review $RESULTS/trajectory.png. Does the motion stay clear of both walls?"

pause "ATTACH the tool. Then warm the robot up for 15-20 minutes before continuing."

echo "=== 2. dry run at 20% amplitude ==="
robot fpi_run_trajectory --ip "$IP" --traj /data/raw/excite.csv \
      --out /data/raw/dryrun --loaded --dry-run 0.2
pause "Dry run complete. Motion looked safe?"

# Interleaving matters: alternating which configuration goes first in each block makes
# the two runs' mean collection times equal, so linear thermal drift cancels exactly.
# See docs/THEORY.md section 4.1.
echo "=== 3. static sweep, block A (tool attached) ==="
robot fpi_static_poses --ip "$IP" --poses /data/raw/poses.csv \
      --out /data/raw/static_loaded --loaded

pause "REMOVE the tool."
echo "=== 3b. static sweep, block B (bare) ==="
robot fpi_static_poses --ip "$IP" --poses /data/raw/poses.csv \
      --out /data/raw/static_bare --bare

echo "=== 4. dynamic run, bare first this time (alternating order) ==="
robot fpi_run_trajectory --ip "$IP" --traj /data/raw/excite.csv \
      --out /data/raw/dyn_bare --bare

pause "ATTACH the tool again."
robot fpi_run_trajectory --ip "$IP" --traj /data/raw/excite.csv \
      --out /data/raw/dyn_loaded --loaded

echo "=== 5. validation trajectory (held out) ==="
robot fpi_run_trajectory --ip "$IP" --traj /data/raw/excite.csv \
      --out /data/raw/val_loaded --loaded
pause "REMOVE the tool."
robot fpi_run_trajectory --ip "$IP" --traj /data/raw/excite.csv \
      --out /data/raw/val_bare --bare

echo "=== 6. identification ==="
fpi ident run \
  --static-loaded  "$RAW/static_loaded"  --static-bare  "$RAW/static_bare" \
  --dynamic-loaded "$RAW/dyn_loaded"     --dynamic-bare "$RAW/dyn_bare" \
  --validation-loaded "$RAW/val_loaded"  --validation-bare "$RAW/val_bare" \
  --out "$RESULTS"

echo
echo "Done. Read $RESULTS/report.md before entering anything into Desk."
echo "Then verify: |tau_ext_hat_filtered| < 0.5 Nm at rest, and drift-free hand guiding."
