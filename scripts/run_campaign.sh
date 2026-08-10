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

# Declare the tool to the robot on LOADED runs. Read from config/tool.yaml so there is
# one source of truth. Without this the robot carries an unmodelled payload: gravity
# compensation is wrong, the arm sags into the impedance error, tracking differs from the
# bare run, and the controller can abort with tau_J_range_violation.
LOAD_MASS=$(python3 -c "import yaml;d=yaml.safe_load(open('config/tool.yaml'));print(d['mass_scale'] or 0)")
LOAD_COM=$(python3 -c "
import yaml
d = yaml.safe_load(open('config/tool.yaml'))
bb = d['bounding_box']
print(','.join(str(0.5*(a+b)) for a, b in zip(bb['min'], bb['max'])))")
echo "declaring tool to the robot on loaded runs: ${LOAD_MASS} kg at [${LOAD_COM}] m"
LOADED_ARGS="--loaded --load-mass ${LOAD_MASS} --load-com ${LOAD_COM}"

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
      --out /data/raw/dryrun $LOADED_ARGS --dry-run 0.2
pause "Dry run complete. Motion looked safe?"

# -----------------------------------------------------------------------------------
# The dynamic stage is collected in four blocks ordered L B B L (ABBA).
#
# Thermal drift is a function of wall-clock time, so the difference of torques inherits
# k * (mean loaded time - mean bare time). ABBA makes that difference exactly zero --
# and it stays zero even though the tool swaps take minutes, because the two swaps sit
# symmetrically about the middle. `L B L B` does NOT cancel: it leaves the bare blocks
# one slot later on average, i.e. a constant offset on every sample. Neither does
# `L L B B`, which is worst of all.
#
# Note only two tool swaps are needed, not three: the middle B B pair is contiguous.
# See docs/THEORY.md section 4.1.
#
# The static stage is collected as a single pair. Its signal (the ~0.4 Nm gravity
# torque) is roughly fifty times the drift residual, so the ordering does not matter
# there; it matters enormously for the dynamic stage, whose inertia signature is
# ~0.008 Nm.
# -----------------------------------------------------------------------------------

echo "=== 3. dynamic block 1 of 4: LOADED ==="
robot fpi_run_trajectory --ip "$IP" --traj /data/raw/excite.csv \
      --out /data/raw/dyn_loaded_1 $LOADED_ARGS

pause "REMOVE the tool.  (swap 1 of 2)"

echo "=== 4. dynamic block 2 of 4: BARE ==="
robot fpi_run_trajectory --ip "$IP" --traj /data/raw/excite.csv \
      --out /data/raw/dyn_bare_1 --bare

echo "=== 5. dynamic block 3 of 4: BARE (no swap -- this is the B B pair) ==="
robot fpi_run_trajectory --ip "$IP" --traj /data/raw/excite.csv \
      --out /data/raw/dyn_bare_2 --bare

echo "=== 5b. static sweep, bare (tool is already off) ==="
robot fpi_static_poses --ip "$IP" --poses /data/raw/poses.csv \
      --out /data/raw/static_bare --bare

pause "ATTACH the tool.  (swap 2 of 2)"

echo "=== 6. dynamic block 4 of 4: LOADED ==="
robot fpi_run_trajectory --ip "$IP" --traj /data/raw/excite.csv \
      --out /data/raw/dyn_loaded_2 $LOADED_ARGS

echo "=== 6b. static sweep, loaded ==="
robot fpi_static_poses --ip "$IP" --poses /data/raw/poses.csv \
      --out /data/raw/static_loaded $LOADED_ARGS

echo "=== 7. identification ==="
# The two blocks per configuration are passed as a comma-separated list; the pipeline
# concatenates them and drops the settling period from EACH block, which is what keeps
# the ABBA balance intact.
fpi ident run \
  --static-loaded  "$RAW/static_loaded"  --static-bare  "$RAW/static_bare" \
  --dynamic-loaded "$RAW/dyn_loaded_1,$RAW/dyn_loaded_2" \
  --dynamic-bare   "$RAW/dyn_bare_1,$RAW/dyn_bare_2" \
  --out "$RESULTS"

echo
echo "Done. Read $RESULTS/report.md before entering anything into Desk."
echo "Then verify: |tau_ext_hat_filtered| < 0.5 Nm at rest, and drift-free hand guiding."
