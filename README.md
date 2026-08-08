# franka_payload_id

Identify the inertial parameters of a tool bolted to a Franka Emika Panda's flange —
mass, centre of mass and inertia tensor — so they can be entered into Desk's
**End Effector** settings.

Franka ships no tool for this: Desk only lets you type the numbers in, and getting them
wrong degrades gravity compensation, collision detection, external-wrench estimation and
impedance control.

The method is **difference of torques**: run the same trajectory twice, once with the
tool and once without, and subtract. The arm's entire dynamics cancel, along with
torque-sensor bias and most friction, leaving only the payload. The result therefore does
not depend on the accuracy of any Panda dynamic model — which matters, because Franka has
never published one.

See **[docs/THEORY.md](docs/THEORY.md)** for the derivations and a formula-to-code index.

---

## Layout

```
config/     limits, robot, workspace, tool, experiment  (all tunable, all documented)
assets/     Panda URDF + meshes, reference excitation trajectory
cpp/        libfranka 0.9.2 collector: fpi_check, fpi_move_to,
            fpi_static_poses, fpi_run_trajectory
python/     model, trajectory design, preprocessing, estimators, reporting
docs/       THEORY.md
docker/     Dockerfile.analysis (anywhere) and Dockerfile.robot (robot PC)
```

Two independent Docker images by design. `analysis` has no libfranka and needs no robot,
so essentially all development and every test runs on a laptop. `robot` is the small,
fragile one that talks to the FCI.

---

## Requirements

| | |
|---|---|
| Robot | Franka Emika Panda (FER). **Not FR3** — see below. |
| libfranka | **0.9.2**, the last release that talks to a Panda; ≥ 0.10 is FR3-only |
| Robot system version | ≥ 4.2.1 (Desk → Settings → System) |
| Robot PC | Ubuntu 20.04 with a **PREEMPT_RT** kernel |
| Dev machine | anything with Docker or Python ≥ 3.10 |

The PREEMPT_RT kernel must be on the **host**: containers share the host kernel, so there
is no such thing as a real-time image. Without it, `robot.control()` aborts with
`communication_constraints_violation`. Read-only streaming (`fpi_check`) works fine
without it.

---

## Quick start — no robot needed

```bash
docker compose -f docker/docker-compose.yml build analysis
docker compose -f docker/docker-compose.yml run --rm analysis pytest -q

# whole pipeline on generated data, with known ground truth
docker compose -f docker/docker-compose.yml run --rm analysis fpi ident synthetic
```

Or natively:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/pytest -q
.venv/bin/fpi ident synthetic
```

> If you have ROS on `PYTHONPATH`, its pytest plugins leak into the venv. Use
> `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`, or just use the
> container.

---

## Before touching the robot

**Measure the walls.** `config/workspace.yaml` ships with deliberately conservative
*placeholders*, and every export refuses until they are replaced. Jog the flange to touch
each wall at three well-separated points, record the positions, then:

```bash
fpi fit-plane --points wall_behind.csv --name wall_behind
```

and paste the result in with `measured: true`.

**Fill in `config/tool.yaml`**: the tool's mass from a scale (the strongest validation
available, and it sharpens the centre of mass) and its bounding box in the flange frame
(used for the workspace check and as the inertia prior).

---

## Running a campaign

```bash
# 0. verify the link. Do not proceed if this fails.
fpi_check --ip 172.16.0.2 --seconds 10 --out /data/raw/check

# 1. frame-convention gate: does the URDF describe this robot?
#    (set the Desk end-effector transform to identity first)
fpi verify-fk --log data/raw/check

# 2. design and export
fpi poses generate --out assets/static_poses.csv
fpi poses export   --out data/raw/poses.csv
fpi traj generate  --out assets/excitation.json
fpi traj check     --traj assets/excitation.json
fpi traj view      --out data/results/trajectory.png     # eyeball it
fpi traj export    --traj assets/excitation.json --out data/raw/excite.csv

# 3. dry run at 20 % amplitude, tool attached
fpi_run_trajectory --ip 172.16.0.2 --traj /data/raw/excite.csv \
                   --out /data/raw/dryrun --loaded --dry-run 0.2

# 4. collect. Warm up 15-20 min first, and INTERLEAVE the runs (see below).
#    ...or just use scripts/run_campaign.sh, which sequences the ABBA blocks for you.
fpi_run_trajectory --ip 172.16.0.2 --traj /data/raw/excite.csv --out /data/raw/dyn_loaded_1 --loaded
#    >>> remove the tool <<<
fpi_run_trajectory --ip 172.16.0.2 --traj /data/raw/excite.csv --out /data/raw/dyn_bare_1   --bare
fpi_run_trajectory --ip 172.16.0.2 --traj /data/raw/excite.csv --out /data/raw/dyn_bare_2   --bare
fpi_static_poses   --ip 172.16.0.2 --poses /data/raw/poses.csv --out /data/raw/static_bare  --bare
#    >>> attach the tool <<<
fpi_run_trajectory --ip 172.16.0.2 --traj /data/raw/excite.csv --out /data/raw/dyn_loaded_2 --loaded
fpi_static_poses   --ip 172.16.0.2 --poses /data/raw/poses.csv --out /data/raw/static_loaded --loaded

# 5. identify
fpi ident run --static-loaded  data/raw/static_loaded --static-bare data/raw/static_bare \
              --dynamic-loaded data/raw/dyn_loaded_1,data/raw/dyn_loaded_2 \
              --dynamic-bare   data/raw/dyn_bare_1,data/raw/dyn_bare_2 \
              --out data/results
```

Run the robot binaries inside the container:

```bash
docker run --rm -it --network=host --cap-add=SYS_NICE \
  --ulimit rtprio=99 --ulimit rttime=-1 --ulimit memlock=-1 \
  -v "$PWD/data:/data" fpi-robot:0.9.2 <command>
```

`--network=host` is effectively mandatory (FCI is TCP 1337 + UDP 1338 at 1 kHz; bridge
NAT causes `communication_constraints_violation`), and `/tmp` must not be `noexec`
because `loadModel()` downloads a shared object at runtime.

---

## Protocol details that actually matter

These are not polish. Each was measured to change the answer:

1. **Warm up 15–20 minutes.** Harmonic-drive friction falls noticeably as joints warm.
2. **Collect the dynamic stage in four blocks ordered `L B B L`** (ABBA), not
   `L B L B` and certainly not `L L B B`. This equalises the two configurations' mean
   collection times so thermal drift cancels *exactly* — and it stays exact even though
   the tool swaps take minutes, because the two swaps sit symmetrically about the middle.
   It also needs only two tool changes, since the middle `B B` pair is contiguous.
   In the self-test this single choice changes the inertia error by more than an order of
   magnitude. `scripts/run_campaign.sh` walks you through it.

   Pass the blocks as a comma-separated list; the pipeline concatenates them and drops
   the settling period from *each* block, which is what keeps the balance intact:

   ```
   fpi ident run --dynamic-loaded data/raw/dyn_loaded_1,data/raw/dyn_loaded_2 \
                 --dynamic-bare   data/raw/dyn_bare_1,data/raw/dyn_bare_2
   ```

   The static stage does not need this: its gravity signal is ~50x the drift residual.
3. **Zero the configured load in both runs** (the collector does this unless you pass
   `--no-zero-load`). Different load settings mean the internal controller tracks
   differently in each run, so the difference is no longer the payload alone. `assess_run`
   rejects runs that violate this.
4. **Weigh the tool.** It constrains Stage A and is the strongest validation you have.

---

## What to expect

**Mass and centre of mass are excellent.** The static stage is a four-parameter,
well-conditioned problem: better than 1 % on mass and ~1 mm on the centre of mass. Since
gravity compensation, collision thresholds and `tau_ext_hat_filtered` depend only on
these, Stage A alone gives most of the practical benefit.

**The inertia tensor is hard, and the report says so.** For a small tool the inertia
torque signature sits ~15× *below* the wrist torque-sensor noise floor, and the
parallel-axis term $m\lVert c\rVert^2$ is several times the tool's own $I_C$. Expect a few
percent to a few tens of percent on the diagonal and much worse on the products of
inertia. This is geometry, not a defect of the method — see THEORY.md §11. The pipeline
reports per-parameter relative uncertainty, flags anything the data did not determine as
*prior-dominated*, and falls back to the uniform-density bounding-box inertia rather than
reporting a noise fit.

That fallback is fine in practice: Franka uses the inertia only for feedforward inverse
dynamics (~10⁻² N·m against a 12–87 N·m range) and a sub-percent change to the impedance
mass matrix.

---

## Verifying the result

`fpi ident run` writes `payload_params.yaml` (machine-readable, in Desk's units and
frames) and `report.md`, which includes:

* the Desk fields and an equivalent `setLoad` call;
* per-parameter relative uncertainty with prior-dominated flags;
* Stage A vs Stage B agreement — a disagreement means friction leaked in or the runs
  differed;
* a **friction cancellation diagnostic** regressing the residual on `sign(q̇)` and `q̇`;
* held-out cross-validation RMSE per joint;
* the scale check.

Final acceptance on the robot: enter the values, then confirm
`|tau_ext_hat_filtered| < 0.5 N·m` at rest across the workspace and that hand-guiding is
drift-free.

---

## Licence and attribution

BSD-3-Clause. See `NOTICE` for vendored Apache-2.0 files from libfranka and for the
provenance of the URDF's link inertias (Gaz et al., RA-L 2019 — *not* manufacturer data,
and used here only as a prior).
