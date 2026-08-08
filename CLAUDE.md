# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Payload inertial-parameter identification for a Franka Emika **Panda (FER)**: identify a
flange-mounted tool's mass, centre of mass and inertia tensor for Desk's End Effector
fields. A C++ collector logs `franka::RobotState` at 1 kHz; an offline Python pipeline
(Pinocchio + cvxpy) turns the logs into parameters.

`docs/THEORY.md` is the authoritative explanation of the maths and contains a
formula → code → test index in §12. **Read it before changing anything in `model/`,
`traj/` or `ident/`** — most of the non-obvious choices are justified there, and the
document is meant to stay in sync with the code.

## Commands

```bash
# Tests. Clear PYTHONPATH: a system ROS install leaks pytest plugins into the venv.
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q

# Whole pipeline on generated data with known ground truth (the acceptance test)
env -u PYTHONPATH .venv/bin/python -m franka_payload_id.cli ident synthetic

# Containers
docker compose -f docker/docker-compose.yml build analysis   # anywhere
docker compose -f docker/docker-compose.yml run --rm analysis pytest -q
docker build -f docker/Dockerfile.robot -t fpi-robot:0.9.2 . # builds libfranka 0.9.2

# C++ only builds with libfranka present; it is off by default
cmake -B build -DBUILD_ROBOT_COLLECTOR=ON && cmake --build build
```

Dev environment: `uv venv .venv && uv pip install -r requirements.txt && uv pip install --no-deps -e .`

## Hard constraints — do not "fix" these

* **libfranka 0.9.2 only.** ≥ 0.10 is FR3-only and will refuse to connect to a Panda.
  Do not bump it.
* **Panda limits, not FR3 limits.** `config/panda_limits.yaml` is from the FCI docs.
  Joint 4 is entirely negative and joint 6 almost entirely positive; code that assumes
  symmetric ranges is wrong.
* **`tau_J` is not gravity-compensated.** It is the raw link-side sensor. Only the
  *commanded* path is compensated.
* **`I_ee`/`I_load` are about the centre of mass**, in flange axes, column-major.
  `F_x_C*` is w.r.t. the flange origin.
* **Pinocchio's `phi` ordering is `xx, xy, yy, xz, yz, zz`** and the six entries are the
  inertia about the frame **origin**. Never unpack it by hand — use `phi_to_mci` /
  `InertialParams.from_phi`.
* **`pinocchio.computeStaticRegressor` is not the gravity regressor.** Use
  `computeJointTorqueRegressor(model, data, q, 0, 0)`.
* **`pinocchio.rnea` does not populate `data.oMi`.** Frame placements need
  `forwardKinematics`; using rnea leaves every placement at the identity, silently.
* **Regressor call order.** In `payload_regressor`, the body regressor must be evaluated
  and copied *before* `computeFrameJacobian`, which re-runs `forwardKinematics` and zeroes
  `data.v`/`data.a_gf`.
* **The record schema is a cross-language contract.** `data/robot_log.py::SCHEMA` and
  `cpp/include/fpi/state_log.hpp::kRecordSize` must agree; a test asserts it and the C++
  self-checks at runtime.
* **Never weaken the export safety gate.** `traj/export.py` refuses to write a trajectory
  while `config/workspace.yaml` still holds unmeasured placeholder planes. The robot
  stands in a corner facing outward.

## Conventions

* Config lives in `config/*.yaml` and is loaded through `config.py`. No magic numbers in
  algorithms.
* Estimators return dataclasses with a `summary()`; `report.py` owns all user-facing
  formatting.
* Anything asserted in a docstring should have a test. Several tests exist specifically to
  document a trap (`test_pinocchio_static_regressor_is_not_the_gravity_regressor`,
  `test_lowpass_and_central_differences_commute`).
* Uncertainty is computed via SVD, never `pinv` — truncation makes an unidentifiable
  parameter look well determined. See `covariance_from_design`.

## Status

All build units are implemented and the suite passes hardware-free. Not yet run on the
real robot: `config/workspace.yaml` still holds placeholder wall planes and
`config/tool.yaml` a placeholder mass. Both must be filled in on the robot PC before
collection — see README "Before touching the robot".
