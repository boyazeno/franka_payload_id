"""``fpi`` command line.

Deliberately thin: every subcommand loads config, calls into :mod:`.pipeline`,
:mod:`.traj` or :mod:`.ident`, and prints. The logic lives in those modules so it can
be driven from a notebook or a test without argument parsing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .config import Config, asset_dir, data_dir
from .model import PandaModel


# ---------------------------------------------------------------------------
def _load(args) -> tuple[PandaModel, Config]:
    cfg = Config.load(Path(args.config) if args.config else None)
    pm = PandaModel.load(Path(args.urdf) if args.urdf else None)
    return pm, cfg


# ---------------------------------------------------------------------------
# traj
# ---------------------------------------------------------------------------
def cmd_traj_generate(args) -> int:
    from .traj.optimize import optimize_trajectory

    pm, cfg = _load(args)
    opt = cfg.experiment.trajectory["optimizer"]
    tr = cfg.experiment.trajectory

    result = optimize_trajectory(
        pm, cfg.workspace, cfg.derated_limits(),
        n_harmonics=int(args.harmonics or tr["n_harmonics"]),
        base_frequency=float(args.frequency or tr["base_frequency_hz"]),
        n_collocation=int(tr["n_collocation"]),
        length_scale=float(opt["length_scale"]),
        criterion=str(opt["objective"]),
        n_restarts=int(args.restarts or opt["n_restarts"]),
        max_iter=int(opt["max_iter"]),
        seed=int(args.seed if args.seed is not None else opt["seed"]))

    print(f"condition number (column-scaled): {result.condition:.1f}")
    print(f"log det:                          {result.log_det:.2f}")
    print(result.report.summary())

    target = float(opt["target_condition"])
    if result.condition > target:
        print(f"\nNOTE: condition {result.condition:.1f} exceeds the target {target:.0f}. "
              "For a small tool this is common -- the inertia columns are intrinsically "
              "weakly excited. Check the per-parameter %sigma in the final report rather "
              "than treating the condition number alone as pass/fail.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = result.trajectory.to_dict()
    payload["_provenance"] = {"condition": float(result.condition),
                              "log_det": float(result.log_det),
                              "criterion": str(opt["objective"])}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0 if result.ok else 1


def cmd_traj_export(args) -> int:
    from .traj.export import UnsafeExportError, export_trajectory
    from .traj.fourier import FourierTrajectory

    pm, cfg = _load(args)
    source = Path(args.traj) if args.traj else asset_dir() / "excitation_reference.json"
    tr = FourierTrajectory.from_dict(json.loads(source.read_text(encoding="utf-8")))

    try:
        path = export_trajectory(
            args.out, tr, pm, cfg.workspace, cfg.derated_limits(),
            sample_rate_hz=float(cfg.experiment.trajectory["sample_rate_hz"]),
            n_periods=int(args.periods or cfg.experiment.trajectory["n_periods"]),
            force=args.force)
    except UnsafeExportError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"wrote {path}")
    return 0


def cmd_traj_check(args) -> int:
    from .traj.constraints import check_trajectory
    from .traj.fourier import FourierTrajectory

    pm, cfg = _load(args)
    source = Path(args.traj) if args.traj else asset_dir() / "excitation_reference.json"
    tr = FourierTrajectory.from_dict(json.loads(source.read_text(encoding="utf-8")))
    report = check_trajectory(pm, cfg.workspace, cfg.derated_limits(), tr, n_samples=1000)
    print(report.summary())
    return 0 if report.ready_for_hardware else 1


def cmd_traj_view(args) -> int:
    from .viz import view_trajectory
    from .traj.fourier import FourierTrajectory

    pm, cfg = _load(args)
    source = Path(args.traj) if args.traj else asset_dir() / "excitation_reference.json"
    tr = FourierTrajectory.from_dict(json.loads(source.read_text(encoding="utf-8")))
    return view_trajectory(pm, cfg, tr, out=Path(args.out) if args.out else None)


# ---------------------------------------------------------------------------
# poses
# ---------------------------------------------------------------------------
def cmd_poses_generate(args) -> int:
    from .traj.optimize import optimize_static_poses

    pm, cfg = _load(args)
    st = cfg.experiment.static
    poses = optimize_static_poses(
        pm, cfg.workspace, cfg.derated_limits(),
        n_poses=int(args.count or st["n_poses"]),
        length_scale=float(cfg.experiment.trajectory["optimizer"]["length_scale"]),
        seed=int(args.seed if args.seed is not None else st["seed"]))

    from .model import stack_gravity_regressor
    scale = np.array([1.0, 0.1, 0.1, 0.1])
    cond = float(np.linalg.cond(stack_gravity_regressor(pm, poses) * scale))
    print(f"selected {len(poses)} poses, gravity-regressor condition {cond:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, poses, delimiter=",")
    print(f"wrote {out}")
    return 0


def cmd_poses_export(args) -> int:
    from .traj.export import UnsafeExportError, export_static_poses
    from .traj.fourier import StaticPoseSet

    pm, cfg = _load(args)
    st = cfg.experiment.static
    poses = np.atleast_2d(np.loadtxt(args.poses, delimiter=","))
    offset = np.full(7, float(st["approach_offset_rad"]))
    pose_set = StaticPoseSet(poses, offset, bidirectional=bool(st["bidirectional"]))

    try:
        path = export_static_poses(args.out, pose_set, pm, cfg.workspace,
                                   cfg.derated_limits(), force=args.force)
    except UnsafeExportError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"wrote {path} ({len(pose_set.waypoints())} rows)")
    return 0


# ---------------------------------------------------------------------------
# ident
# ---------------------------------------------------------------------------
def cmd_ident_run(args) -> int:
    from .pipeline import PairPaths, identify

    pm, cfg = _load(args)

    def blocks(value: str | None) -> list[Path] | None:
        """Comma-separated list of run stems, one per collection block."""
        if not value:
            return None
        return [Path(v.strip()) for v in value.split(",") if v.strip()]

    def pair(loaded: str | None, bare: str | None, name: str) -> PairPaths | None:
        lo, ba = blocks(loaded), blocks(bare)
        if lo is None and ba is None:
            return None
        if lo is None or ba is None:
            raise SystemExit(f"error: --{name}-loaded and --{name}-bare must both be given")
        return PairPaths(lo, ba)

    static_pair = pair(args.static_loaded, args.static_bare, "static")
    dynamic_pair = pair(args.dynamic_loaded, args.dynamic_bare, "dynamic")
    validation_pair = pair(args.validation_loaded, args.validation_bare, "validation")

    if static_pair is None and dynamic_pair is None:
        print("error: supply at least one of --static-loaded/--static-bare or "
              "--dynamic-loaded/--dynamic-bare", file=sys.stderr)
        return 1

    report = identify(pm, cfg, static_pair=static_pair, dynamic_pair=dynamic_pair,
                      validation_pair=validation_pair,
                      quality_gate=not args.no_quality_gate)

    out_dir = Path(args.out) if args.out else data_dir() / "results"
    yaml_path = report.write_yaml(out_dir / "payload_params.yaml")
    md_path = report.write_markdown(out_dir / "report.md")
    print(report.to_markdown())
    print(f"\nwrote {yaml_path}\nwrote {md_path}")
    return 0


def cmd_ident_synthetic(args) -> int:
    """Full pipeline on generated data. The hardware-free acceptance test."""
    from .selftest import run_synthetic_pipeline

    pm, cfg = _load(args)
    return run_synthetic_pipeline(pm, cfg, seed=int(args.seed or 0),
                                  out=Path(args.out) if args.out else None)


# ---------------------------------------------------------------------------
# verification helpers
# ---------------------------------------------------------------------------
def cmd_verify_fk(args) -> int:
    """Compare a logged O_T_EE against Pinocchio's flange FK.

    The frame-convention gate: run ``fpi_check --out`` on the robot with the Desk
    end-effector transform set to identity, then point this at the resulting log.
    """
    from .data import load_run

    pm, _ = _load(args)
    run = load_run(args.log)
    o_t_ee = run.block("O_T_EE", 16)

    worst_pos, worst_rot = 0.0, 0.0
    step = max(1, run.n_samples // 200)
    for k in range(0, run.n_samples, step):
        measured = o_t_ee[k].reshape(4, 4, order="F")
        predicted = pm.flange_placement(run.q[k]).homogeneous
        worst_pos = max(worst_pos, float(np.linalg.norm(measured[:3, 3] - predicted[:3, 3])))
        rel = measured[:3, :3].T @ predicted[:3, :3]
        angle = np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
        worst_rot = max(worst_rot, float(angle))

    print(f"flange FK cross-check over {run.n_samples} samples:")
    print(f"  max position error : {worst_pos * 1e3:.4f} mm")
    print(f"  max rotation error : {np.degrees(worst_rot):.5f} deg")

    ok = worst_pos < 1e-3 and worst_rot < np.radians(0.1)
    if not ok:
        print("\nFAIL. Either the Desk end-effector transform is not identity for this\n"
              "log (F_T_EE must be the identity for a bare flange comparison), or the\n"
              "URDF does not describe this robot. Do not proceed to collection.")
    else:
        print("\nPASS -- the URDF flange frame matches the robot's.")
    return 0 if ok else 1


def cmd_fit_plane(args) -> int:
    """Fit a wall plane from >=3 measured flange positions."""
    points = np.atleast_2d(np.loadtxt(args.points, delimiter=","))
    if points.shape[0] < 3:
        print("error: need at least 3 points", file=sys.stderr)
        return 1
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid)
    normal = vt[-1]
    # Orient the normal so the robot base (origin) lies in the allowed half-space.
    offset = -float(normal @ centroid)
    if offset < 0:
        normal, offset = -normal, -offset

    residual = float(np.abs(points @ normal + offset).max())
    print("fitted plane (base frame):")
    print(f"  normal: [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")
    print(f"  offset: {offset:.6f}")
    print(f"  max residual of the input points: {residual * 1e3:.2f} mm")
    print("\nPaste into config/workspace.yaml:")
    print(f"  - name: {args.name}")
    print(f"    normal: [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")
    print(f"    offset: {offset:.6f}")
    print("    measured: true")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fpi", description="Franka Panda payload inertial-parameter identification")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--config", help="directory holding the config/*.yaml files")
    p.add_argument("--urdf", help="override the Panda URDF path")
    sub = p.add_subparsers(dest="command", required=True)

    traj = sub.add_parser("traj", help="excitation trajectory design").add_subparsers(
        dest="sub", required=True)

    g = traj.add_parser("generate", help="optimise a new excitation trajectory")
    g.add_argument("--out", default=str(asset_dir() / "excitation.json"))
    g.add_argument("--harmonics", type=int)
    g.add_argument("--frequency", type=float)
    g.add_argument("--restarts", type=int)
    g.add_argument("--seed", type=int)
    g.set_defaults(func=cmd_traj_generate)

    e = traj.add_parser("export", help="write the 1 kHz CSV the collector replays")
    e.add_argument("--traj")
    e.add_argument("--out", required=True)
    e.add_argument("--periods", type=int)
    e.add_argument("--force", action="store_true",
                   help="export even if the safety check fails (use only after "
                        "independently verifying the trajectory)")
    e.set_defaults(func=cmd_traj_export)

    c = traj.add_parser("check", help="safety-check a trajectory")
    c.add_argument("--traj")
    c.set_defaults(func=cmd_traj_check)

    v = traj.add_parser("view", help="render the trajectory with the wall planes")
    v.add_argument("--traj")
    v.add_argument("--out", help="write a PNG instead of opening meshcat")
    v.set_defaults(func=cmd_traj_view)

    poses = sub.add_parser("poses", help="static pose design").add_subparsers(
        dest="sub", required=True)

    pg = poses.add_parser("generate", help="select D-optimal static poses")
    pg.add_argument("--out", default=str(asset_dir() / "static_poses.csv"))
    pg.add_argument("--count", type=int)
    pg.add_argument("--seed", type=int)
    pg.set_defaults(func=cmd_poses_generate)

    pe = poses.add_parser("export", help="write the pose CSV the collector replays")
    pe.add_argument("--poses", default=str(asset_dir() / "static_poses.csv"))
    pe.add_argument("--out", required=True)
    pe.add_argument("--force", action="store_true")
    pe.set_defaults(func=cmd_poses_export)

    ident = sub.add_parser("ident", help="identification").add_subparsers(
        dest="sub", required=True)

    ir = ident.add_parser("run", help="identify from collected logs")
    _blocks_help = ("run stem, or a comma-separated list of stems -- one per collection "
                    "block of the ABBA schedule")
    ir.add_argument("--static-loaded", help=_blocks_help)
    ir.add_argument("--static-bare", help=_blocks_help)
    ir.add_argument("--dynamic-loaded", help=_blocks_help)
    ir.add_argument("--dynamic-bare", help=_blocks_help)
    ir.add_argument("--validation-loaded", help=_blocks_help)
    ir.add_argument("--validation-bare", help=_blocks_help)
    ir.add_argument("--out")
    ir.add_argument("--no-quality-gate", action="store_true")
    ir.set_defaults(func=cmd_ident_run)

    isyn = ident.add_parser("synthetic", help="run the whole pipeline on generated data")
    isyn.add_argument("--seed", type=int)
    isyn.add_argument("--out")
    isyn.set_defaults(func=cmd_ident_synthetic)

    vf = sub.add_parser("verify-fk", help="compare a logged O_T_EE against the URDF FK")
    vf.add_argument("--log", required=True)
    vf.set_defaults(func=cmd_verify_fk)

    fp = sub.add_parser("fit-plane", help="fit a wall plane from touch points")
    fp.add_argument("--points", required=True, help="CSV of x,y,z rows in the base frame")
    fp.add_argument("--name", default="wall")
    fp.set_defaults(func=cmd_fit_plane)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
