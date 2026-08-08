"""The hardware-free acceptance test, wired the same way as the real campaign.

``fpi ident synthetic`` runs this. It generates ground truth with Pinocchio, writes it
through the **real** record format, reads it back through the **real** loader, and
pushes it through the **real** estimators -- so a bug anywhere in that chain shows up
here rather than on the robot.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from .config import Config, asset_dir
from .data.robot_log import RunMetadata, save_run
from .model import PandaModel, bounding_box_prior, phi_from_mci, phi_to_mci
from .pipeline import PairPaths, identify
from .synthetic import NoiseModel, simulate_dynamic_pair, simulate_static_pair
from .traj.fourier import FourierTrajectory, StaticPoseSet
from .traj.optimize import optimize_static_poses


def reference_tool_phi(cfg: Config) -> np.ndarray:
    """Ground-truth tool, derived from ``config/tool.yaml`` so the two agree.

    The mass is taken from the configured scale reading and the geometry from the
    configured bounding box, then both are perturbed away from the prior: the CoM is
    offset from the box centre and the inertia is scaled down and given off-diagonal
    terms. If the truth *were* the prior, the regularised estimator would look perfect
    for the wrong reason.

    Deriving from the config also keeps the self-test honest when a user edits
    ``tool.yaml``: a hard-coded 0.73 kg tool would silently fight the configured mass
    constraint and fail for a reason that has nothing to do with the pipeline.
    """
    mass = cfg.tool.mass_scale
    if mass is None:
        raise ValueError("config/tool.yaml must define mass_scale for the synthetic test")

    centre = 0.5 * (cfg.tool.bbox_min + cfg.tool.bbox_max)
    extent = cfg.tool.bbox_max - cfg.tool.bbox_min
    com = centre + np.array([-0.15, 0.20, -0.10]) * extent

    box = bounding_box_prior(mass, cfg.tool.bbox_min, cfg.tool.bbox_max).inertia_com
    inertia_com = 0.55 * box
    off = 0.04 * float(np.trace(box)) / 3.0
    inertia_com = inertia_com + np.array([[0.0, off, 0.6 * off],
                                          [off, 0.0, -0.4 * off],
                                          [0.6 * off, -0.4 * off, 0.0]])
    return phi_from_mci(mass, com, inertia_com)


def _write_static_pair(directory: Path, pm: PandaModel, poses: np.ndarray,
                       phi_true: np.ndarray, noise: NoiseModel,
                       seed: int) -> PairPaths:
    """Generate a static pair and persist it in the collector's record format."""
    q, tau_l, tau_b, direction = simulate_static_pair(
        pm, poses, phi_true, noise=noise, seed=seed, bidirectional=True)

    dwell = 200  # samples per pose visit
    n_rows = q.shape[0]

    def write(tau: np.ndarray, loaded: bool, name: str) -> Path:
        # Expand each pose visit into a dwell block, as the collector would.
        q_rep = np.repeat(q, dwell, axis=0)
        tau_rep = np.repeat(tau, dwell, axis=0)
        k = q_rep.shape[0]
        from .data.robot_log import make_records
        values = make_records(
            seq=np.arange(k), time_s=np.arange(k) * 1e-3, dt_s=np.full(k, 1e-3),
            q=q_rep, dq=np.zeros_like(q_rep), q_d=q_rep, dq_d=np.zeros_like(q_rep),
            ddq_d=np.zeros_like(q_rep), tau_J=tau_rep, tau_J_d=np.zeros_like(tau_rep),
            dtau_J=np.zeros_like(tau_rep), tau_ext=np.zeros_like(tau_rep),
            o_t_ee=np.tile(np.eye(4).flatten(order="F"), (k, 1)),
            success_rate=np.zeros(k), robot_mode=np.full(k, 1.0), errors=np.zeros(k))
        meta = RunMetadata(run_id=name, kind="static", loaded=loaded,
                           robot_ip="synthetic", sample_rate_hz=1000.0,
                           samples_per_period=dwell, n_periods=n_rows,
                           notes="synthetic static sweep")
        stem = directory / name
        save_run(stem, values, meta)
        return stem

    return PairPaths(write(tau_l, True, "static_loaded"),
                     write(tau_b, False, "static_bare"))


def _write_dynamic_pair(directory: Path, pm: PandaModel, traj: FourierTrajectory,
                        phi_true: np.ndarray, noise: NoiseModel, n_periods: int,
                        seed: int, prefix: str) -> PairPaths:
    loaded, bare = simulate_dynamic_pair(pm, traj, phi_true, noise=noise,
                                         n_periods=n_periods, seed=seed)
    out = []
    for run, name in ((loaded, f"{prefix}_loaded"), (bare, f"{prefix}_bare")):
        run.meta.run_id = name
        stem = directory / name
        save_run(stem, run.values, run.meta)
        out.append(stem)
    return PairPaths(out[0], out[1])


def run_synthetic_pipeline(pm: PandaModel, cfg: Config, *, seed: int = 0,
                           out: Path | None = None, n_periods: int = 10,
                           n_poses: int = 40, verbose: bool = True) -> int:
    """Generate, persist, identify, report. Returns a process exit code."""
    phi_true = reference_tool_phi(cfg)
    mass_true, com_true, inertia_true = phi_to_mci(phi_true)
    noise = NoiseModel.from_config(cfg.experiment.synthetic)

    traj_file = asset_dir() / "excitation_reference.json"
    traj = FourierTrajectory.from_dict(json.loads(traj_file.read_text(encoding="utf-8")))

    poses = optimize_static_poses(pm, cfg.workspace, cfg.derated_limits(),
                                  n_poses=n_poses, seed=seed, n_candidates=1500)

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        static_pair = _write_static_pair(directory, pm, poses, phi_true, noise, seed)
        dynamic_pair = _write_dynamic_pair(directory, pm, traj, phi_true, noise,
                                           n_periods, seed + 1, "dynamic")
        validation_pair = _write_dynamic_pair(directory, pm, traj, phi_true, noise,
                                              max(n_periods // 2, 2), seed + 2, "validation")

        report = identify(pm, cfg, static_pair=static_pair, dynamic_pair=dynamic_pair,
                          validation_pair=validation_pair, quality_gate=False)

    final = report.final
    mass_err = abs(final.mass - mass_true) / mass_true
    com_err = float(np.abs(final.com - com_true).max())
    inertia_err = float(np.abs(np.diag(final.inertia_com) - np.diag(inertia_true)).max())

    if verbose:
        print(report.to_markdown())
        print("\n## Recovery against known ground truth\n")
        print(f"  mass    : {final.mass:.5f} vs {mass_true:.5f} kg "
              f"({100 * mass_err:.3f} %)")
        print(f"  CoM     : max error {com_err * 1e3:.3f} mm")
        print(f"  inertia : max diagonal error {inertia_err:.3e} kg m^2 "
              f"(true diagonal ~{np.diag(inertia_true).mean():.1e})")

    if out is not None:
        report.write_yaml(Path(out) / "payload_params.yaml")
        report.write_markdown(Path(out) / "report.md")
        if verbose:
            print(f"\nwrote {Path(out) / 'payload_params.yaml'}")

    ok = mass_err < 0.01 and com_err < 2e-3
    if verbose:
        print(f"\nresult: {'PASS' if ok else 'FAIL'} "
              "(mass within 1 %, centre of mass within 2 mm)")
    return 0 if ok else 1
