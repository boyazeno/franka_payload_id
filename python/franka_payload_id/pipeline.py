"""End-to-end orchestration: logs in, Desk-ready parameters out.

The CLI is a thin wrapper over these functions so the same pipeline can be driven from
a notebook or a test without going through argument parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .data.preprocess import (
    DynamicDataset,
    StaticDataset,
    average_static_dwells,
    build_dynamic_dataset,
    build_static_dataset,
    combine_approaches,
)
from .data.quality import assert_pair_compatible, assess_run
from .data.robot_log import RunLog, load_run
from .ident import identify_dynamic_sdp, identify_static
from .ident.dynamic_sdp import prediction_rmse
from .ident.logchol import identify_dynamic_logchol
from .ident.validate import cross_validate, friction_residual_diagnostic
from .model import InertialParams, PandaModel, bounding_box_prior
from .report import IdentificationReport


@dataclass
class PairPaths:
    """One or more collection blocks per configuration.

    Under the ABBA schedule each configuration is collected in several separate blocks
    (`L B B L`), so both sides are lists. They are concatenated in the order given, and
    the settling period is dropped from each block rather than only from the first --
    dropping it once would leave the blocks unequal and reintroduce the very drift
    imbalance ABBA exists to remove.
    """

    loaded: Path | list[Path]
    bare: Path | list[Path]

    @staticmethod
    def _as_list(value: Path | list[Path]) -> list[Path]:
        return [Path(value)] if isinstance(value, (str, Path)) else [Path(v) for v in value]

    @property
    def loaded_blocks(self) -> list[Path]:
        return self._as_list(self.loaded)

    @property
    def bare_blocks(self) -> list[Path]:
        return self._as_list(self.bare)


def concatenate_runs(paths: list[Path], *, periods_per_block: int | None = None,
                     verbose: bool = True) -> RunLog:
    """Load several blocks of one configuration and concatenate them into one log.

    Each block is trimmed to a whole number of periods and then to the **common**
    number of periods across blocks. What the ABBA cancellation requires is that every
    block contribute equally to the period average -- and the unit of that average is a
    period, not a sample. Raw sample counts routinely differ by a few dozen because a
    dropped frame costs one callback while the run still ends on the same wall-clock
    deadline; those samples sit in the trailing partial period that is discarded anyway.
    """
    runs = [load_run(p) for p in paths]

    first = runs[0].meta
    for other in runs[1:]:
        if other.meta.loaded != first.loaded:
            raise ValueError("cannot concatenate loaded and bare blocks into one log")
        if other.meta.samples_per_period != first.samples_per_period:
            raise ValueError("blocks have different period lengths")
        if abs(other.meta.sample_rate_hz - first.sample_rate_hz) > 1e-9:
            raise ValueError("blocks were recorded at different sample rates")

    spp = int(first.samples_per_period)
    if spp <= 0:
        raise ValueError("samples_per_period is not recorded in the run metadata")

    whole = [r.n_samples // spp for r in runs]
    if min(whole) < 1:
        raise ValueError(
            f"a block holds less than one whole period ({min(whole)} of {spp} samples "
            "each); the run was aborted too early to be usable")

    keep = int(min(whole)) if periods_per_block is None else int(periods_per_block)
    if keep > min(whole):
        raise ValueError(
            f"asked to keep {keep} periods per block but the shortest block holds only "
            f"{min(whole)}")

    if verbose and (len(set(whole)) > 1 or any(r.n_samples != keep * spp for r in runs)):
        detail = ", ".join(f"{p.name}: {r.n_samples} samples -> {keep} periods"
                           for p, r in zip(paths, runs))
        print(f"trimming each block to {keep} whole periods ({detail})")

    meta = runs[0].meta
    meta.n_blocks = len(runs)
    meta.n_periods = keep * len(runs)
    meta.run_id = "+".join(p.name for p in paths)
    return RunLog(np.vstack([r.values[: keep * spp] for r in runs]), meta)


def periods_per_block(run: RunLog) -> int:
    blocks = max(int(run.meta.n_blocks or 1), 1)
    return int(run.meta.n_periods) // blocks


def trim_to_periods_per_block(run: RunLog, keep: int) -> RunLog:
    """Re-slice an already-concatenated log down to ``keep`` periods per block."""
    spp = int(run.meta.samples_per_period)
    blocks = max(int(run.meta.n_blocks or 1), 1)
    current = periods_per_block(run)
    if keep == current:
        return run
    if keep > current:
        raise ValueError(f"cannot grow {current} periods per block to {keep}")

    rows = np.concatenate([
        np.arange(b * current * spp, b * current * spp + keep * spp) for b in range(blocks)
    ])
    meta = run.meta
    meta.n_periods = keep * blocks
    return RunLog(run.values[rows], meta)


def _prior(cfg: Config, mass_hint: float | None = None) -> InertialParams:
    """Prior J_0 for the regulariser: CAD if given, else a uniform-density box."""
    tool = cfg.tool
    if tool.cad_mass is not None and tool.cad_com is not None and tool.cad_inertia is not None:
        return InertialParams(tool.cad_mass, tool.cad_com, tool.cad_inertia, frame="flange")
    mass = mass_hint or tool.mass_scale
    if mass is None:
        raise ValueError(
            "no prior available: set either tool.cad.* or tool.mass_scale in config/tool.yaml")
    return bounding_box_prior(mass, tool.bbox_min, tool.bbox_max)


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------
def static_dataset_from_runs(loaded: RunLog, bare: RunLog, *,
                             window_fraction: float = 0.5) -> StaticDataset:
    """Reduce a pair of static-pose logs to per-pose torque differences."""
    assert_pair_compatible(loaded, bare)

    def reduce(run: RunLog):
        return average_static_dwells(
            run.q, run.tau_J, n_rows=run.meta.n_periods,
            samples_per_row=run.meta.samples_per_period,
            window_fraction=window_fraction)

    q_l, tau_l, std_l = reduce(loaded)
    q_b, tau_b, std_b = reduce(bare)

    # The pose file alternates approach directions, so rows pair up two by two.
    direction = np.tile([1, -1], len(q_l))[: len(q_l)]
    q_lc, tau_lc = combine_approaches(q_l, tau_l, direction)
    q_bc, tau_bc = combine_approaches(q_b, tau_b, direction)

    sigma = np.sqrt((std_l ** 2 + std_b ** 2).mean(axis=0))
    sigma = np.where(sigma > 0.0, sigma, 1.0)
    return build_static_dataset(q_lc, tau_lc, q_bc, tau_bc, sigma=sigma)


def run_static_stage(pm: PandaModel, cfg: Config, dataset: StaticDataset):
    tool = cfg.tool
    return identify_static(
        pm, dataset,
        mass_scale=tool.mass_scale,
        mass_tolerance=tool.mass_scale_tolerance,
        use_mass_constraint=tool.use_mass_constraint,
        length_scale=float(cfg.experiment.trajectory["optimizer"]["length_scale"]),
    )


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------
def dynamic_dataset_from_runs(loaded: RunLog, bare: RunLog, cfg: Config) -> DynamicDataset:
    assert_pair_compatible(loaded, bare)

    # The two configurations can also end up with different period counts -- the runs
    # are separate invocations and each ends on its own wall-clock deadline. Trim both
    # to the common minimum so every block on both sides weighs the same in the period
    # average, which is what keeps the ABBA drift cancellation exact.
    common = min(periods_per_block(loaded), periods_per_block(bare))
    if common < 2:
        raise ValueError(
            f"only {common} whole period(s) per block survive; at least 2 are needed "
            "because the first is discarded as settling transient")
    if periods_per_block(loaded) != common or periods_per_block(bare) != common:
        print(f"trimming both configurations to {common} periods per block "
              f"(loaded had {periods_per_block(loaded)}, bare {periods_per_block(bare)})")
        loaded = trim_to_periods_per_block(loaded, common)
        bare = trim_to_periods_per_block(bare, common)

    pp = cfg.experiment.preprocess
    return build_dynamic_dataset(
        loaded.q, loaded.tau_J, bare.q, bare.tau_J,
        sample_rate_hz=loaded.meta.sample_rate_hz,
        samples_per_period=loaded.meta.samples_per_period,
        cutoff_hz=float(pp["cutoff_hz"]),
        filter_order=int(pp["filter_order"]),
        decimate_to_hz=float(pp["decimate_to_hz"]),
        drop_first_period=bool(pp["drop_first_period"]),
        edge_trim_s=float(pp["edge_trim_s"]),
        zero_velocity_threshold=float(pp["zero_velocity_threshold"]),
        n_blocks=int(getattr(loaded.meta, "n_blocks", 1) or 1),
    )


def run_dynamic_stage(pm: PandaModel, cfg: Config, dataset: DynamicDataset, *,
                      prior: InertialParams, warm_start: InertialParams | None = None):
    est = cfg.experiment.estimator
    length_scale = float(cfg.experiment.trajectory["optimizer"]["length_scale"])
    threshold = float(cfg.experiment.validation["sigma_threshold_pct"])
    method = str(est.get("method", "sdp"))

    if method == "logchol":
        return identify_dynamic_logchol(
            pm, dataset, prior=prior, warm_start=warm_start, length_scale=length_scale,
            gamma=float(est["logchol"].get("gamma", 0.0) or 0.0),
            max_iter=int(est["logchol"]["max_iter"]),
            x_tol=float(est["logchol"]["x_tol"]),
            sigma_threshold_pct=threshold)

    sdp = est["sdp"]
    gamma = sdp.get("gamma")
    mass_bounds = None
    if cfg.tool.use_mass_constraint and cfg.tool.mass_scale is not None:
        tol = cfg.tool.mass_scale_tolerance
        mass_bounds = (cfg.tool.mass_scale * (1 - tol), cfg.tool.mass_scale * (1 + tol))

    return identify_dynamic_sdp(
        pm, dataset, prior=prior, length_scale=length_scale,
        gamma=float(gamma) if gamma is not None else 1e-2,
        solver=str(sdp["solver"]),
        psd_epsilon=float(sdp["psd_epsilon"]),
        use_entropic_prior=bool(sdp["use_entropic_prior"]),
        use_bounding_ellipsoid=bool(sdp["use_bounding_ellipsoid"]),
        bbox=(cfg.tool.bbox_min, cfg.tool.bbox_max),
        mass_bounds=mass_bounds,
        robust=str(est.get("robust", "none")),
        huber_delta_scale=float(est.get("huber_delta_scale", 1.345)),
        irls_iterations=int(est.get("irls_iterations", 3)),
        sigma_threshold_pct=threshold)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def identify(pm: PandaModel, cfg: Config, *,
             static_pair: PairPaths | None = None,
             dynamic_pair: PairPaths | None = None,
             validation_pair: PairPaths | None = None,
             quality_gate: bool = True) -> IdentificationReport:
    """Run whichever stages the supplied data allows and build the report."""
    notes: list[str] = []
    static_result = None
    dynamic_result = None
    validation = None
    friction = None

    def _load(pair: PairPaths) -> tuple[RunLog, RunLog]:
        if quality_gate:
            # Gate each block individually: a single bad block should be identifiable,
            # and its problems would be diluted by concatenation.
            for label, blocks in (("loaded", pair.loaded_blocks), ("bare", pair.bare_blocks)):
                for path in blocks:
                    report = assess_run(load_run(path))
                    if not report.ok:
                        raise ValueError(
                            f"{label} block {path} failed quality gating:\n{report.summary()}")
        return concatenate_runs(pair.loaded_blocks), concatenate_runs(pair.bare_blocks)

    if static_pair is not None:
        loaded, bare = _load(static_pair)
        static_result = run_static_stage(pm, cfg, static_dataset_from_runs(loaded, bare))

    if dynamic_pair is not None:
        loaded, bare = _load(dynamic_pair)
        dataset = dynamic_dataset_from_runs(loaded, bare, cfg)
        prior = _prior(cfg, static_result.mass if static_result else None)
        warm = static_result.as_params(prior.inertia_com) if static_result else None
        dynamic_result = run_dynamic_stage(pm, cfg, dataset, prior=prior, warm_start=warm)
        friction = friction_residual_diagnostic(dataset, pm, dynamic_result.phi)

        if validation_pair is not None:
            v_loaded, v_bare = _load(validation_pair)
            v_dataset = dynamic_dataset_from_runs(v_loaded, v_bare, cfg)
            validation = cross_validate(pm, v_dataset, dynamic_result.phi)

    return assemble_report(cfg, static_result, dynamic_result, validation, friction, notes)


def assemble_report(cfg: Config, static_result, dynamic_result, validation,
                    friction, notes: list[str]) -> IdentificationReport:
    """Decide the final parameter set from whichever stages ran.

    Precedence follows how much the data actually says:

    * mass and centre of mass come from **Stage A** when it ran, because the static
      problem is far better conditioned than the dynamic one;
    * the inertia tensor comes from Stage B, unless Stage B flagged it as
      prior-dominated, in which case the uniform-density bounding-box value is used and
      labelled as such -- that is more honest, and more useful, than a noise-fit.
    """
    inertia_source = "identified"

    if dynamic_result is not None:
        inertia = dynamic_result.params.inertia_com
        inertia_names = {"Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz"}
        if inertia_names.issubset(set(dynamic_result.prior_dominated)):
            inertia = _prior(cfg, static_result.mass if static_result else None).inertia_com
            inertia_source = "uniform-density bounding box (dynamic stage was prior-dominated)"
            notes.append(
                "Every inertia component exceeded the relative-standard-deviation "
                "threshold, so the reported inertia is the uniform-density bounding-box "
                "prior rather than an identified value.")
        elif dynamic_result.prior_dominated:
            inertia_source = ("identified, but "
                              + ", ".join(dynamic_result.prior_dominated)
                              + " are prior-dominated")
    elif static_result is not None:
        inertia = _prior(cfg, static_result.mass).inertia_com
        inertia_source = "uniform-density bounding box (no dynamic stage was run)"
        notes.append("Only the static stage was run, so the inertia tensor is the "
                     "bounding-box prior. For most uses this is adequate: Franka's "
                     "gravity compensation and collision thresholds depend only on the "
                     "mass and centre of mass.")
    else:
        raise ValueError("nothing to report: neither stage produced a result")

    if static_result is not None:
        mass, com = static_result.mass, static_result.com
    else:
        mass, com = dynamic_result.params.mass, dynamic_result.params.com

    final = InertialParams(mass, com, inertia, frame="flange")

    return IdentificationReport(
        tool_name=cfg.tool.name, final=final,
        static=static_result, dynamic=dynamic_result, validation=validation,
        friction_diagnostic=friction, mass_scale=cfg.tool.mass_scale,
        inertia_source=inertia_source, notes=notes)
