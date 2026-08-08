"""Turn identification results into the numbers that go into Desk.

Two artefacts are produced:

``payload_params.yaml``
    Machine-readable, in exactly the units and frames Franka expects: mass in kg,
    centre of mass w.r.t. the **flange** frame in m, and the inertia tensor **about the
    centre of mass** in flange-frame axes, given both as a matrix and column-major
    flattened for ``Robot::setLoad``.

``report.md``
    The human-facing record: what was measured, how well, and -- importantly -- which
    parameters the data did *not* determine. A parameter whose relative standard
    deviation exceeds the threshold is reported as prior-dominated rather than dressed
    up as a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .ident.dynamic_sdp import DynamicResult
from .ident.static_ls import StaticResult
from .ident.validate import ValidationReport
from .model import InertialParams, consistency_report
from .model.params import PARAM_NAMES


@dataclass
class IdentificationReport:
    tool_name: str
    final: InertialParams
    static: StaticResult | None = None
    dynamic: DynamicResult | None = None
    validation: ValidationReport | None = None
    friction_diagnostic: dict | None = None
    bootstrap: dict | None = None
    mass_scale: float | None = None
    inertia_source: str = "identified"
    notes: list[str] = field(default_factory=list)

    # -- machine-readable ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        fields = self.final.as_desk_fields()
        # Independent list objects, so PyYAML does not emit an anchor/alias pair for the
        # centre of mass. Aliases are valid YAML but a nuisance to read and to copy from.
        out: dict[str, Any] = {
            "tool": self.tool_name,
            "frame": "flange (panda_link8)",
            "desk_end_effector_parameters": {
                "mass_kg": fields["mass"],
                "center_of_mass_m": list(fields["com"]),
                "inertia_about_com_kg_m2": [list(r) for r in fields["inertia_matrix"]],
            },
            "set_load_arguments": {
                "load_mass": fields["mass"],
                "F_x_Cload": list(fields["com"]),
                "load_inertia_column_major": list(fields["inertia_column_major"]),
            },
            "inertia_source": self.inertia_source,
            "physical_consistency": consistency_report(self.final.to_phi()),
        }
        if self.static is not None:
            out["stage_a_static"] = {
                "mass_kg": float(self.static.mass),
                "com_m": [float(v) for v in self.static.com],
                "condition_number": float(self.static.condition),
                "residual_rms_nm": float(self.static.residual_rms_nm),
                "relative_std_pct": [float(v) for v in self.static.sigma_pct],
                "n_poses": int(self.static.n_poses),
                "mass_constrained_to_scale": bool(self.static.mass_constrained),
            }
        if self.dynamic is not None:
            out["stage_b_dynamic"] = {
                "phi": [float(v) for v in self.dynamic.phi],
                "phi_names": list(PARAM_NAMES),
                "condition_number": float(self.dynamic.condition),
                "residual_rms_nm": float(self.dynamic.residual_rms_nm),
                "relative_std_pct": [float(v) for v in self.dynamic.sigma_pct],
                "prior_dominated": list(self.dynamic.prior_dominated),
                "n_equations": int(self.dynamic.n_equations),
                "gamma": float(self.dynamic.gamma),
                "solver": self.dynamic.solver,
                "physically_consistent": bool(self.dynamic.physically_consistent),
            }
        if self.validation is not None:
            out["cross_validation"] = {
                "overall_rmse_nm": float(self.validation.overall_rmse),
                "rmse_per_joint_nm": [float(v) for v in self.validation.rmse_per_joint],
                "relative_per_joint": [float(v) for v in self.validation.relative_per_joint],
            }
        if self.mass_scale is not None:
            err = 100.0 * abs(self.final.mass - self.mass_scale) / self.mass_scale
            out["scale_check"] = {
                "measured_kg": float(self.mass_scale),
                "identified_kg": float(self.final.mass),
                "relative_error_pct": float(err),
            }
        if self.notes:
            out["notes"] = list(self.notes)
        return out

    def write_yaml(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)
        return path

    # -- human-readable -----------------------------------------------------
    def to_markdown(self) -> str:
        f = self.final
        lines: list[str] = []
        lines.append(f"# Payload identification — {self.tool_name}\n")

        lines.append("## Values for Desk → Settings → End Effector\n")
        lines.append("All quantities are in the **flange frame** (`panda_link8`). The")
        lines.append("inertia is **about the centre of mass**, not about the flange origin.\n")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| mass | **{f.mass:.4f} kg** |")
        lines.append(f"| CoM x | **{f.com[0]:+.5f} m** |")
        lines.append(f"| CoM y | **{f.com[1]:+.5f} m** |")
        lines.append(f"| CoM z | **{f.com[2]:+.5f} m** |")
        for i, ax in enumerate("xyz"):
            for j, bx in enumerate("xyz"):
                if j >= i:
                    lines.append(f"| I{ax}{bx} | **{f.inertia_com[i, j]:+.6e} kg·m²** |")
        lines.append("")
        lines.append("Equivalent `libfranka` call (inertia column-major):\n")
        cm = ", ".join(f"{v:.6e}" for v in f.inertia_com.flatten(order="F"))
        lines.append("```cpp")
        lines.append(f"robot.setLoad({f.mass:.6f},")
        lines.append(f"              {{{{{f.com[0]:.6f}, {f.com[1]:.6f}, {f.com[2]:.6f}}}}},")
        lines.append(f"              {{{{{cm}}}}});")
        lines.append("```\n")

        cons = consistency_report(f.to_phi())
        ok = "yes" if cons["physically_consistent"] else "**NO**"
        lines.append(f"Physically consistent (J(φ) ≻ 0): {ok}. "
                     f"Principal moments: "
                     + ", ".join(f"{v:.3e}" for v in cons.get("principal_moments", []))
                     + "\n")

        if self.static is not None:
            lines.append("## Stage A — static gravity identification\n")
            lines.append("```")
            lines.append(self.static.summary())
            lines.append("```\n")

        if self.dynamic is not None:
            lines.append("## Stage B — dynamic identification\n")
            lines.append("```")
            lines.append(self.dynamic.summary())
            lines.append("```\n")
            if self.dynamic.prior_dominated:
                lines.append("> **Prior-dominated parameters.** The following were not "
                             "determined by the data (relative standard deviation above "
                             "the threshold) and have fallen back towards the prior: "
                             + ", ".join(f"`{p}`" for p in self.dynamic.prior_dominated)
                             + ". This is expected for a small tool: the inertia torque "
                             "signature sits well below the joint torque-sensor noise "
                             "floor, and the parallel-axis term m·|c|² dominates the "
                             "tool's own inertia. See docs/THEORY.md §11.\n")

        if self.static is not None and self.dynamic is not None:
            dm = abs(self.dynamic.params.mass - self.static.mass)
            dc = float(np.abs(self.dynamic.params.com - self.static.com).max())
            agree = "agree" if (dm / max(self.static.mass, 1e-9) < 0.02 and dc < 3e-3) \
                else "**DISAGREE**"
            lines.append("## Stage A vs Stage B\n")
            lines.append(f"The two stages {agree}: mass differs by {dm * 1e3:.2f} g "
                         f"({100 * dm / max(self.static.mass, 1e-9):.2f} %), centre of mass "
                         f"by at most {dc * 1e3:.2f} mm.\n")
            if agree.startswith("**"):
                lines.append("> A disagreement here points at friction leaking into the "
                             "dynamic fit, or at the two runs not having traversed the "
                             "same trajectory. Check the friction diagnostic below.\n")

        if self.validation is not None:
            lines.append("## Cross-validation on held-out data\n")
            lines.append("```")
            lines.append(self.validation.summary())
            lines.append("```\n")

        if self.friction_diagnostic is not None:
            lines.append("## Friction cancellation diagnostic\n")
            lines.append("Residual regressed onto sign(q̇) and q̇. A significant Coulomb "
                         "term means friction did **not** cancel between the loaded and "
                         "bare runs — usually a thermal or repeatability problem.\n")
            lines.append("| joint | Coulomb [Nm] | viscous [Nm·s] | noise σ [Nm] |")
            lines.append("|---|---|---|---|")
            d = self.friction_diagnostic
            for j in range(len(d["coulomb_nm"])):
                lines.append(f"| {j + 1} | {d['coulomb_nm'][j]:+.4f} | "
                             f"{d['viscous_nms'][j]:+.4f} | {d['noise_nm'][j]:.4f} |")
            lines.append("")

        if self.mass_scale is not None:
            err = 100.0 * abs(f.mass - self.mass_scale) / self.mass_scale
            verdict = "consistent" if err < 2.0 else "**INCONSISTENT**"
            lines.append("## Independent mass check\n")
            lines.append(f"Scale reading {self.mass_scale:.4f} kg vs identified "
                         f"{f.mass:.4f} kg — {err:.2f} % ({verdict}).\n")
            if err >= 2.0:
                lines.append("> This is the single strongest validation available. A "
                             "disagreement above ~2 % means something structural is wrong "
                             "(runs swapped, wrong frame, torque-sensor bias) — do not "
                             "enter these values.\n")

        if self.notes:
            lines.append("## Notes\n")
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")

        return "\n".join(lines)

    def write_markdown(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path
