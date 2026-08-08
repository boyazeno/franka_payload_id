"""The record format shared with the C++ collector, and the quality gates.

The schema is the contract between two languages. A field added on one side and not
the other would otherwise shift every subsequent column silently, so the sidecar
carries the schema the collector actually used and the loader refuses a mismatch.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from franka_payload_id.data.quality import assert_pair_compatible, assess_run
from franka_payload_id.data.robot_log import (
    RECORD_SIZE,
    SCHEMA,
    RunLog,
    RunMetadata,
    load_run,
    make_records,
    save_run,
)

CPP_HEADER = Path(__file__).resolve().parents[1] / "cpp" / "include" / "fpi" / "state_log.hpp"


def _records(n: int, *, rng, success: float = 1.0, mode: float = 2.0,
             errors: float = 0.0, dt: float = 1e-3) -> np.ndarray:
    seven = lambda: rng.normal(size=(n, 7))  # noqa: E731
    return make_records(
        seq=np.arange(n), time_s=np.arange(n) * dt, dt_s=np.full(n, dt),
        q=seven(), dq=seven(), q_d=seven(), dq_d=seven(), ddq_d=seven(),
        tau_J=seven(), tau_J_d=seven(), dtau_J=seven(), tau_ext=seven(),
        o_t_ee=np.tile(np.eye(4).flatten(order="F"), (n, 1)),
        success_rate=np.full(n, success), robot_mode=np.full(n, mode),
        errors=np.full(n, errors))


def test_record_size_matches_the_schema():
    assert RECORD_SIZE == len(SCHEMA)
    assert RECORD_SIZE == 3 + 9 * 7 + 16 + 3


def test_cpp_and_python_agree_on_the_record_width():
    """Guards the cross-language contract from drifting.

    The C++ side asserts its own generated name list against this constant at runtime,
    so checking the constant here closes the loop at build time.
    """
    text = CPP_HEADER.read_text(encoding="utf-8")
    match = re.search(r"kRecordSize\s*=\s*([^;]+);", text)
    assert match, "kRecordSize not found in state_log.hpp"
    assert eval(match.group(1).split("//")[0].strip()) == RECORD_SIZE  # noqa: S307


def test_roundtrip_through_disk(tmp_path, rng):
    values = _records(50, rng=rng)
    meta = RunMetadata(run_id="t", kind="trajectory", loaded=True,
                       sample_rate_hz=1000.0, samples_per_period=25, n_periods=2)
    bin_path, meta_path = save_run(tmp_path / "run", values, meta)
    assert bin_path.exists() and meta_path.exists()

    run = load_run(tmp_path / "run")
    np.testing.assert_allclose(run.values, values, atol=0)
    assert run.meta.run_id == "t"
    assert run.meta.loaded is True
    np.testing.assert_allclose(run.tau_J, values[:, [SCHEMA.index(f"tau_J_{i}")
                                                     for i in range(7)]])


def test_load_accepts_either_filename(tmp_path, rng):
    save_run(tmp_path / "run", _records(5, rng=rng), RunMetadata())
    for candidate in ("run", "run.bin", "run.meta.json"):
        assert load_run(tmp_path / candidate).n_samples == 5


def test_schema_mismatch_is_rejected(tmp_path, rng):
    save_run(tmp_path / "run", _records(5, rng=rng), RunMetadata())
    meta_path = tmp_path / "run.meta.json"
    data = json.loads(meta_path.read_text())
    data["schema"] = data["schema"][:-1]           # collector from a different build
    meta_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different record schema"):
        load_run(tmp_path / "run")


def test_truncated_file_is_rejected(tmp_path, rng):
    save_run(tmp_path / "run", _records(5, rng=rng), RunMetadata())
    bin_path = tmp_path / "run.bin"
    raw = bin_path.read_bytes()
    bin_path.write_bytes(raw[: len(raw) - 24])
    with pytest.raises(ValueError, match="truncated or corrupt"):
        load_run(tmp_path / "run")


def test_missing_sidecar_is_rejected(tmp_path, rng):
    save_run(tmp_path / "run", _records(5, rng=rng), RunMetadata())
    (tmp_path / "run.meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="metadata sidecar"):
        load_run(tmp_path / "run")


def test_wrong_width_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="expected"):
        save_run(tmp_path / "run", np.zeros((10, 5)), RunMetadata())


# ---------------------------------------------------------------- quality
def test_good_run_passes(rng):
    run = RunLog(_records(2000, rng=rng), RunMetadata())
    assert assess_run(run).ok


def test_low_success_rate_is_rejected(rng):
    values = _records(2000, rng=rng)
    values[500:600, SCHEMA.index("control_command_success_rate")] = 0.80
    report = assess_run(RunLog(values, RunMetadata()))
    assert not report.ok
    assert any("success_rate" in p for p in report.problems)


def test_long_control_periods_are_rejected(rng):
    values = _records(2000, rng=rng)
    values[:200, SCHEMA.index("dt_s")] = 5e-3
    report = assess_run(RunLog(values, RunMetadata()))
    assert not report.ok
    assert any("control periods exceeded" in p for p in report.problems)


def test_reflex_mode_is_rejected(rng):
    values = _records(2000, rng=rng)
    values[1000:1100, SCHEMA.index("robot_mode")] = 4.0     # kReflex
    report = assess_run(RunLog(values, RunMetadata()))
    assert not report.ok
    assert any("Idle/Move" in p for p in report.problems)


def test_nonzero_configured_load_is_rejected(rng):
    """Both runs of a pair must be collected with the load zeroed."""
    meta = RunMetadata(m_load=0.5)
    report = assess_run(RunLog(_records(500, rng=rng), meta))
    assert not report.ok
    assert any("configured total load" in p for p in report.problems)


def test_pair_compatibility_checks(rng):
    values = _records(100, rng=rng)
    loaded = RunLog(values, RunMetadata(loaded=True, samples_per_period=50))
    bare = RunLog(values, RunMetadata(loaded=False, samples_per_period=50))
    assert_pair_compatible(loaded, bare)

    with pytest.raises(ValueError, match="both runs are marked"):
        assert_pair_compatible(loaded, RunLog(values, RunMetadata(loaded=True,
                                                                 samples_per_period=50)))

    mismatched = RunLog(values, RunMetadata(loaded=False, samples_per_period=50, m_ee=0.7))
    with pytest.raises(ValueError, match="different configured end-effector"):
        assert_pair_compatible(loaded, mismatched)

    with pytest.raises(ValueError, match="different period lengths"):
        assert_pair_compatible(loaded, RunLog(values, RunMetadata(loaded=False,
                                                                 samples_per_period=20)))
