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
def _bare_meta(**kw) -> RunMetadata:
    """A bare run declaring zero load: valid, so other gates can be tested alone."""
    return RunMetadata(loaded=False, **kw)


def test_good_run_passes(rng):
    run = RunLog(_records(2000, rng=rng), _bare_meta())
    assert assess_run(run).ok


def test_low_success_rate_is_rejected(rng):
    values = _records(2000, rng=rng)
    values[500:600, SCHEMA.index("control_command_success_rate")] = 0.80
    report = assess_run(RunLog(values, _bare_meta()))
    assert not report.ok
    assert any("success_rate" in p for p in report.problems)


def test_long_control_periods_are_rejected(rng):
    values = _records(2000, rng=rng)
    values[:200, SCHEMA.index("dt_s")] = 5e-3
    report = assess_run(RunLog(values, _bare_meta()))
    assert not report.ok
    assert any("control periods exceeded" in p for p in report.problems)


def test_reflex_mode_is_rejected(rng):
    values = _records(2000, rng=rng)
    values[1000:1100, SCHEMA.index("robot_mode")] = 4.0     # kReflex
    report = assess_run(RunLog(values, _bare_meta()))
    assert not report.ok
    assert any("Idle/Move" in p for p in report.problems)


def test_bare_run_declaring_a_load_is_rejected(rng):
    meta = RunMetadata(loaded=False, m_load=0.5)
    report = assess_run(RunLog(_records(500, rng=rng), meta))
    assert not report.ok
    assert any("BARE run declares a load" in p for p in report.problems)


def test_loaded_run_declaring_no_load_is_rejected(rng):
    """Carrying an unmodelled tool breaks gravity compensation and risks a reflex."""
    meta = RunMetadata(loaded=True, m_load=0.0)
    report = assess_run(RunLog(_records(500, rng=rng), meta))
    assert not report.ok
    assert any("LOADED run declares zero load" in p for p in report.problems)


def test_truthfully_declared_pair_is_accepted(rng):
    loaded = RunLog(_records(500, rng=rng), RunMetadata(loaded=True, m_load=0.5))
    bare = RunLog(_records(500, rng=rng), RunMetadata(loaded=False))
    assert assess_run(loaded).ok
    assert assess_run(bare).ok
    assert_pair_compatible(loaded, bare)


# ---------------------------------------------------------------- block concatenation
def test_concatenate_blocks(tmp_path, rng):
    """Each configuration is collected in several blocks under the ABBA schedule."""
    from franka_payload_id.pipeline import concatenate_runs

    paths = []
    for i in range(2):
        meta = RunMetadata(run_id=f"blk{i}", loaded=True, samples_per_period=50,
                           n_periods=4, sample_rate_hz=1000.0)
        save_run(tmp_path / f"blk{i}", _records(200, rng=rng), meta)
        paths.append(tmp_path / f"blk{i}")

    merged = concatenate_runs(paths)
    assert merged.n_samples == 400
    assert merged.meta.n_blocks == 2
    assert merged.meta.n_periods == 8


def test_concatenate_trims_ragged_blocks_to_whole_periods(tmp_path, rng, capsys):
    """Dropped frames leave blocks a few samples short; that must not be fatal.

    A drop costs one callback while the run still ends on the same wall-clock deadline,
    so raw sample counts differ slightly. What ABBA needs is equal *periods* per block,
    and the extra samples live in the trailing partial period that is discarded anyway.
    """
    from franka_payload_id.pipeline import concatenate_runs, periods_per_block

    # 4 periods of 50 samples, minus a handful of dropped frames.
    save_run(tmp_path / "a", _records(199, rng=rng),
             RunMetadata(loaded=True, samples_per_period=50, n_periods=4))
    save_run(tmp_path / "b", _records(187, rng=rng),
             RunMetadata(loaded=True, samples_per_period=50, n_periods=4))

    merged = concatenate_runs([tmp_path / "a", tmp_path / "b"])
    assert periods_per_block(merged) == 3          # floor(187/50) == 3 is the binding one
    assert merged.n_samples == 2 * 3 * 50
    assert merged.meta.n_blocks == 2
    assert "trimming each block" in capsys.readouterr().out


def test_concatenate_rejects_incompatible_blocks(tmp_path, rng):
    from franka_payload_id.pipeline import concatenate_runs

    save_run(tmp_path / "a", _records(200, rng=rng),
             RunMetadata(loaded=True, samples_per_period=50, n_periods=4))
    save_run(tmp_path / "c", _records(200, rng=rng),
             RunMetadata(loaded=False, samples_per_period=50, n_periods=4))
    with pytest.raises(ValueError, match="loaded and bare"):
        concatenate_runs([tmp_path / "a", tmp_path / "c"])

    # A block that does not even hold one whole period is a genuine failure.
    save_run(tmp_path / "d", _records(30, rng=rng),
             RunMetadata(loaded=True, samples_per_period=50, n_periods=1))
    with pytest.raises(ValueError, match="less than one whole period"):
        concatenate_runs([tmp_path / "a", tmp_path / "d"])


def test_trim_to_periods_per_block_preserves_block_layout(tmp_path, rng):
    from franka_payload_id.pipeline import (concatenate_runs, periods_per_block,
                                            trim_to_periods_per_block)

    for i in range(2):
        save_run(tmp_path / f"blk{i}", _records(200, rng=rng),
                 RunMetadata(loaded=True, samples_per_period=50, n_periods=4))
    merged = concatenate_runs([tmp_path / "blk0", tmp_path / "blk1"], verbose=False)
    assert periods_per_block(merged) == 4

    trimmed = trim_to_periods_per_block(merged, 2)
    assert periods_per_block(trimmed) == 2
    assert trimmed.n_samples == 2 * 2 * 50
    # The kept rows must be the FIRST two periods of each block, not the first four
    # periods of the concatenation -- otherwise one block would vanish entirely.
    np.testing.assert_allclose(trimmed.values[:100], merged.values[:100])
    np.testing.assert_allclose(trimmed.values[100:], merged.values[200:300])


def test_single_block_still_works(tmp_path, rng):
    from franka_payload_id.pipeline import concatenate_runs

    save_run(tmp_path / "solo", _records(200, rng=rng),
             RunMetadata(loaded=True, samples_per_period=50, n_periods=4))
    merged = concatenate_runs([tmp_path / "solo"])
    assert merged.n_samples == 200
    assert merged.meta.n_blocks == 1


def test_pair_compatibility_checks(rng):
    values = _records(100, rng=rng)
    loaded = RunLog(values, RunMetadata(loaded=True, samples_per_period=50, m_load=0.5))
    bare = RunLog(values, RunMetadata(loaded=False, samples_per_period=50))
    assert_pair_compatible(loaded, bare)

    with pytest.raises(ValueError, match="both runs are marked"):
        assert_pair_compatible(loaded, RunLog(values, RunMetadata(loaded=True,
                                                                 samples_per_period=50)))

    # A bare run must not claim to be carrying anything.
    bad_bare = RunLog(values, RunMetadata(loaded=False, samples_per_period=50, m_ee=0.7))
    with pytest.raises(ValueError, match="bare run declares a load"):
        assert_pair_compatible(loaded, bad_bare)

    # A loaded run that declares nothing means the tool was carried unmodelled.
    unmodelled = RunLog(values, RunMetadata(loaded=True, samples_per_period=50))
    with pytest.raises(ValueError, match="declares zero load"):
        assert_pair_compatible(unmodelled, bare)

    with pytest.raises(ValueError, match="different period lengths"):
        assert_pair_compatible(loaded, RunLog(values, RunMetadata(loaded=False,
                                                                 samples_per_period=20)))
