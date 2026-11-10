import copy
import itertools
import json

import pytest

from reticular_diploid.analysis.public_distillation_figures import collect_learning_curves, sample_efficiency


def test_sample_figures_keep_negative_results_and_require_complete_seed_grid():
    data = {"schema_version": "reticular-public-final-evaluation/v1", "records": [
        {"effect_variants": size, "training_people": count, "seed": seed, "method": method,
         "metrics": {"r_squared": -.2 if seed == 7 else .1}}
        for size, count, seed, method in itertools.product([16, 128, 1024], [32, 128, 400],
                                                          [7, 8], ["scratch", "pretrained", "joint"])]}
    rows = sample_efficiency(data)
    assert len(rows) == 27
    assert all(row["minimum"] == -.2 and row["maximum"] == .1 for row in rows)
    missing = copy.deepcopy(data)
    missing["records"].pop()
    with pytest.raises(ValueError, match="omits or duplicates"):
        sample_efficiency(missing)
    data["records"].append(data["records"][0])
    with pytest.raises(ValueError, match="omits or duplicates"):
        sample_efficiency(data)


def test_curve_export_removes_minibatch_logs_and_binds_selected_checkpoint(tmp_path):
    root = tmp_path / "A1-pretrained-n400-s7"
    root.mkdir()
    (root / "contract.json").write_text("{}")
    selection = root / "selected-checkpoint.json"
    selection.write_text(json.dumps({"sha256": "chosen"}))
    point = {"step": 1000, "person_draws": 2000, "region_targets_seen": 2400000,
             "total_worker_seconds": 190., "peak_allocated_bytes": 1000,
             "validation": {"r_squared": .2, "mse": .8, "pearson_r": .5, "people": 55},
             "training_batch_mse": .123, "training_batch_ntp_ce": 3.4}
    (root / "learning-curve.jsonl").write_text(json.dumps(point) + "\n")
    model = {"id": root.name, "effect_variants": 16, "training_people": 400,
             "seed": 7, "arm": "pretrained", "checkpoint_sha256": "chosen"}
    exported = collect_learning_curves([tmp_path], {"records": [model]})
    result = exported["records"][0]["points"][0]
    assert set(result) == set(point) - {"training_batch_mse", "training_batch_ntp_ce"}
    assert result["validation"] == point["validation"]
    selection.write_text(json.dumps({"sha256": "another"}))
    with pytest.raises(ValueError, match="another selected"):
        collect_learning_curves([tmp_path], {"records": [model]})
