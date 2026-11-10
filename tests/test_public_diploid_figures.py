import json

import pytest

from reticular_diploid.analysis.public_diploid_figures import load_causal_curve


def test_figure_refuses_legacy_window_scores(tmp_path):
    path = tmp_path / "curve.json"
    receipt = {"validation_people": 55, "canonical_sha256": "fixed"}
    data = {"schema_version": "reticular-public-ntp-curve/v1", "locked_test_used": False,
            "training_contract": {"corpus": receipt}, "records": [{"validation": {"people": 55}}],
            "evaluation_ownership": "legacy"}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="prefix-only"):
        load_causal_curve(path, receipt)
    data["evaluation_ownership"] = "prefix_causal"
    path.write_text(json.dumps(data))
    assert load_causal_curve(path, receipt)[1] == data["records"]
    data["records"][0]["validation"]["people"] = 54
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="complete validation"):
        load_causal_curve(path, receipt)
