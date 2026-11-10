import pytest

from reticular_diploid.experiments.benchmark_public_diploid import summarize


def test_summary_counts_all_tokens_and_keeps_framing_separate():
    result = summarize([2, 4], [0, 2], [4, 8])
    assert result["mean"] == result["median"] == 3
    assert result["p95"] == pytest.approx(3.9)
    assert result["maximum"] == 4
    assert result["motif_token_fraction"] == pytest.approx(1 / 3)
    assert result["residual_token_fraction"] == pytest.approx(2 / 3)
    assert result["residual_only_mean"] == 6
    assert result["framing_tokens_per_person"] == 3
    assert summarize([0], [0], [0])["motif_token_fraction"] is None
    with pytest.raises(ValueError):
        summarize([], [], [])
