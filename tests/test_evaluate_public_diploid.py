import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from reticular_diploid.experiments.evaluate_public_diploid import bigram_baseline, unigram_baseline


def test_unigram_fits_training_counts_and_scores_content_and_eos_once():
    receipt = {"vocabulary_size": 3, "special_tokens": {"eos": 2}}
    canonical = {"target_ids": np.array([0, 0, 1, 1]), "target_indptr": np.array([0, 2, 4]),
                 "participant_split": np.array([0, 1])}
    metrics = unigram_baseline(receipt, canonical)
    # Training: token0 twice and EOS once, then Jeffreys smoothing per ID.
    expected = -2 * math.log(.5 / 4.5) - math.log(1.5 / 4.5)
    assert metrics["nll_nats"] == pytest.approx(expected)
    assert metrics["scored_targets"] == 3
    assert metrics["bits_per_person"] == pytest.approx(expected / math.log(2))
    with pytest.raises(ValueError):
        unigram_baseline(receipt, canonical, smoothing=0)


def test_bigram_owns_true_end_and_backs_off_for_unseen_training_contexts():
    receipt = {"vocabulary_size": 4, "special_tokens": {"eos": 2, "forward": 3}}
    canonical = {"target_ids": np.array([0, 0, 1, 1]), "target_indptr": np.array([0, 2, 4]),
                 "participant_split": np.array([0, 1])}
    metrics = bigram_baseline(receipt, canonical)
    # P(1|forward) uses the seen training prefix; token1 is an unseen context.
    expected = -math.log(10 * .1 / 11) - math.log(.1) - math.log(.3)
    assert metrics["nll_nats"] == pytest.approx(expected)
    assert metrics["scored_targets"] == 3
    with pytest.raises(ValueError):
        bigram_baseline(receipt, canonical, concentration=0)
    test = {**canonical, "participant_split": np.array([0, 2])}
    assert bigram_baseline(receipt, test, split=2) == metrics
    assert unigram_baseline(receipt, test, split=2) == unigram_baseline(receipt, canonical)


def test_prefix_causal_likelihood_counts_all_content_and_true_eos_once():
    import torch
    from torch import nn

    from reticular_diploid.experiments.evaluate_public_diploid import evaluate_causal

    class Constant(nn.Module):
        def forward(self, ids, coordinate_ids):
            return torch.zeros((*ids.shape, 20), device=ids.device)

    codec = SimpleNamespace(motif_entries=2, first=4, second=4,
                            positions=np.arange(16), content_size=14)
    ids = np.array([0] * 4 + [2, 10] + [0] * 6, np.int32)
    canonical = {"target_ids": ids, "target_indptr": np.array([0, 0, len(ids)]),
                 "target_coordinate_ids": np.arange(len(ids)), "participant_split": np.array([1, 1])}
    receipt = {"content_size": 14, "context": 8, "stride": 3,
               "special_tokens": {"pad": 14, "bos": 15, "eos": 17, "forward": 18}}
    result = evaluate_causal(Constant(), receipt, canonical, codec, torch.device("cpu"), batch_size=2)
    assert result["people"] == 2 and result["scored_targets"] == len(ids) + 2
    assert result["cross_entropy_nats"] == pytest.approx(math.log(20), rel=1e-7)
    assert result["bits_per_person"] == pytest.approx((len(ids) + 2) / 2 * math.log2(20), rel=1e-7)
    assert result["prefix_independence_canary"]["future_suffix_max_logit_difference"] == 0


@pytest.mark.parametrize("leak", ["future", "other_person"])
def test_actual_network_probe_rejects_future_and_cross_person_dependencies(leak):
    import torch
    from torch import nn

    from reticular_diploid.experiments.evaluate_public_diploid import verify_prefix_independence

    class Leaky(nn.Module):
        def forward(self, ids, coordinate_ids):
            if leak == "future":
                return ids.float().mean(1)[:, None, None].expand(-1, ids.shape[1], 3)
            return ids[:, 0].float().mean().expand(*ids.shape, 3)

    with pytest.raises(ValueError, match="future tokens or another"):
        verify_prefix_independence(Leaky(), {"context": 8, "content_size": 14}, torch.device("cpu"))


def test_curve_rescore_retains_selection_and_rejects_mixed_contracts(tmp_path, monkeypatch):
    import torch
    from torch import nn

    from reticular_diploid.experiments import evaluate_public_diploid as evaluation
    from reticular_diploid.models import public_kimi

    class Constant(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

    receipt = {"public_data_only": True}
    contract = {"schema_version": "reticular-public-ntp/v1", "corpus": receipt,
                "model_config": {"vocabulary_size": 20}}
    source = tmp_path / "checkpoints"
    source.mkdir()
    (source / "contract.json").write_text(json.dumps(contract))
    monkeypatch.setattr(evaluation, "clean_commit", lambda: "synthetic")
    monkeypatch.setattr(evaluation, "load_corpus", lambda _: (receipt, {}, {}))
    monkeypatch.setattr(evaluation.FrozenDiploidTokenizer, "load", lambda *a: None)
    evaluated_splits = []

    def score(*args, split=1):
        evaluated_splits.append(split)
        return {"bits_per_person": 3.}

    monkeypatch.setattr(evaluation, "evaluate_causal", score)
    monkeypatch.setattr(evaluation, "unigram_baseline", lambda *a: {})
    monkeypatch.setattr(evaluation, "bigram_baseline", lambda *a: {})
    monkeypatch.setattr(public_kimi, "NoPECausalTransformer", Constant)
    receipt["tokenizer_sha256"] = "tokenizer"
    for step in [2, 200]:
        torch.save({"step": step, "exposure": step * 4, "contract": contract,
                    "model": Constant(None).state_dict()}, source / f"step-{step:06d}.pt")
    selected = {"path": "step-000002.pt", "sha256": evaluation.sha256(source / "step-000002.pt"), "step": 2}
    (source / "selected-checkpoint.json").write_text(json.dumps(selected))
    # Persist the complete contract after receipt binding above.
    (source / "contract.json").write_text(json.dumps(contract))
    args = SimpleNamespace(output=tmp_path / "curve.json", tokenizer=tmp_path / "vocab.npz",
                           ownership="prefix_causal", corpus=tmp_path, checkpoint_directory=source,
                           device="cpu", batch_size=2, training_diagnostics=True)
    evaluation.rescore_curve(args)
    result = json.loads(args.output.read_text())
    assert result["frozen_selection"] == selected
    assert [row["target_tokens_seen"] for row in result["records"]] == [8, 800]
    assert evaluated_splits == [1, 0, 1, 0]
    assert all(row["training"] == row["validation"] for row in result["records"])
    broken = torch.load(source / "step-000200.pt", weights_only=True)
    broken["contract"]["recipe_sha256"] = "another-recipe"
    torch.save(broken, source / "step-000200.pt")
    args.output = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="contract or step"):
        evaluation.rescore_curve(args)
    assert not args.output.exists()


def test_causal_precision_restores_backend_flags_even_on_failure():
    import torch

    from reticular_diploid.experiments.evaluate_public_diploid import causal_precision

    before = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        with pytest.raises(RuntimeError, match="synthetic failure"):
            with causal_precision(torch.device("cuda")):
                assert not torch.backends.cuda.matmul.allow_tf32
                assert not torch.backends.cudnn.allow_tf32
                raise RuntimeError("synthetic failure")
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before
