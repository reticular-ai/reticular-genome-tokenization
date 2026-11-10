import pytest

from reticular_diploid.experiments.evaluate_public_distillation import verify_contract


def test_reload_accepts_identical_reproduction_but_not_changed_teacher_or_data():
    corpus = {"canonical_sha256": "a", "code_commit": "first"}
    teachers = {"scores_sha256": "b", "code_commit": "first", "normalization": "training only",
                "teachers": [{"effect_variants": 16, "training_mean": .1, "training_std": .2,
                              "ld_mean_r_squared": .123}]}
    contract = {"schema_version": "reticular-public-distillation/v1", "corpus": corpus,
                "ntp_parent": {"schema_version": "reticular-public-ntp/v1", "corpus": corpus},
                "teachers": teachers}
    verify_contract(contract, {**corpus, "code_commit": "rebuilt"}, {**teachers, "code_commit": "rebuilt"})
    verify_contract(contract, corpus, {**teachers, "teachers": [{**teachers["teachers"][0], "ld_mean_r_squared": .124}]})
    with pytest.raises(ValueError, match="teacher"):
        verify_contract(contract, corpus, {**teachers, "scores_sha256": "other"})
    with pytest.raises(ValueError, match="corpus"):
        verify_contract(contract, {**corpus, "canonical_sha256": "other"}, teachers)
    with pytest.raises(ValueError, match="teacher"):
        verify_contract(contract, corpus, {**teachers, "teachers": [{**teachers["teachers"][0], "training_std": .3}]})
