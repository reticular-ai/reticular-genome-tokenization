import pytest
from reticular_diploid.models.public_kimi import ModelConfig, NoPECausalTransformer

def test_geometry_features_cannot_silently_enter_public_ablation():
    with pytest.raises(ValueError, match="geometry-free"):
        NoPECausalTransformer(ModelConfig(vocabulary_size=64, grg_feature_dim=4, token_mixer="kda_hybrid", position_encoding="alibi_long"))
