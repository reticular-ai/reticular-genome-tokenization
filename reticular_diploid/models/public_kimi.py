"""Geometry-free public Kimi decoder, preserving native parameter names and RNG.

ModelConfig, coordinate/Engram components and enabled initialization operations
are ported from the verified native decoder. No graph construction, vocabulary
fitting, operational data loading or controlled artifacts are included here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reticular_diploid.models.kimi import KDAHybridEncoder


@dataclass(frozen=True)
class ModelConfig:
    vocabulary_size: int
    d_model: int = 64
    layers: int = 2
    heads: int = 4
    mlp_width: int = 256
    dropout: float = 0.0
    position_encoding: str = "none"
    maximum_position_embeddings: int = 4096
    coordinate_vocabulary_size: int = 0
    coordinate_factor_width: int = 0
    genetic_distance_bias: bool = False
    genetic_distance_min_half_life_cm: float = 0.01
    genetic_distance_max_half_life_cm: float = 16.0
    grg_feature_dim: int = 0
    grg_feature_artifact_sha256: str = ""
    grg_relative_dim: int = 0
    grg_relative_geometry: str = "none"
    grg_relative_distance_scale: float = 1.0
    grg_relative_artifact_sha256: str = ""
    grg_relative_mode: str = "scalar_distance"
    grg_relative_liere_planes: int = 8
    grg_output_kernel: bool = False
    grg_output_geometry: str = "lorentz"
    grg_output_preserve_known_mass: bool = False
    token_mixer: str = "dense"
    hybrid_global_mixer: str = "dense"
    kda_mode: str = "chunk"
    kda_head_dim: int = 128
    kda_expand_v: float = 1.0
    kda_conv_size: int = 4
    kda_use_short_conv: bool = True
    attention_residual_block_size: int = 0
    moe_routed_experts: int = 0
    moe_top_k: int = 2
    moe_latent_dim: int = 192
    moe_expert_hidden_dim: int = 304
    moe_shared_experts: int = 2
    moe_first_dense_layers: int = 1
    moe_quantile_balancing: bool = True
    engram_max_ngram: int = 0
    # Optional stable logical vocabulary used only for Engram hashes when
    # physical embedding rows are reused across blocks.
    engram_key_vocabulary_size: int = 0
    engram_heads_per_ngram: int = 2
    engram_head_dim: int = 16
    engram_table_size: int = 131071
    engram_layer: int = 1
    engram_conv_kernel: int = 4
    engram_seed: int = 0
    engram_raw_orders: tuple[int, ...] = ()
    engram_semantic_orders: tuple[int, ...] = ()
    engram_injection_layers: tuple[int, ...] = ()
    engram_embedding_budget: int = 0
    engram_conv_dilation: int = 0
    engram_grg_count: int = 0
    engram_residual_first_namespace: int = 0
    engram_residual_second_namespace: int = 0
    engram_variant_count: int = 0


class FactorizedCoordinateEmbedding(nn.Module):
    """Sum a coarse-row embedding and a shared within-row embedding."""

    def __init__(self, vocabulary_size: int, factor_width: int, width: int) -> None:
        super().__init__()
        if factor_width <= 0 or vocabulary_size % factor_width:
            raise ValueError("Coordinate factor width must divide its vocabulary")
        self.factor_width = factor_width
        self.coarse = nn.Embedding(vocabulary_size // factor_width, width)
        self.fine = nn.Embedding(factor_width, width)

    def forward(self, coordinate_ids: torch.Tensor) -> torch.Tensor:
        return self.coarse(coordinate_ids // self.factor_width) + self.fine(
            coordinate_ids % self.factor_width
        )


def _is_prime(value: int) -> bool:
    """Return whether a small positive integer is prime."""

    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _distinct_primes_at_least(start: int, count: int) -> tuple[int, ...]:
    """Return deterministic distinct table sizes, matching Engram's prime moduli."""

    if start < 2 or count <= 0:
        raise ValueError("Engram table size and table count must be positive")
    values: list[int] = []
    candidate = start
    while len(values) < count:
        if _is_prime(candidate):
            values.append(candidate)
        candidate += 1
    return tuple(values)


class EngramMemory(nn.Module):
    """DeepSeek-style hashed suffix N-gram memory for a single residual stream."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.engram_max_ngram < 2:
            raise ValueError("Enabled Engram requires max N-gram order >= 2")
        if config.engram_heads_per_ngram <= 0 or config.engram_head_dim <= 0:
            raise ValueError("Engram hash-head count and width must be positive")
        if config.engram_table_size < 2 or config.engram_conv_kernel <= 0:
            raise ValueError("Engram table and convolution sizes must be positive")

        self.max_ngram = config.engram_max_ngram
        self.heads_per_ngram = config.engram_heads_per_ngram
        self.head_dim = config.engram_head_dim
        key_vocabulary_size = config.engram_key_vocabulary_size or config.vocabulary_size
        self.pad_id = key_vocabulary_size
        self.orders = tuple(range(2, self.max_ngram + 1))
        table_count = len(self.orders) * self.heads_per_ngram
        table_sizes = _distinct_primes_at_least(config.engram_table_size, table_count)
        offsets = np.zeros(table_count, dtype=np.int64)
        if table_count > 1:
            offsets[1:] = np.cumsum(np.asarray(table_sizes[:-1], dtype=np.int64), dtype=np.int64)
        self.register_buffer("table_sizes", torch.tensor(table_sizes, dtype=torch.long))
        self.register_buffer("table_offsets", torch.from_numpy(offsets))

        # Multipliers are deterministic addressing metadata, not learned state.
        # Bound them so every token product remains in signed int64.
        generator = np.random.default_rng(config.engram_seed + 10_007 * config.engram_layer)
        max_multiplier = np.iinfo(np.int64).max // (key_vocabulary_size + 1)
        half_bound = max(2, max_multiplier // 2)
        multipliers = (
            generator.integers(
                0,
                half_bound,
                size=self.max_ngram,
                dtype=np.int64,
            )
            * 2
            + 1
        )
        self.register_buffer("hash_multipliers", torch.from_numpy(multipliers.astype(np.int64)))

        self.embedding = nn.Embedding(int(sum(table_sizes)), self.head_dim)
        self.memory_width = table_count * self.head_dim
        self.key_projection = nn.Linear(self.memory_width, config.d_model, bias=True)
        self.value_projection = nn.Linear(self.memory_width, config.d_model, bias=True)
        self.key_norm = nn.RMSNorm(config.d_model)
        self.query_norm = nn.RMSNorm(config.d_model)
        self.conv_norm = nn.RMSNorm(config.d_model)
        dilation = self.max_ngram
        self.short_conv = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=config.engram_conv_kernel,
            groups=config.d_model,
            bias=False,
            padding=(config.engram_conv_kernel - 1) * dilation,
            dilation=dilation,
        )

    def zero_initialize_short_conv(self) -> None:
        """Use the zero convolution initialization reported by DeepSeek."""

        nn.init.zeros_(self.short_conv.weight)

    def hash_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return deterministic suffix hashes with shape [batch, length, tables]."""

        if input_ids.ndim != 2:
            raise ValueError("Engram input IDs must have shape [batch, length]")
        _, length = input_ids.shape
        shifted = [input_ids]
        for lag in range(1, self.max_ngram):
            values = torch.full_like(input_ids, self.pad_id)
            if lag < length:
                values[:, lag:] = input_ids[:, :-lag]
            shifted.append(values)

        hashes = []
        table_index = 0
        for order in self.orders:
            mixed = shifted[0] * self.hash_multipliers[0]
            for lag in range(1, order):
                mixed = torch.bitwise_xor(mixed, shifted[lag] * self.hash_multipliers[lag])
            for _ in range(self.heads_per_ngram):
                hashes.append(
                    torch.remainder(mixed, self.table_sizes[table_index])
                    + self.table_offsets[table_index]
                )
                table_index += 1
        return torch.stack(hashes, dim=-1)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[:2] != input_ids.shape:
            raise ValueError("Engram hidden states and token IDs must align")
        memory = self.embedding(self.hash_ids(input_ids)).flatten(start_dim=-2)
        key = self.key_norm(self.key_projection(memory))
        query = self.query_norm(hidden_states)
        score = (key * query).sum(dim=-1) / math.sqrt(hidden_states.shape[-1])
        score = score.abs().clamp_min(1e-6).sqrt() * score.sign()
        gate = torch.sigmoid(score).unsqueeze(-1)
        value = gate * self.value_projection(memory)

        convolved = self.short_conv(self.conv_norm(value).transpose(1, 2))[
            ..., : value.shape[1]
        ].transpose(1, 2)
        return value + F.silu(convolved)

    @property
    def embedding_parameters(self) -> int:
        return self.embedding.weight.numel()

    @property
    def active_parameters_per_token(self) -> int:
        dense = sum(parameter.numel() for parameter in self.parameters())
        return dense - self.embedding_parameters + self.memory_width


class NoPECausalTransformer(nn.Module):
    """The tested zero-dropout, ALiBi-long KDA/MoE/Engram public ablation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if (config.token_mixer != "kda_hybrid" or config.hybrid_global_mixer != "dense"
                or config.position_encoding != "alibi_long" or config.dropout != 0):
            raise ValueError("Public decoder requires zero-dropout dense/KDA hybrid with ALiBi-long")
        if (config.genetic_distance_bias or config.grg_feature_dim or config.grg_relative_dim
                or config.grg_relative_geometry != "none" or config.grg_output_kernel
                or config.engram_raw_orders or config.engram_semantic_orders):
            raise ValueError("Public decoder is a geometry-free ablation without graph or structured memory features")
        if (config.heads <= 0 or config.d_model % config.heads or config.kda_head_dim <= 0
                or config.d_model % config.kda_head_dim or config.maximum_position_embeddings <= 0):
            raise ValueError("Public decoder dimensions are incompatible")
        if config.engram_max_ngram and not 0 <= config.engram_layer < config.layers:
            raise ValueError("Engram injection layer is outside the stack")
        self.config = config
        self.embedding = nn.Embedding(config.vocabulary_size, config.d_model)
        self.position_embedding = None
        exponents = -8.0 * (torch.arange(config.heads, dtype=torch.float32) + 1) / config.heads - 3.0
        self.register_buffer("alibi_slopes", torch.pow(2.0, exponents))
        torch.backends.mha.set_fastpath_enabled(False)
        self.coordinate_embedding = None
        if config.coordinate_vocabulary_size:
            self.coordinate_embedding = (
                FactorizedCoordinateEmbedding(config.coordinate_vocabulary_size,
                                              config.coordinate_factor_width, config.d_model)
                if config.coordinate_factor_width
                else nn.Embedding(config.coordinate_vocabulary_size, config.d_model))
        self.blocks = KDAHybridEncoder(
            d_model=config.d_model, layers=config.layers, dense_heads=config.heads,
            mlp_width=config.mlp_width, dropout=config.dropout, kda_head_dim=config.kda_head_dim,
            kda_expand_v=config.kda_expand_v, kda_mode=config.kda_mode,
            kda_conv_size=config.kda_conv_size, kda_use_short_conv=config.kda_use_short_conv,
            global_mixer=config.hybrid_global_mixer,
            attention_residual_block_size=config.attention_residual_block_size,
            moe_routed_experts=config.moe_routed_experts, moe_top_k=config.moe_top_k,
            moe_latent_dim=config.moe_latent_dim, moe_expert_hidden_dim=config.moe_expert_hidden_dim,
            moe_shared_experts=config.moe_shared_experts, moe_first_dense_layers=config.moe_first_dense_layers,
            moe_quantile_balancing=config.moe_quantile_balancing)
        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocabulary_size, bias=False)
        self.apply(self._initialize)
        self.blocks.drop_replaced_feed_forwards()
        if isinstance(self.coordinate_embedding, FactorizedCoordinateEmbedding):
            nn.init.zeros_(self.coordinate_embedding.fine.weight)
        self.output.weight = self.embedding.weight
        self.engram = None
        if config.engram_max_ngram:
            rng_state = torch.get_rng_state()
            self.engram = EngramMemory(config)
            self.engram.apply(self._initialize)
            self.engram.zero_initialize_short_conv()
            torch.set_rng_state(rng_state)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Embedding | nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _engram_layer_residual(self, layer_index, hidden, ids):
        if self.engram is not None and layer_index == self.config.engram_layer:
            return self.engram(hidden, ids)
        return hidden.new_zeros(())

    def forward_hidden(self, input_ids, padding_mask=None, coordinate_ids=None):
        hidden = self.embedding(input_ids)
        if self.coordinate_embedding is None:
            if coordinate_ids is not None:
                raise ValueError("Coordinates supplied to a model without coordinate embeddings")
        else:
            if coordinate_ids is None or coordinate_ids.shape != input_ids.shape:
                raise ValueError("Coordinate IDs are required and must match token IDs")
            hidden = hidden + self.coordinate_embedding(coordinate_ids)
        options = {"alibi_slopes": self.alibi_slopes}
        if self.engram is not None:
            options["layer_residual"] = lambda index, values: self._engram_layer_residual(index, values, input_ids)
        hidden = self.blocks(hidden, mask=None, src_key_padding_mask=padding_mask, **options)
        return self.norm(hidden)

    def forward(self, input_ids, padding_mask=None, coordinate_ids=None):
        return self.output(self.forward_hidden(input_ids, padding_mask, coordinate_ids))
