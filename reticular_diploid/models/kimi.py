"""Minimal Kimi Delta Attention hybrid used by the existing causal LM."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention
from torch.utils.checkpoint import checkpoint

from reticular_diploid.models.kimi_mla import K3GatedMLA
from reticular_diploid.models.kimi_moe import K3StableLatentMoE

_compiled_flex_attention = torch.compile(flex_attention, fullgraph=True, dynamic=True)


def _causal_alibi_attention(query, key, value, slopes):
    def score_mod(score, batch, head, query_index, key_index):
        relative_bias = slopes[head].float() * (query_index - key_index).float()
        return torch.where(query_index >= key_index, score - relative_bias, float("-inf"))

    # ponytail: eager CPU is a small correctness reference; full lengths use CUDA.
    attention = _compiled_flex_attention if query.is_cuda else flex_attention
    return attention(
        query, key, value, score_mod=score_mod, block_mask=None,
        kernel_options={"BACKEND": "TRITON"},
    )


def hybrid_dense_layer_indices(layers: int) -> tuple[int, ...]:
    """Evenly place the nearest one-in-four dense layers, ending dense."""

    if layers <= 0:
        raise ValueError("layers must be positive")
    dense_count = max(1, math.floor(layers / 4.0 + 0.5))
    return tuple(math.ceil((index + 1) * layers / dense_count) - 1 for index in range(dense_count))


class AttentionResidualRouter(nn.Module):
    """Kimi rank-one softmax routing over completed depth blocks plus a prefix."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=eps)
        self.projection = nn.Linear(d_model, 1, bias=False)

    def probabilities(
        self,
        current: torch.Tensor,
        completed_blocks: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.stack((*completed_blocks, current), dim=2)
        values_float = values.float()
        variance = values_float.square().mean(dim=-1, keepdim=True)
        keys = values_float * torch.rsqrt(variance + self.norm.eps)
        score_weight = self.norm.weight.float() * self.projection.weight.squeeze(0).float()
        probabilities = torch.softmax((keys * score_weight).sum(dim=-1), dim=-1)
        return values, probabilities

    def forward(
        self,
        current: torch.Tensor,
        completed_blocks: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        values, probabilities = self.probabilities(current, completed_blocks)
        routed = torch.einsum("bln,blnd->bld", probabilities, values.float())
        return routed.to(dtype=values.dtype)


class KDAEncoderLayer(nn.Module):
    """Pre-norm KDA mixer plus the incumbent Transformer feed-forward block."""

    def __init__(
        self,
        *,
        d_model: int,
        mlp_width: int,
        dropout: float,
        head_dim: int,
        expand_v: float,
        mode: str,
        conv_size: int,
        use_short_conv: bool,
        layer_idx: int,
    ) -> None:
        super().__init__()
        try:
            from fla.layers import KimiDeltaAttention
        except Exception as error:
            raise RuntimeError(
                "KDA requires flash-linear-attention==0.5.2 and its dependencies"
            ) from error
        if d_model % head_dim:
            raise ValueError("KDA head_dim must divide d_model")
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = KimiDeltaAttention(
            hidden_size=d_model,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=d_model // head_dim,
            mode=mode,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            layer_idx=layer_idx,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, mlp_width)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_width, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def token_mixing_update(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        valid_tokens = None if padding_mask is None else (~padding_mask).to(dtype=torch.long)
        mixed, _, _ = self.mixer(
            self.norm1(hidden),
            attention_mask=valid_tokens,
        )
        return self.dropout1(mixed)

    def feed_forward_update(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.norm2(hidden)
        return self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(normalized)))))

    def forward(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden = hidden + self.token_mixing_update(hidden, padding_mask)
        hidden = hidden + self.feed_forward_update(hidden)
        return hidden


class KDAHybridEncoder(nn.Module):
    """Drop-in hybrid stack with optional K3 Stable LatentMoE channel mixers."""

    def __init__(
        self,
        *,
        d_model: int,
        layers: int,
        dense_heads: int,
        mlp_width: int,
        dropout: float,
        kda_head_dim: int,
        kda_expand_v: float,
        kda_mode: str,
        kda_conv_size: int,
        kda_use_short_conv: bool,
        global_mixer: str = "dense",
        attention_residual_block_size: int = 0,
        moe_routed_experts: int = 0,
        moe_top_k: int = 2,
        moe_latent_dim: int = 192,
        moe_expert_hidden_dim: int = 304,
        moe_shared_experts: int = 2,
        moe_first_dense_layers: int = 1,
        moe_quantile_balancing: bool = True,
    ) -> None:
        super().__init__()
        if attention_residual_block_size < 0:
            raise ValueError("attention_residual_block_size cannot be negative")
        if global_mixer not in {"dense", "mla"}:
            raise ValueError("global_mixer must be dense or mla")
        if not 0 <= moe_first_dense_layers <= layers:
            raise ValueError("moe_first_dense_layers must lie within the stack")
        if moe_routed_experts and not 1 <= moe_top_k < moe_routed_experts:
            raise ValueError("MoE top-k must be smaller than routed expert count")
        dense_indices = set(hybrid_dense_layer_indices(layers))
        modules: list[nn.Module] = []
        pattern: list[str] = []
        for index in range(layers):
            if index in dense_indices:
                modules.append(
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=dense_heads,
                        dim_feedforward=mlp_width,
                        dropout=dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                )
                pattern.append("dense")
            else:
                modules.append(
                    KDAEncoderLayer(
                        d_model=d_model,
                        mlp_width=mlp_width,
                        dropout=dropout,
                        head_dim=kda_head_dim,
                        expand_v=kda_expand_v,
                        mode=kda_mode,
                        conv_size=kda_conv_size,
                        use_short_conv=kda_use_short_conv,
                        layer_idx=index,
                    )
                )
                pattern.append("kda")
        self.layers = nn.ModuleList(modules)
        self.pattern = tuple(
            "mla" if global_mixer == "mla" and kind == "dense" else kind for kind in pattern
        )
        self.global_mixer = global_mixer
        self.attention_residual_block_size = attention_residual_block_size
        if attention_residual_block_size:
            # Router constructors consume CPU RNG before the parent applies its
            # controlled initialization. Restore it so every incumbent KDA/FFN
            # parameter remains seed-identical to the no-AttnRes model.
            rng_state = torch.get_rng_state()
            self.attention_residual_attention_routers = nn.ModuleList(
                AttentionResidualRouter(d_model) for _ in range(layers)
            )
            self.attention_residual_mlp_routers = nn.ModuleList(
                AttentionResidualRouter(d_model) for _ in range(layers)
            )
            self.attention_residual_output_router: AttentionResidualRouter | None = (
                AttentionResidualRouter(d_model)
            )
            torch.set_rng_state(rng_state)
        else:
            self.attention_residual_attention_routers = nn.ModuleList()
            self.attention_residual_mlp_routers = nn.ModuleList()
            self.attention_residual_output_router = None

        self.channel_mixer_pattern = tuple(
            "k3_moe" if moe_routed_experts and index >= moe_first_dense_layers else "dense"
            for index in range(layers)
        )
        # Construct these after all incumbent layers and AttnRes routers. The
        # restored RNG plus the parent's controlled initialization keeps every
        # common parameter seed-identical to the incumbent model.
        rng_state = torch.get_rng_state()
        if moe_routed_experts:
            channel_mixers: list[nn.Module] = []
            for kind in self.channel_mixer_pattern:
                channel_mixers.append(
                    K3StableLatentMoE(
                        d_model=d_model,
                        latent_dim=moe_latent_dim,
                        hidden_dim=moe_expert_hidden_dim,
                        num_routed_experts=moe_routed_experts,
                        top_k=moe_top_k,
                        num_shared_experts=moe_shared_experts,
                        quantile_balancing=moe_quantile_balancing,
                    )
                    if kind == "k3_moe"
                    else nn.Identity()
                )
            self.channel_mixers = nn.ModuleList(channel_mixers)
        else:
            self.channel_mixers = nn.ModuleList(nn.Identity() for _ in range(layers))
        self.channel_dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(layers))
        torch.set_rng_state(rng_state)

        # Build MLA after every incumbent module so the default dense path and
        # all parameters shared with it remain seed-identical.
        rng_state = torch.get_rng_state()
        if global_mixer == "mla":
            self.mla_mixers = nn.ModuleList(
                K3GatedMLA(
                    d_model=d_model,
                    heads=dense_heads,
                    latent_dim=d_model // 2,
                    q_head_dim=d_model // dense_heads,
                    dropout=dropout,
                )
                if index in dense_indices
                else nn.Identity()
                for index in range(layers)
            )
        else:
            self.mla_mixers = nn.ModuleList(nn.Identity() for _ in range(layers))
        torch.set_rng_state(rng_state)

    @staticmethod
    def _dense_token_mixing_update(
        layer: nn.TransformerEncoderLayer,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
        alibi_slopes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = layer.norm1(hidden)
        if alibi_slopes is None:
            attended = layer.self_attn(
                normalized, normalized, normalized,
                attn_mask=mask, key_padding_mask=padding_mask, need_weights=False,
            )[0]
        else:
            attention = layer.self_attn
            if mask is not None or padding_mask is not None or attention.dropout:
                raise ValueError("Compiled ALiBi requires zero dropout and no additional dense mask")
            if alibi_slopes.shape != (attention.num_heads,):
                raise ValueError("ALiBi slopes must match the attention heads")
            batch, length, width = normalized.shape
            projected = F.linear(normalized, attention.in_proj_weight, attention.in_proj_bias)
            query, key, value = (
                part.reshape(batch, length, attention.num_heads, width // attention.num_heads)
                .transpose(1, 2)
                for part in projected.chunk(3, dim=-1)
            )
            attended = _causal_alibi_attention(query, key, value, alibi_slopes)
            attended = attended.transpose(1, 2).reshape(batch, length, width)
            attended = F.linear(attended, attention.out_proj.weight, attention.out_proj.bias)
        return layer.dropout1(attended)

    @staticmethod
    def _dense_feed_forward_update(
        layer: nn.TransformerEncoderLayer,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        normalized = layer.norm2(hidden)
        return layer.dropout2(
            layer.linear2(layer.dropout(layer.activation(layer.linear1(normalized))))
        )

    def _mla_token_mixing_update(
        self,
        index: int,
        layer: nn.TransformerEncoderLayer,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return layer.dropout1(self.mla_mixers[index](layer.norm1(hidden), mask, padding_mask))

    def _channel_mixing_update(
        self,
        index: int,
        kind: str,
        layer: nn.Module,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.channel_mixer_pattern[index] == "k3_moe":
            normalized = layer.norm2(hidden)
            update = self.channel_mixers[index](normalized, padding_mask)
            return self.channel_dropouts[index](update)
        if kind in {"dense", "mla"}:
            return self._dense_feed_forward_update(layer, hidden)
        return layer.feed_forward_update(hidden)

    def drop_replaced_feed_forwards(self) -> None:
        """Release incumbent FFN weights after seed-matched initialization."""

        for index, layer in enumerate(self.layers):
            if self.pattern[index] == "mla":
                layer.self_attn = nn.Identity()
            if self.channel_mixer_pattern[index] != "k3_moe":
                continue
            layer.linear1 = nn.Identity()
            layer.dropout = nn.Identity()
            layer.linear2 = nn.Identity()
            layer.dropout2 = nn.Identity()

    def attention_mixing_metadata(self) -> dict[str, object]:
        mla_layers = [
            mixer
            for kind, mixer in zip(self.pattern, self.mla_mixers, strict=True)
            if kind == "mla"
        ]
        return {
            "layer_pattern": list(self.pattern),
            "global_mixer": self.global_mixer,
            "mla": None
            if not mla_layers
            else {
                "name": "kimi_k3_gated_mla",
                "layers": len(mla_layers),
                "geometry": mla_layers[0].diagnostics(),
            },
        }

    def channel_mixing_metadata(self) -> dict[str, object]:
        moe_layers = [
            mixer
            for kind, mixer in zip(self.channel_mixer_pattern, self.channel_mixers, strict=True)
            if kind == "k3_moe"
        ]
        if not moe_layers:
            return {
                "name": "dense_gelu_ffn",
                "layer_pattern": list(self.channel_mixer_pattern),
            }
        example = moe_layers[0]
        return {
            "name": "kimi_k3_stable_latent_moe",
            "layer_pattern": list(self.channel_mixer_pattern),
            "moe_layers": len(moe_layers),
            "dense_layers": len(self.layers) - len(moe_layers),
            "routed_experts_per_layer": example.num_routed_experts,
            "shared_experts_per_layer": example.num_shared_experts,
            "top_k": example.top_k,
            "latent_dim": example.latent_dim,
            "expert_hidden_dim": example.hidden_dim,
            "quantile_balancing": example.quantile_balancing,
            "active_parameters_per_moe_token": (example.active_parameters_per_token),
            "last_layer_diagnostics": moe_layers[-1].diagnostics(),
        }

    def attention_residual_source_counts(self) -> dict[str, object]:
        if not self.attention_residual_block_size:
            return {"attention": [], "mlp": [], "output": 0}
        completed = 0
        attention: list[int] = []
        mlp: list[int] = []
        for index in range(len(self.layers)):
            attention.append(completed + 1)
            if index % self.attention_residual_block_size == 0:
                completed += 1
            mlp.append(completed + 1)
        return {
            "attention": attention,
            "mlp": mlp,
            "output": completed + 1,
        }

    def _activation_call(self, function, *args):
        if getattr(self, "activation_checkpointing", False) and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def _forward_attention_residual(
        self,
        src: torch.Tensor,
        mask: torch.Tensor | None,
        src_key_padding_mask: torch.Tensor | None,
        dense_padding_mask: torch.Tensor | None,
        layer_residual: Callable[[int, torch.Tensor], torch.Tensor] | None,
        alibi_slopes: torch.Tensor | None,
        dense_attention_update: Callable | None,
    ) -> torch.Tensor:
        completed_blocks: list[torch.Tensor] = []
        prefix_sum: torch.Tensor | None = src
        for index, (kind, layer) in enumerate(zip(self.pattern, self.layers, strict=True)):
            if prefix_sum is None:
                raise RuntimeError("Attention Residual prefix is unexpectedly empty")
            if layer_residual is not None:
                prefix_sum = prefix_sum + layer_residual(index, prefix_sum)
            hidden = prefix_sum
            if completed_blocks:
                hidden = self.attention_residual_attention_routers[index](
                    prefix_sum, completed_blocks
                )
            if index % self.attention_residual_block_size == 0:
                completed_blocks.append(prefix_sum)
                prefix_sum = None

            if kind == "dense":
                update = (dense_attention_update(index, layer, hidden)
                          if dense_attention_update is not None else
                          self._activation_call(
                              self._dense_token_mixing_update,
                              layer, hidden, mask, dense_padding_mask, alibi_slopes))
            elif kind == "mla":
                update = self._mla_token_mixing_update(
                    index, layer, hidden, mask, dense_padding_mask
                )
            else:
                update = self._activation_call(
                    layer.token_mixing_update, hidden, src_key_padding_mask)
            prefix_sum = update if prefix_sum is None else prefix_sum + update

            hidden = self.attention_residual_mlp_routers[index](prefix_sum, completed_blocks)
            update = self._activation_call(
                partial(self._channel_mixing_update, index, kind, layer),
                hidden, src_key_padding_mask,
            )
            prefix_sum = prefix_sum + update

        if self.attention_residual_output_router is None or prefix_sum is None:
            raise RuntimeError("Attention Residual output router is unavailable")
        return self.attention_residual_output_router(prefix_sum, completed_blocks)

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        layer_residual: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
        alibi_slopes: torch.Tensor | None = None,
        dense_attention_update: Callable | None = None,
    ) -> torch.Tensor:
        if dense_attention_update is not None and alibi_slopes is None:
            raise ValueError("Topology correction requires compiled causal ALiBi")
        if alibi_slopes is not None and (mask is not None or self.global_mixer != "dense"):
            raise ValueError("Compiled ALiBi requires the dense global mixer without another mask")
        hidden = src
        # Preserve the incumbent ALiBi policy: suffix padding only goes to KDA.
        dense_padding_mask = (
            src_key_padding_mask
            if alibi_slopes is None and (mask is None or mask.ndim == 2)
            else None
        )
        if self.attention_residual_block_size:
            return self._forward_attention_residual(
                src,
                mask,
                src_key_padding_mask,
                dense_padding_mask,
                layer_residual,
                alibi_slopes,
                dense_attention_update,
            )
        if "k3_moe" in self.channel_mixer_pattern:
            for index, (kind, layer) in enumerate(zip(self.pattern, self.layers, strict=True)):
                if layer_residual is not None:
                    hidden = hidden + layer_residual(index, hidden)
                if kind == "dense":
                    update = (dense_attention_update(index, layer, hidden)
                              if dense_attention_update is not None else
                              self._activation_call(
                                  self._dense_token_mixing_update,
                                  layer, hidden, mask, dense_padding_mask, alibi_slopes))
                elif kind == "mla":
                    update = self._mla_token_mixing_update(
                        index, layer, hidden, mask, dense_padding_mask
                    )
                else:
                    update = self._activation_call(
                        layer.token_mixing_update, hidden, src_key_padding_mask)
                hidden = hidden + update
                hidden = hidden + self._activation_call(
                    partial(self._channel_mixing_update, index, kind, layer),
                    hidden, src_key_padding_mask,
                )
            return hidden
        for index, (kind, layer) in enumerate(zip(self.pattern, self.layers, strict=True)):
            if layer_residual is not None:
                hidden = hidden + layer_residual(index, hidden)
            if kind == "dense":
                if alibi_slopes is None:
                    hidden = self._activation_call(
                        layer, hidden, src_mask=mask,
                        src_key_padding_mask=dense_padding_mask)
                else:
                    update = (
                        dense_attention_update(index, layer, hidden)
                        if dense_attention_update is not None else
                        self._activation_call(
                            self._dense_token_mixing_update,
                            layer, hidden, mask, dense_padding_mask, alibi_slopes)
                    )
                    hidden = hidden + update
                    hidden = hidden + self._activation_call(
                        self._dense_feed_forward_update, layer, hidden)
            elif kind == "mla":
                hidden = hidden + self._mla_token_mixing_update(
                    index, layer, hidden, mask, dense_padding_mask
                )
                hidden = hidden + self._activation_call(
                    partial(self._channel_mixing_update, index, kind, layer),
                    hidden, src_key_padding_mask)
            else:
                hidden = self._activation_call(
                    layer, hidden, src_key_padding_mask)
        return hidden
