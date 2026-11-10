"""Minimal Kimi K3 gated latent attention for the haplotype benchmark."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _attention_bias(
    attention_mask: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
    *,
    batch: int,
    heads: int,
    length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Translate PyTorch-MHA masks to SDPA additive-bias semantics."""

    bias = attention_mask
    if bias is not None:
        if bias.shape[-2:] != (length, length):
            raise ValueError("MLA attention mask has the wrong sequence shape")
        if bias.dtype == torch.bool:
            additive = torch.zeros(bias.shape, dtype=dtype, device=device)
            bias = additive.masked_fill(bias.to(device), float("-inf"))
        else:
            bias = bias.to(device=device, dtype=dtype)
        if bias.ndim == 2:
            bias = bias[None, None]
        elif bias.ndim == 3 and bias.shape[0] == batch * heads:
            bias = bias.reshape(batch, heads, length, length)
        elif bias.ndim == 3 and bias.shape[0] == batch:
            bias = bias[:, None]
        elif bias.ndim != 4:
            raise ValueError("MLA attention mask must be 2-D, 3-D, or 4-D")
        if bias.shape[0] not in (1, batch) or bias.shape[1] not in (1, heads):
            raise ValueError("MLA attention mask does not broadcast over batch/heads")
    if padding_mask is not None:
        if padding_mask.shape != (batch, length):
            raise ValueError("MLA padding mask has the wrong shape")
        if bias is None:
            bias = torch.zeros(batch, 1, length, length, dtype=dtype, device=device)
        else:
            bias = bias.expand(batch, bias.shape[1], length, length).clone()
        bias.masked_fill_(padding_mask[:, None, None, :], float("-inf"))
    return bias


class K3GatedMLA(nn.Module):
    """K3 equation 7: shared latent KV, full-rank output gate, global attention."""

    def __init__(
        self,
        *,
        d_model: int,
        heads: int,
        latent_dim: int,
        q_head_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model <= 0 or heads <= 0 or d_model % heads:
            raise ValueError("MLA model width must divide evenly across heads")
        if latent_dim <= 0 or q_head_dim <= 0:
            raise ValueError("MLA latent and query dimensions must be positive")
        self.d_model = d_model
        self.heads = heads
        self.latent_dim = latent_dim
        self.q_head_dim = q_head_dim
        self.v_head_dim = d_model // heads
        self.dropout = dropout
        self.query = nn.Linear(d_model, heads * q_head_dim, bias=False)
        self.compress_kv = nn.Linear(d_model, latent_dim, bias=False)
        self.key_up = nn.Linear(latent_dim, heads * q_head_dim, bias=False)
        self.value_up = nn.Linear(latent_dim, d_model, bias=False)
        self.output_gate = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("last_gate_mean", torch.zeros(()), persistent=False)
        self.register_buffer("last_qk_scale", torch.zeros(()), persistent=False)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.d_model:
            raise ValueError("MLA inputs must have shape [batch,length,width]")
        batch, length, _ = inputs.shape
        query = self.query(inputs).reshape(
            batch, length, self.heads, self.q_head_dim
        ).transpose(1, 2)
        latent = self.compress_kv(inputs)
        key = self.key_up(latent).reshape(
            batch, length, self.heads, self.q_head_dim
        ).transpose(1, 2)
        value = self.value_up(latent).reshape(
            batch, length, self.heads, self.v_head_dim
        ).transpose(1, 2)
        bias = _attention_bias(
            attention_mask,
            padding_mask,
            batch=batch,
            heads=self.heads,
            length=length,
            dtype=query.dtype,
            device=query.device,
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=bias is None,
        )
        combined = attended.transpose(1, 2).reshape(batch, length, self.d_model)
        gate = torch.sigmoid(self.output_gate(inputs))
        result = self.output(combined * gate)
        with torch.no_grad():
            self.last_gate_mean.copy_(gate.float().mean())
            self.last_qk_scale.copy_(
                query.float().square().mean().sqrt()
                * key.float().square().mean().sqrt()
                * math.sqrt(self.q_head_dim)
            )
        return result

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "heads": self.heads,
            "q_head_dim": self.q_head_dim,
            "value_head_dim": self.v_head_dim,
            "kv_latent_dim": self.latent_dim,
            "kv_cache_compression_ratio": (
                self.heads * (self.q_head_dim + self.v_head_dim) / self.latent_dim
            ),
            "gate_mean": float(self.last_gate_mean),
            "qk_scale": float(self.last_qk_scale),
        }

    # ponytail: full-sequence benchmark only; add a compressed decode cache when
    # incremental generation becomes an evaluated workload.
