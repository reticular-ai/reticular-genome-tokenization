"""Small, trainable Kimi K3 Stable LatentMoE for the haplotype benchmark.

The equations follow Moonshot's K3 report and were cross-checked against the
MIT-licensed ``pablo-reyes8/kimi-k3-pytorch`` reference.  Sparse expert
execution uses the differentiable ``torch._grouped_mm`` path used by NVIDIA
NeMo AutoModel when CUDA is available; a simple loop remains as a CPU oracle.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def situ_glu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    beta_gate: float = 4.0,
    beta_up: float = 25.0,
) -> torch.Tensor:
    """K3's bounded SiTU-GLU, with nonlinearities evaluated in FP32."""

    gate_float = gate.float()
    up_float = up.float()
    activated = (
        beta_gate
        * torch.tanh(gate_float / beta_gate)
        * torch.sigmoid(gate_float)
        * beta_up
        * torch.tanh(up_float / beta_up)
    )
    return activated.to(dtype=gate.dtype)


class K3SiTUGLU(nn.Module):
    """One dense/shared K3 SiTU expert."""

    def __init__(self, width: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(width, hidden_dim, bias=False)
        self.up = nn.Linear(width, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, width, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down(situ_glu(self.gate(inputs), self.up(inputs)))


class K3GroupedExperts(nn.Module):
    """Top-k latent experts stored as two grouped-matmul weight tensors."""

    def __init__(
        self,
        num_experts: int,
        latent_dim: int,
        hidden_dim: int,
        *,
        grouped_mm: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.grouped_mm = grouped_mm
        self.last_backend = "reference_loop"
        self.gate_and_up = nn.Parameter(
            torch.empty(num_experts, latent_dim, 2 * hidden_dim)
        )
        self.down = nn.Parameter(
            torch.empty(num_experts, hidden_dim, latent_dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.gate_and_up, mean=0.0, std=0.02)
        nn.init.normal_(self.down, mean=0.0, std=0.02)

    @staticmethod
    def _group_assignments(
        selected_experts: torch.Tensor,
        selected_weights: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, top_k = selected_experts.shape
        flat_experts = selected_experts.reshape(-1)
        flat_tokens = (
            torch.arange(tokens, device=selected_experts.device)[:, None]
            .expand(tokens, top_k)
            .reshape(-1)
        )
        order = torch.argsort(flat_experts, stable=True)
        sorted_experts = flat_experts[order]
        sorted_tokens = flat_tokens[order]
        sorted_weights = selected_weights.reshape(-1)[order]
        counts = torch.bincount(sorted_experts, minlength=num_experts)
        offsets = counts.cumsum(0).to(torch.int32)
        return sorted_experts, sorted_tokens, sorted_weights, offsets

    def _grouped_forward(
        self,
        latent: torch.Tensor,
        sorted_tokens: torch.Tensor,
        sorted_weights: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        permuted = latent.index_select(0, sorted_tokens)
        gate_and_up = self.gate_and_up.to(dtype=permuted.dtype)
        down = self.down.to(dtype=permuted.dtype)
        projected = torch._grouped_mm(permuted, gate_and_up, offsets)
        gate, up = projected.chunk(2, dim=-1)
        activated = situ_glu(gate, up) * sorted_weights[:, None].to(
            dtype=projected.dtype
        )
        return torch._grouped_mm(activated, down, offsets)

    def _reference_forward(
        self,
        latent: torch.Tensor,
        sorted_experts: torch.Tensor,
        sorted_tokens: torch.Tensor,
        sorted_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs: list[torch.Tensor] = []
        output_tokens: list[torch.Tensor] = []
        gate_and_up = self.gate_and_up.to(dtype=latent.dtype)
        down = self.down.to(dtype=latent.dtype)
        for expert_index in range(self.num_experts):
            positions = torch.nonzero(
                sorted_experts == expert_index, as_tuple=False
            ).flatten()
            if positions.numel() == 0:
                continue
            token_ids = sorted_tokens.index_select(0, positions)
            projected = latent.index_select(0, token_ids) @ gate_and_up[expert_index]
            gate, up = projected.chunk(2, dim=-1)
            activated = situ_glu(gate, up) * sorted_weights.index_select(
                0, positions
            )[:, None].to(dtype=projected.dtype)
            outputs.append(activated @ down[expert_index])
            output_tokens.append(token_ids)
        if not outputs:
            return latent.new_zeros((0, self.latent_dim))
        return torch.cat(outputs, dim=0), torch.cat(output_tokens, dim=0)

    def forward(
        self,
        latent: torch.Tensor,
        selected_experts: torch.Tensor,
        selected_weights: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError("latent expert inputs have the wrong shape")
        if selected_experts.shape != selected_weights.shape:
            raise ValueError("expert indices and weights must align")
        (
            sorted_experts,
            sorted_tokens,
            sorted_weights,
            offsets,
        ) = self._group_assignments(
            selected_experts, selected_weights, self.num_experts
        )
        use_grouped = (
            self.grouped_mm
            and latent.is_cuda
            and hasattr(torch, "_grouped_mm")
        )
        if use_grouped:
            self.last_backend = "torch._grouped_mm"
            expert_outputs = self._grouped_forward(
                latent, sorted_tokens, sorted_weights, offsets
            )
            output_tokens = sorted_tokens
        else:
            self.last_backend = "reference_loop"
            expert_outputs, output_tokens = self._reference_forward(
                latent,
                sorted_experts,
                sorted_tokens,
                sorted_weights,
            )
        accumulation_dtype = (
            torch.float64 if latent.dtype == torch.float64 else torch.float32
        )
        aggregate = torch.zeros(
            latent.shape[0],
            self.latent_dim,
            device=latent.device,
            dtype=accumulation_dtype,
        )
        return aggregate.index_add(
            0, output_tokens, expert_outputs.to(dtype=accumulation_dtype)
        )


class K3StableLatentMoE(nn.Module):
    """Scaled K3 Stable LatentMoE with exact next-step Quantile Balancing."""

    def __init__(
        self,
        *,
        d_model: int,
        latent_dim: int,
        hidden_dim: int,
        num_routed_experts: int,
        top_k: int,
        num_shared_experts: int = 2,
        quantile_balancing: bool = True,
        norm_eps: float = 1e-6,
        grouped_mm: bool = True,
    ) -> None:
        super().__init__()
        if not 1 <= top_k < num_routed_experts:
            raise ValueError("K3 MoE requires 1 <= top_k < routed experts")
        if min(d_model, latent_dim, hidden_dim, num_shared_experts) <= 0:
            raise ValueError("K3 MoE dimensions and shared experts must be positive")
        if grouped_mm and (latent_dim % 8 or hidden_dim % 8):
            raise ValueError(
                "BF16 grouped expert latent and hidden dimensions must divide by 8"
            )
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        self.quantile_balancing = quantile_balancing
        self.shared_experts = nn.ModuleList(
            K3SiTUGLU(d_model, hidden_dim)
            for _ in range(num_shared_experts)
        )
        self.latent_down = nn.Linear(d_model, latent_dim, bias=False)
        self.router = nn.Linear(d_model, num_routed_experts, bias=False)
        self.routed_experts = K3GroupedExperts(
            num_routed_experts,
            latent_dim,
            hidden_dim,
            grouped_mm=grouped_mm,
        )
        self.routed_norm = nn.RMSNorm(latent_dim, eps=norm_eps)
        self.latent_up = nn.Linear(latent_dim, d_model, bias=False)
        self.register_buffer(
            "routing_bias", torch.zeros(num_routed_experts)
        )
        self.register_buffer(
            "last_expert_load",
            torch.zeros(num_routed_experts, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "last_weight_entropy", torch.zeros(()), persistent=False
        )

    @torch.no_grad()
    def _commit_next_quantile_bias(
        self,
        raw_scores: torch.Tensor,
        biased_cutoff: torch.Tensor,
    ) -> None:
        margins = raw_scores.float() - biased_cutoff.float()[:, None]
        quantile = 1.0 - self.top_k / self.num_routed_experts
        corrections = -torch.quantile(
            margins, quantile, dim=0, interpolation="linear"
        )
        corrections -= corrections.mean()
        self.routing_bias.copy_(corrections)

    def _route(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        raw_scores = torch.sigmoid(self.router(inputs).float())
        biased_scores = raw_scores + self.routing_bias.float()
        need_cutoff = self.training and self.quantile_balancing
        requested = self.top_k + int(need_cutoff)
        top_values, top_indices = torch.topk(
            biased_scores, k=requested, dim=-1
        )
        selected = top_indices[:, : self.top_k]
        selected_raw = raw_scores.gather(1, selected)
        weights = selected_raw / selected_raw.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-9)
        cutoff = top_values[:, self.top_k] if need_cutoff else None
        return selected, weights, (raw_scores, cutoff) if need_cutoff else None

    def forward(
        self,
        inputs: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.d_model:
            raise ValueError("K3 MoE inputs must have shape [batch,length,width]")
        flat_inputs = inputs.reshape(-1, self.d_model)
        if padding_mask is None:
            valid = torch.ones(
                flat_inputs.shape[0], device=inputs.device, dtype=torch.bool
            )
        else:
            if padding_mask.shape != inputs.shape[:2]:
                raise ValueError("K3 MoE padding mask does not align with inputs")
            valid = (~padding_mask).reshape(-1)
        valid_inputs = flat_inputs[valid]
        if valid_inputs.shape[0] == 0:
            return torch.zeros_like(inputs)

        shared = torch.zeros_like(valid_inputs)
        for expert in self.shared_experts:
            shared = shared + expert(valid_inputs)

        latent = self.latent_down(valid_inputs)
        selected, weights, balancing = self._route(valid_inputs)
        aggregate = self.routed_experts(latent, selected, weights)
        routed = self.latent_up(
            self.routed_norm(aggregate).to(dtype=latent.dtype)
        )
        valid_output = shared + routed
        flat_output = torch.zeros_like(flat_inputs)
        flat_output[valid] = valid_output.to(dtype=flat_output.dtype)

        with torch.no_grad():
            self.last_expert_load.copy_(
                torch.bincount(
                    selected.reshape(-1), minlength=self.num_routed_experts
                )
            )
            self.last_weight_entropy.copy_(
                -(weights * weights.clamp_min(1e-9).log()).sum(-1).mean()
            )
            if balancing is not None:
                raw_scores, cutoff = balancing
                if cutoff is None:
                    raise RuntimeError("Quantile Balancing cutoff is missing")
                self._commit_next_quantile_bias(raw_scores, cutoff)
        return flat_output.reshape_as(inputs)

    @property
    def routed_expert_parameters(self) -> int:
        return 3 * self.latent_dim * self.hidden_dim

    @property
    def active_parameters_per_token(self) -> int:
        fixed = (
            sum(parameter.numel() for expert in self.shared_experts for parameter in expert.parameters())
            + self.latent_down.weight.numel()
            + self.router.weight.numel()
            + self.routed_norm.weight.numel()
            + self.latent_up.weight.numel()
        )
        return fixed + self.top_k * self.routed_expert_parameters

    def diagnostics(self) -> dict[str, float | int | str]:
        load = self.last_expert_load.float()
        mean = load.mean()
        load_cv = float(load.std(unbiased=False) / mean) if mean > 0 else math.nan
        return {
            "backend": self.routed_experts.last_backend,
            "routed_experts": self.num_routed_experts,
            "top_k": self.top_k,
            "zero_load_experts": int((load == 0).sum()),
            "load_cv": load_cv,
            "weight_entropy": float(self.last_weight_entropy),
            "routing_bias_std": float(
                self.routing_bias.float().std(unbiased=False)
            ),
        }
