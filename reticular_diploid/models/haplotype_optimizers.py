"""Stock PyTorch Muon for hidden matrices, with AdamW for other parameters.

The RMS adjustment follows https://arxiv.org/abs/2502.16982. This is whole-
matrix Muon by default. An explicit Q/K/V head partition follows Kimi K3's
per-head orthogonalization with stock Torch RMS scaling. Packed MoE banks remain
per expert; neither option changes model parameters or their gradients.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from reticular_diploid.models.kimi_moe import K3GroupedExperts

_SCHEMA = "pgs-hybrid-muon-adamw/v1"
_HEAD_PARTITION = "qkv_per_head_v1"


def _attention_head_rows(model: nn.Module) -> dict[int, list[list[int]]]:
    """Recognize only native fused dense QKV and FLA KDA projection layouts."""
    from fla.layers import KimiDeltaAttention

    rows: dict[int, list[list[int]]] = {}

    def add(parameter, heads, width, projections=1):
        if (not isinstance(parameter, nn.Parameter) or parameter.ndim != 2
                or not parameter.is_contiguous() or type(heads) is not int or heads <= 0
                or type(width) is not int or width <= 0
                or parameter.shape[0] != projections * heads * width
                or id(parameter) in rows):
            raise ValueError("Unsupported or aliased attention head matrix layout")
        rows[id(parameter)] = [[start, start + width]
                              for start in range(0, parameter.shape[0], width)]

    for name, module in model.named_modules():
        if not name.startswith("blocks."):
            continue
        if isinstance(module, nn.MultiheadAttention):
            if (module.in_proj_weight is None or module.embed_dim % module.num_heads
                    or module.in_proj_weight.shape != (3 * module.embed_dim, module.embed_dim)):
                raise ValueError("Per-head Muon requires the native fused dense QKV layout")
            add(module.in_proj_weight, module.num_heads, module.embed_dim // module.num_heads, 3)
        elif isinstance(module, KimiDeltaAttention):
            for field, heads, width in (("q_proj", module.num_heads, module.head_k_dim),
                                        ("k_proj", module.num_heads, module.head_k_dim),
                                        ("v_proj", module.num_v_heads, module.head_v_dim)):
                projection = getattr(module, field)
                if not isinstance(projection, nn.Linear) or projection.weight.shape[1] != module.hidden_size:
                    raise ValueError("Unsupported KDA attention projection")
                add(projection.weight, heads, width)
    if not rows:
        raise ValueError("Per-head Muon requires recognized native attention projections")
    return rows


class HybridMuonAdamW(torch.optim.Optimizer):
    """Expose original parameters/state while delegating updates to PyTorch.

    Expert views share parameter and momentum storage. Their temporary gradient
    views are assigned only at step(), after native accumulation and clipping.
    Checkpoints contain original-shaped state and a strict grouping manifest.
    """

    def __init__(self, groups: list[dict[str, Any]]) -> None:
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("Hybrid Muon requires the installed PyTorch Muon optimizer")
        if not groups or groups[0].get("optimizer_kind") != "muon":
            raise ValueError("Hybrid Muon requires a nonempty hidden-matrix group first")
        parameters = [p for group in groups for p in group["params"]]
        if len({id(p) for p in parameters}) != len(parameters):
            raise ValueError("Hybrid optimizer groups must be disjoint")
        super().__init__(groups, {})
        self.completed_steps = 0
        self._matrix_views: list[tuple[nn.Parameter, list[torch.Tensor]]] = []
        matrices = []
        self._head_slices: dict[nn.Parameter, list[slice]] = {}
        group = self.param_groups[0]
        if "attention_partition" in group or "attention_head_partitions" in group:
            if group.get("attention_partition") != _HEAD_PARTITION:
                raise ValueError("Unsupported Muon attention partition")
            named = dict(zip(group["param_names"], group["params"], strict=True))
            recipes = group.get("attention_head_partitions")
            if not isinstance(recipes, list) or not recipes:
                raise ValueError("Muon attention partition lacks row identities")
            for recipe in recipes:
                if not isinstance(recipe, dict) or set(recipe) != {"param_name", "row_slices"}:
                    raise ValueError("Invalid Muon attention row identity")
                parameter = named.get(recipe["param_name"])
                spans = recipe["row_slices"]
                if (parameter is None or parameter.ndim != 2 or parameter in self._head_slices
                        or not parameter.is_contiguous() or not isinstance(spans, list) or not spans):
                    raise ValueError("Invalid Muon attention row coverage")
                end = 0
                for span in spans:
                    if (not isinstance(span, list) or len(span) != 2
                            or any(type(x) is not int for x in span)
                            or span[0] != end or not end < span[1] <= parameter.shape[0]):
                        raise ValueError("Muon attention rows must cover each element exactly once")
                    end = span[1]
                if end != parameter.shape[0]:
                    raise ValueError("Muon attention rows do not cover the full matrix")
                self._head_slices[parameter] = [slice(a, b) for a, b in spans]
        for parameter in self.param_groups[0]["params"]:
            if parameter.ndim not in (2, 3):
                raise ValueError("Muon parameters must be matrices or explicit expert banks")
            if parameter in self._head_slices:
                views = [parameter[span].detach() for span in self._head_slices[parameter]]
            else:
                views = ([parameter] if parameter.ndim == 2 else
                         [part.detach() for part in parameter.unbind(0)])
            self._matrix_views.append((parameter, views))
            matrices.extend(views)
        muon_group = {**self.param_groups[0], "params": matrices}
        self._muon = torch.optim.Muon([muon_group])
        self._adamw = torch.optim.AdamW(
            [dict(group) for group in self.param_groups[1:]], betas=(0.9, 0.95)
        )

    def _bind_children(self) -> None:
        for public, child in zip(
            self.param_groups, [*self._muon.param_groups, *self._adamw.param_groups], strict=True
        ):
            child.update({key: value for key, value in public.items() if key != "params"})
        self._muon.state = defaultdict(dict)
        self._adamw.state = defaultdict(dict)
        for parameter, views in self._matrix_views:
            if parameter.ndim == 2 and parameter not in self._head_slices:
                self._muon.state[parameter] = self.state[parameter]
                continue
            state = self.state[parameter]
            if parameter.grad is not None and "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(parameter)
            for index, view in enumerate(views):
                span = self._head_slices[parameter][index] if parameter in self._head_slices else index
                view.grad = None if parameter.grad is None else parameter.grad[span]
                if "momentum_buffer" in state:
                    self._muon.state[view] = {"momentum_buffer": state["momentum_buffer"][span]}
        for group in self.param_groups[1:]:
            for parameter in group["params"]:
                self._adamw.state[parameter] = self.state[parameter]

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        # Views can retain a reference to the previous model gradient buffer.
        for parameter, views in self._matrix_views:
            if parameter.ndim == 3 or parameter in self._head_slices:
                for view in views:
                    view.grad = None

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._bind_children()
        self._muon.step()
        self._adamw.step()
        for parameter, _ in self._matrix_views:
            if parameter.grad is not None:
                self.state[parameter]["step"] = int(self.state[parameter].get("step", 0)) + 1
        self.completed_steps += 1
        return loss

    def state_dict(self) -> dict[str, Any]:
        result = super().state_dict()
        result.update(hybrid_schema=_SCHEMA, completed_steps=self.completed_steps)
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if (state_dict.get("hybrid_schema") != _SCHEMA
                or type(state_dict.get("completed_steps")) is not int
                or state_dict["completed_steps"] < 0):
            raise ValueError("Checkpoint is not an authenticated hybrid optimizer state")
        saved = state_dict.get("param_groups")
        if not isinstance(saved, list) or len(saved) != len(self.param_groups):
            raise ValueError("Hybrid optimizer grouping differs from checkpoint")
        for old, current in zip(saved, self.param_groups, strict=True):
            # LR changes with the schedule; every other setting and named shape is fixed.
            expected = {k: v for k, v in current.items() if k not in {"params", "lr"}}
            observed = {k: v for k, v in old.items() if k not in {"params", "lr"}}
            if observed != expected or len(old["params"]) != len(current["params"]):
                raise ValueError("Hybrid optimizer parameter identity or policy differs from checkpoint")
            for key, parameter in zip(old["params"], current["params"], strict=True):
                state = state_dict["state"].get(key, {})
                step = float(state.get("step", 0))
                if state and (not math.isfinite(step) or step <= 0 or step != int(step)
                              or step > state_dict["completed_steps"]):
                    raise ValueError("Hybrid optimizer state counter differs from completed updates")
                buffers = ("momentum_buffer",) if current["optimizer_kind"] == "muon" else ("exp_avg", "exp_avg_sq")
                if state and any(not isinstance(state.get(name), torch.Tensor)
                                 or state[name].shape != parameter.shape for name in buffers):
                    raise ValueError("Hybrid optimizer state shape differs from parameter")
        super().load_state_dict(state_dict)
        self.completed_steps = state_dict["completed_steps"]
        self._bind_children()


def hybrid_muon_adamw(
    model: nn.Module, delta: nn.Module, *, learning_rate: float, weight_decay: float,
    engram_multiplier: float = 5.0, momentum: float = 0.95, ns_steps: int = 5,
    attention_partition: str | None = None,
) -> HybridMuonAdamW:
    """Use Muon on encoder linear/attention weights and per-expert matrix banks.

    AdamW retains embeddings, tied/untied output heads, all Engram parameters,
    normalization, convolution, scalar/vector parameters, routing logits, graph
    additions and all block deltas. Only Engram embeddings use the 5x/no-decay group.
    """
    if any(not math.isfinite(x) or x <= 0 for x in (learning_rate, engram_multiplier)):
        raise ValueError("Learning rate and Engram multiplier must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Weight decay must be finite and nonnegative")
    if not 0 < momentum < 1 or type(ns_steps) is not int or ns_steps <= 0:
        raise ValueError("Muon momentum and iteration count are invalid")
    if attention_partition not in (None, _HEAD_PARTITION):
        raise ValueError("Unsupported Muon attention partition")
    head_rows = _attention_head_rows(model) if attention_partition is not None else {}
    engram = getattr(model, "engram", None)
    memory = getattr(engram, "embedding", None)
    memory_ids = {id(p) for p in memory.parameters()} if memory is not None else set()
    excluded = {id(p) for p in engram.parameters()} if engram is not None else set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) or name == "output":
            excluded.update(id(p) for p in module.parameters())
    hidden_ids: set[int] = set()
    for name, module in model.named_modules():
        if not name.startswith("blocks.") or any("router" in part for part in name.split(".")):
            continue
        if isinstance(module, nn.Linear):
            hidden_ids.add(id(module.weight))
        elif isinstance(module, nn.MultiheadAttention):
            for field in ("in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"):
                parameter = getattr(module, field, None)
                if parameter is not None:
                    hidden_ids.add(id(parameter))
        elif isinstance(module, K3GroupedExperts):
            hidden_ids.update((id(module.gate_and_up), id(module.down)))
    hidden_ids -= excluded
    if not set(head_rows) <= hidden_ids:
        raise ValueError("Attention head partition includes a non-Muon parameter")
    named = [("model." + name, p) for name, p in model.named_parameters() if p.requires_grad]
    named += [("delta." + name, p) for name, p in delta.named_parameters() if p.requires_grad]
    buckets: list[list[tuple[str, nn.Parameter]]] = [[], [], []]
    for name, parameter in named:
        index = 0 if id(parameter) in hidden_ids else (2 if id(parameter) in memory_ids else 1)
        buckets[index].append((name, parameter))
    if not buckets[0] or not buckets[1]:
        raise ValueError("Hybrid optimizer requires hidden matrices and AdamW parameters")
    groups = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        lr = learning_rate * (engram_multiplier if index == 2 else 1.0)
        group: dict[str, Any] = {
            "params": [p for _, p in bucket], "param_names": [name for name, _ in bucket],
            "param_shapes": [list(p.shape) for _, p in bucket],
            "optimizer_kind": "muon" if index == 0 else "adamw",
            "group_name": ("hidden_matrices", "adamw_common", "engram_embeddings")[index],
            "lr": lr, "initial_lr": lr, "weight_decay": 0.0 if index == 2 else weight_decay,
        }
        if index == 0:
            group.update(momentum=momentum, nesterov=True, ns_steps=ns_steps,
                         ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-7,
                         adjust_lr_fn="match_rms_adamw")
            if attention_partition is not None:
                group.update(attention_partition=attention_partition, attention_head_partitions=[
                    {"param_name": name, "row_slices": head_rows[id(parameter)]}
                    for name, parameter in bucket if id(parameter) in head_rows
                ])
        else:
            group.update(betas=(0.9, 0.95), eps=1e-8, amsgrad=False, maximize=False,
                         foreach=None, capturable=False, differentiable=False, fused=None,
                         decoupled_weight_decay=True)
        groups.append(group)
    return HybridMuonAdamW(groups)


def optimizer_state_readiness(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Describe initialized state honestly for both incumbent AdamW and hybrid Muon.

    Conditional parameters without any gradient may lack state, as in incumbent
    Adam admission. Every observed state must have the correct full-sized buffers.
    """
    hybrid = isinstance(optimizer, HybridMuonAdamW)
    counts = {"adamw_parameter_states": 0, "muon_parameter_states": 0, "muon_matrix_states": 0}
    ready = True
    for group in optimizer.param_groups:
        kind = group.get("optimizer_kind", "adamw")
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if not state:
                continue
            names = ("momentum_buffer",) if kind == "muon" else ("exp_avg", "exp_avg_sq")
            ready &= (float(state.get("step", 0)) > 0 and all(
                isinstance(state.get(name), torch.Tensor) and state[name].shape == parameter.shape
                for name in names))
            counts[kind + "_parameter_states"] += 1
            if kind == "muon":
                counts["muon_matrix_states"] += (len(optimizer._head_slices[parameter])
                    if hybrid and parameter in optimizer._head_slices
                    else parameter.shape[0] if parameter.ndim == 3 else 1)
    total = counts["adamw_parameter_states"] + counts["muon_parameter_states"]
    if hybrid:
        ready &= counts["adamw_parameter_states"] > 0 and counts["muon_parameter_states"] > 0
    return {"optimizer_kind": "hybrid_muon_adamw" if hybrid else "adamw",
            "initialized": bool(total) and bool(ready), "parameter_states": total, **counts}
