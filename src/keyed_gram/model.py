from __future__ import annotations

import math
from typing import TYPE_CHECKING, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call

from .config import GramModelConfig

if TYPE_CHECKING:
    from .keying import AuxPermutationKey


def _call_frozen(module: nn.Module, *args, **kwargs):
    params_and_buffers = {
        **{name: value.detach() for name, value in module.named_parameters()},
        **{name: value for name, value in module.named_buffers()},
    }
    return functional_call(module, params_and_buffers, args, kwargs)


def _call(module: nn.Module, frozen: bool, *args, **kwargs):
    if frozen:
        return _call_frozen(module, *args, **kwargs)
    return module(*args, **kwargs)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GramModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        kv_width = self.num_kv_heads * self.head_dim
        self.rotary = Rotary(self.head_dim, max_seq_len=config.context_length)
        # Keep the upstream GRAM projection layout so architecture and parameter
        # accounting remain directly comparable to the pinned implementation.
        self.c_attn_q = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.c_attn_kv = nn.Linear(config.hidden_size, 2 * kv_width, bias=True)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch, seq_len, _ = x.shape
        q = self.c_attn_q(x).view(batch, seq_len, self.num_heads, self.head_dim)
        kv = self.c_attn_kv(x)
        k, v = kv.split(self.num_kv_heads * self.head_dim, dim=-1)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        q, k = self.rotary(q), self.rotary(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if attention_mask is None:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=self.num_kv_heads != self.num_heads
            )
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                enable_gqa=self.num_kv_heads != self.num_heads,
            )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.c_proj(y)


class Rotary(nn.Module):
    """Rotary embedding used by the pinned GRAM attention implementation."""

    def __init__(self, dim: int, max_seq_len: int) -> None:
        super().__init__()
        if dim % 4:
            raise ValueError("attention head dimension must be divisible by four")
        angular_freq = (1 / 1024) ** torch.linspace(
            0, 1, steps=dim // 4, dtype=torch.float32
        )
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j->ij", positions, angular_freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        cos = self.cos[None, : x.size(-3), None, :]
        sin = self.sin[None, : x.size(-3), None, :]
        left, right = x.float().chunk(2, dim=-1)
        first = left * cos + right * sin
        second = left * (-sin) + right * cos
        return torch.cat((first, second), dim=-1).type_as(x)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, width: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, width, bias=True)
        self.c_proj = nn.Linear(width, hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class GatelessMoE(nn.Module):
    def __init__(self, config: GramModelConfig) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [FeedForward(config.hidden_size, config.core_ffn_width)]
            + [FeedForward(config.hidden_size, config.aux_ffn_width) for _ in range(config.num_aux)]
        )

    def forward(
        self,
        x: Tensor,
        fwd_mask: Tensor,
        bck_mask: Tensor,
        overrides: Mapping[int, Tensor] | None = None,
        expert_scales: Tensor | None = None,
        detach_aux_input: bool = False,
        return_expert_outputs: bool = False,
    ) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
        if fwd_mask.numel() != len(self.experts) or bck_mask.numel() != len(self.experts):
            raise ValueError("expert masks do not match the number of experts")
        if expert_scales is not None and expert_scales.numel() != len(self.experts):
            raise ValueError("expert scales do not match the number of experts")
        output: Tensor | None = None
        expert_outputs: dict[int, Tensor] = {}
        overrides = overrides or {}
        for index, expert in enumerate(self.experts):
            weight = fwd_mask[index]
            if not bool(weight):
                continue
            if index in overrides:
                current = overrides[index]
                if not bool(bck_mask[index]):
                    current = current.detach()
            else:
                expert_input = x.detach() if detach_aux_input and index > 0 else x
                current = _call(expert, not bool(bck_mask[index]), expert_input)
            if expert_scales is not None:
                current = current * expert_scales[index]
            if not bool(weight == 1):
                current = current * weight
            expert_outputs[index] = current
            output = current if output is None else output + current
        if output is None:
            raise ValueError("at least one expert must be active")
        if return_expert_outputs:
            return output, expert_outputs
        return output


class GramBlock(nn.Module):
    def __init__(self, config: GramModelConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.moe = GatelessMoE(config)
        self.norm_1 = nn.RMSNorm(config.hidden_size)
        self.norm_2 = nn.RMSNorm(config.hidden_size)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        fwd_mask: Tensor,
        bck_mask: Tensor,
        overrides: Mapping[int, Tensor] | None = None,
    ) -> Tensor:
        freeze_core = not bool(bck_mask[0])
        normed = _call(self.norm_1, freeze_core, x)
        x = x + _call(self.attn, freeze_core, normed, attention_mask)
        normed = _call(self.norm_2, freeze_core, x)
        return x + self.moe(normed, fwd_mask, bck_mask, overrides=overrides)


class GramTransformer(nn.Module):
    """Decoder-only GRAM model with one core expert and one or more auxiliary experts."""

    def __init__(self, config: GramModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([GramBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        if config.learnable_aux_scale:
            self.aux_scales = nn.Parameter(
                torch.full(
                    (config.num_layers, config.num_aux),
                    float(config.aux_scale_init),
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter("aux_scales", None)
        self.apply(self._init_weights)

    @property
    def num_experts(self) -> int:
        return 1 + self.config.num_aux

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def _attention_mask(self, tokens: Tensor) -> Tensor:
        batch, seq_len = tokens.shape
        starts = torch.zeros_like(tokens, dtype=torch.bool)
        if seq_len > 1:
            starts[:, 1:] = tokens[:, :-1].eq(self.config.eos_token_id)
        segment = starts.cumsum(dim=1)
        same_segment = segment[:, :, None].eq(segment[:, None, :])
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=tokens.device).tril()
        return (same_segment & causal).view(batch, 1, seq_len, seq_len)

    def _keyed_aux_output(
        self,
        x: Tensor,
        layer_index: int,
        expert_index: int,
        key: "AuxPermutationKey",
    ) -> Tensor:
        if expert_index != key.expert_index:
            raise ValueError("key expert index does not match the requested expert")
        if key.num_layers != len(self.blocks) or key.aux_width != self.config.aux_ffn_width:
            raise ValueError("key layout does not match model layout")
        group_size = key.group_size
        if self.config.aux_ffn_width % group_size:
            raise ValueError("key group size does not divide auxiliary width")
        mapping = key.position_mapping()
        up_weights: list[Tensor] = []
        up_biases: list[Tensor] = []
        down_columns: list[Tensor] = []
        groups = self.config.aux_ffn_width // group_size
        for group_index in range(groups):
            src_layer, src_group = mapping.get(
                (layer_index, group_index), (layer_index, group_index)
            )
            expert = self.blocks[src_layer].moe.experts[expert_index]
            start = src_group * group_size
            stop = start + group_size
            up_weights.append(expert.c_fc.weight[start:stop])
            up_biases.append(expert.c_fc.bias[start:stop])
            down_columns.append(expert.c_proj.weight[:, start:stop])
        up_weight = torch.cat(up_weights, dim=0)
        up_bias = torch.cat(up_biases, dim=0)
        down_weight = torch.cat(down_columns, dim=1)
        down_bias = self.blocks[layer_index].moe.experts[expert_index].c_proj.bias
        hidden = F.gelu(F.linear(x, up_weight, up_bias), approximate="tanh")
        return F.linear(hidden, down_weight, down_bias)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        fwd_mask: Tensor | None = None,
        bck_mask: Tensor | None = None,
        key: "AuxPermutationKey | None" = None,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor | None, dict[str, list[Tensor]]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.config.context_length:
            raise ValueError("input sequence exceeds configured context length")
        if fwd_mask is None:
            fwd_mask = torch.tensor([1] + [0] * self.config.num_aux, device=input_ids.device)
        if bck_mask is None:
            bck_mask = fwd_mask.to(dtype=torch.bool)
        fwd_mask = fwd_mask.to(device=input_ids.device)
        bck_mask = bck_mask.to(device=input_ids.device)
        freeze_core = not bool(bck_mask[0])
        attention_mask = self._attention_mask(input_ids)
        diagnostics: dict[str, list[Tensor]] = {
            "aux_inputs": [],
            "aux_residuals": [],
        }
        x = _call(self.embed, freeze_core, input_ids)
        for layer_index, block in enumerate(self.blocks):
            normed = _call(block.norm_1, freeze_core, x)
            x = x + _call(block.attn, freeze_core, normed, attention_mask)
            normed = _call(block.norm_2, freeze_core, x)
            overrides: dict[int, Tensor] = {}
            if key is not None and bool(fwd_mask[key.expert_index]):
                overrides[key.expert_index] = self._keyed_aux_output(
                    normed.detach() if self.config.aux_input_stop_gradient else normed,
                    layer_index,
                    key.expert_index,
                    key,
                )
            expert_scales = None
            if self.aux_scales is not None:
                expert_scales = torch.cat(
                    [normed.new_ones(1), self.aux_scales[layer_index].to(normed.dtype)]
                )
            moe_result = block.moe(
                normed,
                fwd_mask=fwd_mask,
                bck_mask=bck_mask,
                overrides=overrides,
                expert_scales=expert_scales,
                detach_aux_input=self.config.aux_input_stop_gradient,
                return_expert_outputs=return_diagnostics,
            )
            if return_diagnostics:
                moe_output, expert_outputs = moe_result
                diagnostics["aux_inputs"].append(normed.detach())
                diagnostics["aux_residuals"].append(
                    expert_outputs.get(1, torch.zeros_like(normed))
                )
            else:
                moe_output = moe_result
            x = x + moe_output
        x = _call(self.norm, freeze_core, x)
        logits = _call(self.lm_head, freeze_core, x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                ignore_index=-100,
            )
        if return_diagnostics:
            return logits, loss, diagnostics
        return logits, loss

    def parameter_counts(self) -> dict[str, int]:
        core_names = ("embed.", "norm.", "lm_head.")
        core = 0
        auxiliary = 0
        for name, parameter in self.named_parameters():
            if name == "aux_scales":
                auxiliary += parameter.numel()
            elif ".moe.experts.0." in name or ".attn." in name or ".norm_" in name or name.startswith(core_names):
                core += parameter.numel()
            else:
                auxiliary += parameter.numel()
        return {"core": core, "auxiliary": auxiliary, "total": core + auxiliary}
