# coding=utf-8
# Copyright 2023 YuanLab and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn
import copy
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_layers import GenericForSequenceClassification
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    can_return_tuple,
    auto_docstring,
)
from transformers.utils.deprecation import deprecate_kwarg
from verl.models.configs.configuration_yuan import YuanConfig
from einops import rearrange

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "YuanConfig"

try:
    from flash_attn import flash_attn_varlen_func as flash_attn_unpadded_func
    # from flash_attn import flash_attn_func
except ImportError:
    flash_attn_unpadded_func = None
    logger.warning_once('_attn_implementation = flash_attention_2')
try:
    # from apex.normalization import MixedFusedRMSNorm as RMSNorm
    from transformer_engine.pytorch import RMSNorm
except:
    RMSNorm = None

class YuanRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        YuanRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    '''
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    '''
    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

if RMSNorm == None:
    RMSNorm = YuanRMSNorm

class YuanDynamicCache(DynamicCache):
    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.lf1_cache: List[torch.Tensor] = []
        self.lf2_cache: List[torch.Tensor] = []

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx], self.lf1_cache[layer_idx], self.lf2_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx], self.lf1_cache[layer_idx], self.lf2_cache[layer_idx])

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        lf1_cache: torch.Tensor,
        lf2_cache: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append([])
                    self.value_cache.append([])
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                self.lf1_cache.append(lf1_cache)
                self.lf2_cache.append(lf2_cache)
            elif (
                len(self.key_cache[layer_idx]) == 0
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
                self.lf1_cache[layer_idx] = lf1_cache
                self.lf2_cache[layer_idx] = lf2_cache
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-3)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                self.lf1_cache[layer_idx] = lf1_cache
                self.lf2_cache[layer_idx] = lf2_cache

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

class YuanBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: YuanConfig, num_experts: int):
        super().__init__()
        self.ffn_hidden_size = config.moe_config['ffn_hidden_size']
        self.hidden_size = config.hidden_size
        self.use_bias = config.use_bias
        self.hidden_act = config.hidden_act

        # self.w1 = nn.Linear(self.hidden_size, self.ffn_hidden_size*2, bias=self.use_bias)
        # self.w2 = nn.Linear(self.ffn_hidden_size, self.hidden_size, bias=self.use_bias)
        self.num_experts = num_experts
        self.w1 = nn.ModuleList([nn.Linear(self.hidden_size, self.ffn_hidden_size*2, bias=self.use_bias) for _ in range(self.num_experts)])
        self.w2 = nn.ModuleList([nn.Linear(self.ffn_hidden_size, self.hidden_size, bias=self.use_bias) for _ in range(self.num_experts)])

        self.act_fn = ACT2FN[self.hidden_act]

    def forward(self, hidden_states, routing_weights, selected_experts):
        '''
        current_hidden_states = self.w1(hidden_states)
        current_hidden_states = torch.chunk(current_hidden_states, 2, dim=-1)
        current_hidden_states = self.act_fn(current_hidden_states[0]) * current_hidden_states[1]
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states
        '''
        final_hidden_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, self.hidden_size)

            current_hidden_states = self.w1[expert_idx](current_state)
            current_hidden_states = torch.chunk(current_hidden_states, 2, dim=-1)
            current_hidden_states = self.act_fn(current_hidden_states[0]) * current_hidden_states[1]
            current_hidden_states = self.w2[expert_idx](current_hidden_states)

            current_hidden_states = current_hidden_states * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states


class ParallelAttention_router(nn.Module):
    def __init__(self, config, num_experts):
        super(ParallelAttention_router, self).__init__()

        self.hidden_size = config.hidden_size
        self.projection_size = num_experts
        self.query_key_value = nn.Linear(self.hidden_size, self.projection_size*3, bias=False)

    def forward(self, hidden_states):
        mix_layer = self.query_key_value(hidden_states)
        (query_layer, key_layer, value_layer) = torch.chunk(mix_layer, 3, dim=-1)

        query_layer = query_layer.view(*query_layer.shape, 1).float()
        key_layer = key_layer.view(*key_layer.shape, 1).float()
        value_layer = value_layer.view(*value_layer.shape, 1).float()

        attn_weights = torch.matmul(query_layer, key_layer.transpose(1,2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

        attn_output = torch.matmul(attn_weights, value_layer)
        router_output = attn_output.squeeze(2)
        return router_output

class YuanSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accommodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config, num_experts):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.top_k = config.moe_config['moe_top_k']
        self.hidden_size = config.hidden_size

        # gating
        if config.moe_config['router_type'] == 'attn_router':
            self.router = ParallelAttention_router(config, self.num_experts)
        else:
            self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)

        # self.experts = nn.ModuleList([YuanBlockSparseTop2MLP(config) for _ in range(self.num_experts)])
        self.experts = YuanBlockSparseTop2MLP(config, num_experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.router(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        '''
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        '''
        final_hidden_states = self.experts(hidden_states, routing_weights, selected_experts)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

'''
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
'''

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    else:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class LocalizedFiltering(torch.nn.Module):
    """
    Mega's Exponential Moving Average layer, largely left unmodified from the original repo with the exception of
    variable names and moving away from the stateful representation of incremental decoding state. See
    "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.lf_conv2d_group = config.lf_conv2d_group
        self.lf_conv2d_num_pad = config.lf_conv2d_num_pad
        self.conv1 = torch.nn.Conv2d(self.embed_dim, self.embed_dim // 2, (2, 1), stride=(1, 1), padding=(self.lf_conv2d_num_pad, 0), groups=self.lf_conv2d_group)
        self.conv2 = torch.nn.Conv2d(self.embed_dim // 2, self.embed_dim, (2, 1), stride=(1, 1), padding=(self.lf_conv2d_num_pad, 0), groups=self.lf_conv2d_group)
        self.output_layernorm = RMSNorm(self.embed_dim, eps=1e-6)

    def forward(self, hidden_states, lf1_cache=None, lf2_cache=None):
        inputs = hidden_states.transpose(0,1)
        seq_len, bsz, embed_dim = inputs.size()
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Unexpected embedding dimension received: input is {embed_dim}, model expects {self.embed_dim}"
            )
        residual = inputs
        inputs = inputs.view(seq_len, 1, bsz, embed_dim).permute(2, 3, 0, 1)

        if self.training:

            output1 = self.conv1(inputs)
            output1 = output1[:, :, :seq_len, :]

            output2 = self.conv2(output1)
            output2 = output2[:, :, :seq_len, :].permute(2, 3, 0, 1).contiguous()
            output2 = output2.view(seq_len, bsz, embed_dim)
            assert output2.shape == residual.shape
            lf1_cache = None
            lf2_cache = None
        else:
            if lf1_cache == None:
                lf1_cache = torch.zeros((bsz, embed_dim, 1, 1), dtype=inputs.dtype)
            inputs = torch.cat([lf1_cache, inputs], dim=2)  # [b, h, s+1, 1]
            output1 = self.conv1(inputs)
            if lf2_cache == None:
                lf2_cache = torch.zeros((bsz, embed_dim//2, 1, 1), dtype=inputs.dtype)
            output1 = torch.cat([lf2_cache, output1], dim=2)
            output2 = self.conv2(output1).permute([2, 3, 0, 1]).contiguous()
            output2 = output2.view(seq_len, bsz, embed_dim)
            assert output2.shape == residual.shape

            assert list(output2.shape) == list(residual.shape), f'{output2.shape}, {residual.shape}'
            lf1_cache = inputs[:, :, -1:, :].contiguous()
            lf2_cache = output1[:, :, -1:, :].contiguous()
        lf_output = self.output_layernorm(output2+residual)
        lf_output = lf_output.transpose(0, 1)
        return lf_output, lf1_cache, lf2_cache

# flash attn
class YuanFlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, device=None, dtype=None):
        super().__init__()
        assert flash_attn_unpadded_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal

    def forward(self, module: torch.nn.Module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask, dropout=0.0, scaling=1.0, sliding_window=None, output_attentions=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((i.is_cuda for i in (q,k,v)))

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]
        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)
        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q
            is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=q.device)
        output = flash_attn_unpadded_func(q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k, dropout, softmax_scale=scaling, causal=is_causal)

        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output, None

class YuanAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: YuanConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = getattr(config, "hidden_size")
        self.attention_projection_size = getattr(config, "attention_projection_size", config.hidden_size)
        self.num_attention_heads = getattr(config, "num_attention_heads")
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_attention_heads)
        self.head_dim = getattr(config, "head_dim", self.attention_projection_size // self.num_attention_heads)
        self.use_bias = getattr(config, "use_bias", False)
        self.scaling = self.head_dim**-0.5
        self.num_key_value_groups = self.num_attention_heads // config.num_key_value_heads
        self.is_causal = True

        self.lf_gate = LocalizedFiltering(self.config)

        if self.num_key_value_groups == 1:
           self.get_query_key = nn.Linear(self.hidden_size, self.attention_projection_size * 2, bias = self.use_bias)
           self.v_proj = nn.Linear(self.hidden_size, self.attention_projection_size, bias = self.use_bias)
           self.o_proj = nn.Linear(self.attention_projection_size, self.hidden_size, bias = self.use_bias)
        else:
           self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=self.use_bias)
           self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=self.use_bias)
           self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=self.use_bias)
           self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=self.use_bias)

        self.max_position_embeddings = getattr(config, "max_position_embeddings")
        self.rotary_base = getattr(config, "rotary_base")
        self.rotary_percent = getattr(config, "rotary_percent", 1.0)
        self.rotary_dim = int(getattr(config, "rotary_dim", self.attention_projection_size // self.num_attention_heads))
        if self.rotary_percent < 1.0:
            self.rotary_dim = int(self.rotary_dim * self.rotary_percent)
        self.embed_positions = create_sinusoidal_positions(self.rotary_base, self.max_position_embeddings, self.rotary_dim)

        if config._attn_implementation == 'eager' and flash_attn_unpadded_func != None:
            self.core_attention = YuanFlashSelfAttention(causal=True)
        else:
            self._attn_implementation =  'flash_attention_2'

    def _get_embed_positions(self, position_ids):
        embed_positions = self.embed_positions
        if embed_positions.device != position_ids.device:
            embed_positions = embed_positions.to(position_ids.device)
            self.embed_positions = embed_positions
        return embed_positions.repeat(position_ids.shape[0], 1, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_values: Optional[YuanDynamicCache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        value_states = self.v_proj(hidden_states)
        if self.training:
            hidden_states, _, _ = self.lf_gate(hidden_states, lf1_cache, lf2_cache, attn_metadata)
        else:
            if len(past_key_values.key_cache) <= self.layer_idx:
                bsz, seq_len, h = hidden_states.shape
                lf1_cache=torch.zeros((bsz, h, 1, 1), dtype = hidden_states.dtype, device=hidden_states.device)
                lf2_cache=torch.zeros((bsz, h//2, 1, 1), dtype = hidden_states.dtype, device=hidden_states.device)
            else:
                lf1_cache = past_key_values.lf1_cache[self.layer_idx]
                lf2_cache = past_key_values.lf2_cache[self.layer_idx]
            hidden_states, lf1, lf2 = self.lf_gate(hidden_states, lf1_cache, lf2_cache)
        hidden_shape = (*input_shape, -1, self.head_dim)
        if self.num_key_value_groups == 1:
            query_key = self.get_query_key(hidden_states)
            query_key = query_key.view(*query_key.shape[:-1], self.num_attention_heads, int(self.head_dim)*2)
            (query_states, key_states) = torch.chunk(query_key, 2, dim=-1)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)
        '''
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        '''
        # 参考了gptj的实现过程
        embed_positions = self._get_embed_positions(position_ids)
        repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        sincos = torch.gather(embed_positions, 1, repeated_position_ids)
        sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        if self.rotary_dim is not None:
            k_rot = key_states[:, :, :, : self.rotary_dim]
            k_pass = key_states[:, :, :, self.rotary_dim :]

            q_rot = query_states[:, :, :, : self.rotary_dim]
            q_pass = query_states[:, :, :, self.rotary_dim :]
            
            k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
            q_rot = apply_rotary_pos_emb(q_rot, sin, cos)

            key_states = torch.cat([k_rot, k_pass], dim=-1)
            query_states = torch.cat([q_rot, q_pass], dim=-1)
        else:
            key_states = apply_rotary_pos_emb(key_states, sin, cos)
            query_states = apply_rotary_pos_emb(query_states, sin, cos)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, lf1, lf2, self.layer_idx)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            query_states = query_states.transpose(1,2)
            key_states = key_states.transpose(1,2)
        else:
            attention_interface = self.core_attention
            value_states = value_states.transpose(1, 2)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class YuanDecoderLayer(nn.Module):
    def __init__(self, config: YuanConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = YuanAttention(config, layer_idx)
        self.use_moe = getattr(config, "use_moe", False)
        if self.use_moe:
            if 'per_layer_experts_blocks' in config.moe_config:
                assert config.moe_config['per_layer_experts_blocks'] != None
                self.num_experts = config.moe_config['per_layer_experts_blocks'][layer_idx]
            elif 'moe_num_experts' in config.moe_config:
                assert config.moe_config['moe_num_experts'] != None
                self.num_experts = config.moe_config['moe_num_experts']
            else:
                raise ValueError(f'per_layer_experts_blocks or moe_num_experts must in config.moe_config')
            self.mlp = YuanSparseMoeBlock(config, self.num_experts)
        else:
            self.mlp = YuanMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[YuanDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        output_router_logits: Optional[bool] = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
     
        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

def create_sinusoidal_positions(base: float, num_pos: int, dim: int, offset: int=0.0) -> torch.Tensor:
    # inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64) / dim))
    inv_freq = (1.0 / ( base**(torch.arange(0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device()) / dim))).to(torch.float32)
    seq = (
        torch.arange(num_pos, device=inv_freq.device, dtype=inv_freq.dtype)
        + offset
    )

    freqs = torch.outer(seq, inv_freq)
    sinusoid_inp = torch.cat((freqs, freqs), dim=-1)
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=-1)

@torch.fx.wrap
def get_embed_positions(embed_positions, position_ids):
    return embed_positions.to(position_ids.device).repeat(position_ids.shape[0], 1, 1)


def rotary_interleaved(x:torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    # sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    # cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    sin = sin[:, :, None, :]
    cos = cos[:, :, None, :]
    return ((tensor * cos) + (rotary_interleaved(tensor) * sin)).to(tensor.dtype)

YUAN_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`YuanConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Yuan Model outputting raw hidden-states without any specific head on top.",
    YUAN_START_DOCSTRING,
)
class YuanPreTrainedModel(PreTrainedModel):
    config_class = YuanConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["YuanDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


YUAN_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_value_lfs` are passed, the
            legacy cache format will be returned.

            If `past_key_value_lfs` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_value_lfs` key value states are returned and can be used to speed up decoding (see
            `past_key_value_lfs`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Yuan Model outputting raw hidden-states without any specific head on top.",
    YUAN_START_DOCSTRING,
)
class YuanModel(YuanPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`YuanDecoderLayer`]

    Args:
        config: YuanConfig
    """

    def __init__(self, config: YuanConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # vocab embedding layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [YuanDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(YUAN_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[YuanDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 用于reset attn mask
        input_ids1 = copy.deepcopy(input_ids)
        reset_mask_flag = False
        '''
        # 兼容多模态模型
        if past_key_values != None:
            input_ids = input_ids
            input_ids = input_ids[:,-1:]
            if use_cache:
                reset_mask_flag = True
        '''
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        elif input_ids is not None:
            input_ids = input_ids
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds.transpose(0,1)
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")


        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and len(past_key_values.key_cache) == 0:
            past_key_values = YuanDynamicCache()

        seq_length_with_past = seq_length
        past_key_values_length = 0

        '''
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[0]
            seq_length_with_past = seq_length_with_past + past_key_values_length


        # modify to reset position ids
        if past_key_values is not None:
            pos_start = position_ids[:,-1]+1
            pos_end = pos_start+past_key_values[0][0].shape[0]-position_ids.shape[1]+1
            position_ids_k = torch.arange(pos_start.item(), pos_end.item()).to(position_ids.device)
            position_ids_k = position_ids_k.unsqueeze(0)
            position_ids_k = torch.cat((position_ids, position_ids_k), dim=1)
            position_ids = position_ids[:,-1]+past_key_values[0][0].shape[0]-position_ids.shape[1]+1
            position_ids = position_ids.unsqueeze(0)
        else:
            position_ids_k = position_ids
        '''

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            pass
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # if cache_position is None:
        #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        #     cache_position = torch.arange(
        #         past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        #     )
        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)
        # TODO
        causal_mask = None
        '''
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        '''
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: YuanDynamicCache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Yuan. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: YuanConfig,
        past_key_values: YuanDynamicCache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`YuanConfig`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class YuanForCausalLM(YuanPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = YuanModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(YUAN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[YuanDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, YuanForCausalLM

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = False
        # output_router_logits = (
        #     output_router_logits if output_router_logits is not None else self.config.output_router_logits
        # )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        # aux_loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            # if output_router_logits:
            #     output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=None,
        )
class YuanForSequenceClassification(YuanForCausalLM):
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # Similar to `self.model = AutoModel.from_config(config)` but allows to change the base model name if needed in the child class
        # setattr(self, self.base_model_prefix, YuanPreTrainedModel.from_config(config))
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast:
        transformer_outputs: BaseModelOutputWithPast = getattr(self, self.base_model_prefix)(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
