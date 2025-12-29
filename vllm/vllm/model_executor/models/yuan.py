"""Inference-only Yuan model compatible with HuggingFace weights."""
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np
import torch
from torch import einsum, nn, Tensor

import vllm.envs as envs
from vllm.model_executor.models.configuration_yuan import YuanConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.config import LoRAConfig, CacheConfig, VllmConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from transformers.activations import ACT2FN
from vllm.model_executor.utils import set_weight_attrs
from vllm.attention import Attention, AttentionMetadata
# from vllm.model_executor.layers.fused_moe import *
from vllm.model_executor.layers.linear import  (ColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
# from vllm.model_executor.layers.sampler import get_sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead, DEFAULT_VOCAB_PADDING_SIZE)
# from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.model_loader.weight_utils import default_weight_loader # hf_model_weights_iterator)
from vllm.sequence import IntermediateTensors
from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.utils import set_weight_attrs
from .interfaces import SupportsPP

from .utils import (AutoWeightsLoader, PPMissingLayer,
                    is_pp_missing_parameter, make_layers,
                    maybe_prefix)

from vllm.model_executor.layers.layernorm import RMSNorm as VLLM_RMSNorm
from vllm.model_executor.layers.fused_moe import fused_topk_v2, FusedMoE
from vllm.attention import get_attn_backend
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType, direct_register_custom_op,
                        get_dtype_size, is_pin_memory_available)
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.compilation.decorators import support_torch_compile
from vllm.config import get_current_vllm_config


def print_ops_fake(tensor: torch.Tensor, tensor_name: str = None) -> None:
    pass

def print_ops(tensor: torch.Tensor, tensor_name: str = None) -> None:
    print(tensor_name, "shape: ", tensor.shape, "stride: ", tensor.stride(), "sum: ", tensor.sum(), flush=True)

direct_register_custom_op(
    op_name="print_ops",
    op_func=print_ops,
    mutates_args=[],
    fake_impl=print_ops_fake,
    tags=(torch.Tag.needs_fixed_stride_order, ),
)


class YuanRMSNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class ParallelAttention_router(nn.Module):
    def __init__(self, config, num_experts):
        super(ParallelAttention_router, self).__init__()

        self.hidden_size = config.hidden_size
        self.projection_size = num_experts
        self.query_key_value = ReplicatedLinear(self.hidden_size, self.projection_size*3, bias=False)

    def forward(self, hidden_states):
        mix_layer, _ = self.query_key_value(hidden_states)
        (query_layer, key_layer, value_layer) = torch.chunk(mix_layer, 3, dim=-1)
        
        query_layer = query_layer.view(*query_layer.shape, 1).float()
        key_layer = key_layer.view(*key_layer.shape, 1).float()
        value_layer = value_layer.view(*value_layer.shape, 1).float()

        attn_weights = torch.matmul(query_layer, key_layer.transpose(1,2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

        attn_output = torch.matmul(attn_weights, value_layer)
        router_output = attn_output.squeeze(2)
        return router_output


class MoEDroplessTokenDispatcher:
    def __init__(self, num_experts: int, config: YuanConfig) -> None:
        
        self.num_experts = num_experts
        assert self.num_experts > 0, "Expected at least one expert"
        self.router_topk = config.moe_config['moe_top_k']

    def token_permutation(
        self, hidden_states: torch.Tensor, max_prob: torch.Tensor, max_ind: torch.Tensor
    ):
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        if self.router_topk > 1:
            global_local_map = torch.ones_like(max_ind).bool()
            local_indices = max_ind.masked_select(global_local_map)
            local_probs = max_prob.masked_select(global_local_map)
            global_local_map = global_local_map.nonzero()[:, 0]
            global_local_map = global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
            local_hidden_states = torch.gather(hidden_states, 0, global_local_map)

        indices = torch.argsort(local_indices, dim=0)
        tokens_per_expert = torch.histc(
            local_indices,
            bins=self.num_experts,
            min=0,
            max=self.num_experts - 1,
        )
        tokens_per_expert = tokens_per_expert.cpu().to(torch.long)

        indices = indices.view(-1, 1).expand(-1, hidden_states.shape[-1])
        permuted_local_hidden_states = torch.gather(local_hidden_states, 0, indices)
        return (permuted_local_hidden_states, tokens_per_expert, local_probs, indices, global_local_map)

    def token_unpermutation(
        self,
        hidden_states: torch.Tensor,
        scores: torch.Tensor,
        indices: torch.Tensor,
        global_local_map: torch.Tensor = None,
    ):
        scores = scores.to(dtype=hidden_states.dtype)
        unpermuted_local_hidden = torch.zeros_like(hidden_states)
        assert indices.shape == hidden_states.shape, f'{indices.shape}, {hidden_states.shape}'
        unpermuted_local_hidden = unpermuted_local_hidden.scatter(0, indices, hidden_states)

        if self.router_topk > 1:
            unpermuted_local_hidden = unpermuted_local_hidden * scores.view(-1, 1)

        output_total = unpermuted_local_hidden

        if self.router_topk > 1:
            global_num_tokens = self.hidden_shape[0]
            global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
            unpermuted_global_hidden = torch.zeros(
                global_hidden_shape,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            output_total = unpermuted_global_hidden.scatter_add(
                0, global_local_map, unpermuted_local_hidden
            )
        output_total = output_total.view(self.hidden_shape)

        return output_total


class YuanMoeLayer(nn.Module):
    def __init__(self, config:YuanConfig, num_layer: int,num_experts,
                quant_config: Optional[QuantizationConfig] = None,  # Quant Use
                prefix: str = ""):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.top_k = config.moe_config['moe_top_k']
        self.hidden_size = config.hidden_size
        self.num_layer = num_layer
        self.is_old_version = int(os.environ.get('OLD_YUAN_VERSION', 0))
        self.tp_size = get_tensor_model_parallel_world_size()

        if config.moe_config['router_type'] == 'attn_router':
            if self.is_old_version:
                self.gate = ParallelAttention_router(config, self.num_experts)
            else:
                self.router = ParallelAttention_router(config, self.num_experts)
        else:
            if self.is_old_version:
                self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
            else:
                self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.token_dispatcher = MoEDroplessTokenDispatcher(self.num_experts, self.config)
        self.experts = FusedMoE(num_experts=self.num_experts,
                                    top_k=self.top_k,
                                    hidden_size=self.hidden_size,
                                    intermediate_size=config.moe_config['ffn_hidden_size'],
                                    reduce_results=False,
                                    renormalize=config.moe_config["norm_topk_prob"],
                                    quant_config=quant_config,
                                    prefix=f"{prefix}.experts",
                                    custom_routing_function=fused_topk_v2)


    def routing(self, logits: torch.Tensor) -> torch.Tensor:
        top_logits, indices = torch.topk(logits, k=self.top_k, dim=1)
        scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)
        return scores, indices

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.is_old_version:
            logits = self.gate(hidden_states)
        else:
            logits = self.router(hidden_states)
        final_hidden_states = self.experts(hidden_states, logits)
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)
        return final_hidden_states

def _yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(_yarn_find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)

def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class YuanYaRNScaledRotaryEmbedding(nn.Module):
    def __init__(self,
                 dim,
                 rotary_base=10000,
                 max_position_embeddings=2048,
                 scale=1,
                 original_max_position_embeddings=2048,
                 extrapolation_factor=1,
                 attn_factor=1,
                 beta_fast=32,
                 beta_slow=1,
                 dtype=torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = rotary_base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        
        self.revised_yarn()
        self.max_seq_len_cached = max_position_embeddings
        t = np.arange(self.max_seq_len_cached)
        t = torch.tensor(t, device=self.inv_freq.device,dtype=torch.float)
        freqs = torch.outer(t, self.inv_freq)
        self.emb = torch.cat((freqs, freqs), dim=-1)

    def forward(self, seq_len=None):
        return self.emb[:, None, None, :]

    def yarn(self, device):
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)
        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.base,
            self.original_max_position_embeddings
        )
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, self.dim // 2).float().to(device)) \
            * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def revised_yarn(self):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.base,
            self.original_max_position_embeddings
        )
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, self.dim // 2).float()) * self.extrapolation_factor
        inv_freq = inv_freq / ((1-inv_freq_mask)*self.scale + inv_freq_mask)
        self.register_buffer("inv_freq", inv_freq)


class YuanRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, dtype=torch.float32, rotary_interleaved=False, seq_len_interpolation_factor=None):
        super().__init__()
        self.base = base
        self.dim = dim
        self.rotary_interleaved = rotary_interleaved
        self.seq_len_interpolation_factor = seq_len_interpolation_factor

    def forward(self, max_seq_len, offset=0):

        """Forward pass of RoPE embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): _description_. Defaults to 0.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        inv_freq = (1.0 / ( self.base**(torch.arange(0, self.dim, 2, dtype=torch.float32, device=torch.cuda.current_device()) / self.dim))).to(torch.float32)
        
        #max_seq_len_int = max_seq_len.item() if max_seq_len.numel() == 1 else max_seq_len.max().item()
        seq = (
            torch.arange(max_seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
            + offset
        )

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = torch.outer(seq, inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
                freqs.shape[0], -1
            )
        # emb [seq_length, .., dim]
        emb = emb[:, None, None, :]
        #emb = emb[:, None, :] 
        return emb

def _rotate_half_bshd(x: Tensor, rotary_interleaved: bool):
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x_new = torch.stack((-x2, x1), dim=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)

def apply_rotary_pos_emb_bshd(t: Tensor, freqs: Tensor, position_ids: Tensor ,rotary_interleaved: bool = False):
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    dtype = t.dtype
    rot_dim = freqs.shape[-1]
    #if position_ids.shape[1] > 1:
    freqs = freqs[position_ids]
    freqs = freqs.view(t.shape[0],freqs.shape[1],freqs.shape[3])
    #freqs = freqs.view(t.shape[1],freqs.shape[1],freqs.shape[2],freqs.shape[4]).transpose(0,1)
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    t = (t * cos_) + (_rotate_half_bshd(t, rotary_interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)

def apply_rotary_pos_emb_thd(
        t: Tensor, cu_seqlens: Tensor, freqs: Tensor, position_ids: Tensor, rotary_interleaved: bool = False,
):

    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """

    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    return torch.cat(
        [
            apply_rotary_pos_emb_bshd(x.unsqueeze(1), freqs[: x.size(0)], position_ids)
            for x in torch.split(t, seqlens)
        ]
    ).squeeze(1)


def apply_rotary_pos_emb(t: Tensor, freqs: Tensor, position_ids: Tensor, apply_rope_fusion: bool = True, cu_seqlens: Optional[Tensor] = None):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels, or bshd (conventional) / thd (packed seq) format
    """
    return apply_rotary_pos_emb_bshd(t, freqs, position_ids)


class LocalizedFiltering(torch.nn.Module):
    """
    Mega's Exponential Moving Average layer, largely left unmodified from the original repo with the exception of
    variable names and moving away from the stateful representation of incremental decoding state. See
    "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(self, config, cache_config, hidden_size):
        super().__init__()

        self.embed_dim = hidden_size
        self.output_layernorm = VLLM_RMSNorm(self.embed_dim, eps=config.rms_norm_eps)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        
        params_dtype = torch.bfloat16
        self.conv1_weight = nn.Parameter(torch.empty(self.embed_dim // self.tp_size, 
                                                     self.embed_dim, dtype=params_dtype).cuda())
        self.conv2_weight = nn.Parameter(torch.empty(self.embed_dim // 2 // self.tp_size,
                                                     self.embed_dim * 2, dtype=params_dtype).cuda())
        self.register_parameter("conv1_weight", self.conv1_weight)
        self.register_parameter("conv2_weight", self.conv2_weight)

        self.use_lfa_bias = config.use_lfa_bias
        if self.use_lfa_bias:
            self.conv1_bias = nn.Parameter(torch.empty(self.embed_dim // 2, dtype=params_dtype).cuda())
            self.conv2_bias = nn.Parameter(torch.empty(self.embed_dim, dtype=params_dtype).cuda())
            self.register_parameter("conv1_bias", self.conv1_bias)
            self.register_parameter("conv2_bias", self.conv2_bias)
        '''
        通过torch.compile + cudaGraph 支持静态图功能，要求模型的输入在profill阶段就构建好所有的输入Tensor,后续执行Tensor首地址不能发生改变
        torch.compile 通过装饰器的方式，修饰YuanModel，torch.compile会捕获模型的动态性
        '''
        self.start_layer, self.end_layer = get_pp_indices(
            config.num_hidden_layers,
            get_pp_group().rank_in_group,
            get_pp_group().world_size
        )
        #  设置每个GPU分配2Gib的显存大小, lf_cache_nums > kv_cache_nums即可
        if cache_config.cache_dtype == "fp8":
            scale = 2
        else:
            scale = 1
        lf_cache_nums = 2 * 2**30 * scale // (self.embed_dim * 3 // self.tp_size * (self.end_layer - self.start_layer))
        self.lf1_caches = torch.zeros((lf_cache_nums, self.embed_dim // self.tp_size), dtype=params_dtype, device="cuda")
        self.lf2_caches = torch.zeros((lf_cache_nums, self.embed_dim // 2 // self.tp_size), dtype=params_dtype, device="cuda")

    def fused_cat_conv2d(
            self,
            inputs: torch.Tensor,
            pre_lf_indexs: torch.Tensor,
            out_lf_indexs: torch.Tensor,
            input_lf_loc: torch.Tensor,
            out_lf_loc: torch.Tensor,
            inputs_loc: torch.Tensor,
            outputs_loc: torch.Tensor,
            lf_caches: torch.Tensor,
            conv_weight: torch.Tensor,
        ):
        bs = pre_lf_indexs.shape[0]
        lf_cache = lf_caches[pre_lf_indexs]
        new_shape = [inputs.shape[0] + bs, inputs.shape[1]]
        inputs_t = torch.zeros(new_shape, dtype=inputs.dtype, device=inputs.device)
        inputs_t[input_lf_loc] = lf_cache
        inputs_t[inputs_loc] = inputs
        lf_caches.index_put_([out_lf_indexs], inputs_t[out_lf_loc])
        combined_out = torch.matmul(inputs_t, conv_weight)
        output_t = combined_out[:-1, :combined_out.shape[1]//2] + combined_out[1:, combined_out.shape[1]//2:]
        return output_t[outputs_loc]

    def forward(
            self,
            inputs: torch.Tensor,
            pre_lf_indexs: torch.Tensor,
            out_lf_indexs: torch.Tensor,
            input_lf_loc: torch.Tensor,
            out_lf_loc: torch.Tensor,
            inputs_loc: torch.Tensor,
            outputs_loc: torch.Tensor,
        ):
        assert pre_lf_indexs.shape == input_lf_loc.shape
        assert out_lf_indexs.shape == out_lf_loc.shape
        input_t = inputs.chunk(self.tp_size, dim=1)[self.tp_rank]
        output1 = self.fused_cat_conv2d(input_t, pre_lf_indexs, out_lf_indexs, input_lf_loc, out_lf_loc,
                inputs_loc, outputs_loc, self.lf1_caches, self.conv1_weight)
        if self.tp_size > 1:
            output1 = tensor_model_parallel_all_reduce(output1)
        if self.use_lfa_bias:
            output1 = output1 + self.conv1_bias
        output1_t = output1.chunk(self.tp_size, dim=1)[self.tp_rank]
        output2 = self.fused_cat_conv2d(output1_t, pre_lf_indexs, out_lf_indexs, input_lf_loc, out_lf_loc,
                inputs_loc, outputs_loc, self.lf2_caches, self.conv2_weight)
        if self.tp_size > 1:
            output2 = tensor_model_parallel_all_reduce(output2)
        if self.use_lfa_bias:
            output2 = output2 + self.conv2_bias
        output3 = output2 + inputs
        output = self.output_layernorm(output3)
        return output


class YuanMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.up_proj = ColumnParallelLinear(hidden_size,
                                            intermediate_size,
                                            bias=False,)
        self.gate_proj= ColumnParallelLinear(hidden_size,
                                            intermediate_size,
                                            bias=False,)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        x1, _ = self.up_proj(x)
        x3 = self.act_fn(x1)
        x2, _ = self.gate_proj(x)
        x, _ = self.down_proj(x2 * x3)
        return x


class YuanAttention(nn.Module):

    def __init__(
        self,
        config: YuanConfig,
        hidden_size: int,
        attention_projection_size: int,
        num_heads: int,
        num_kv_heads: int,
        attn_head_size=None,
        rope_theta: float = 500000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 4096,
        bias: bool = False,
        sliding_window: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None, # Quant Use
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads

        self.attn_head_size = attention_projection_size // num_heads if attn_head_size is None else attn_head_size
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = config.attention_projection_size // self.total_num_heads if hasattr(config, 'attention_projection_size') else hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        
        self.eps = 1e-6
        self.get_query_key = ColumnParallelLinear(
            hidden_size,
            (self.q_size + self.kv_size) * self.tp_size,
            bias=config.use_bias,
            quant_config=quant_config,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.kv_size * self.tp_size,
            bias=config.use_bias,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.q_size * self.tp_size,
            hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
        )
        
        self.lf_gate = LocalizedFiltering(config, cache_config, hidden_size)
        self.attn = Attention(self.num_heads,
                              self.attn_head_size,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              ) 

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        pre_lf_indexs: torch.Tensor,
        out_lf_indexs: torch.Tensor,
        input_lf_loc: torch.Tensor,
        out_lf_loc: torch.Tensor,
        inputs_loc: torch.Tensor,
        outputs_loc: torch.Tensor,
        use_yarn: bool=False,
        yarn_scale_factor: float=1.0,
        attn_factor: float=1.0,
    ) ->  torch.Tensor:
        v, _ = self.v_proj(hidden_states)
        hidden_states = self.lf_gate(hidden_states, pre_lf_indexs, out_lf_indexs, input_lf_loc, out_lf_loc, inputs_loc, outputs_loc)

        qk, _ = self.get_query_key(hidden_states)
        qk = qk.view(*qk.shape[:-1], self.num_kv_heads, int(qk.shape[-1] // self.num_kv_heads))
        q, k = qk.split([qk.shape[-1] - v.shape[-1] // self.num_kv_heads, v.shape[-1] // self.num_kv_heads], dim=-1)
        q = q.reshape(q.shape[0], self.num_heads, -1)

        q = apply_rotary_pos_emb(q, rotary_pos_emb, positions)
        k = apply_rotary_pos_emb(k, rotary_pos_emb, positions)
        q = q.view(-1, self.q_size)
        k = k.view(-1, self.kv_size)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class YuanDecoderLayer(nn.Module):

    def __init__(
        self,
        config: YuanConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_projection_size = getattr(config, 'attention_projection_size', config.hidden_size)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, 'num_kv_heads', self.num_heads)
        self.self_attn = YuanAttention(
            config=config,
            hidden_size=self.hidden_size,
            attention_projection_size=self.attention_projection_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.use_moe = getattr(config, "use_moe", False)
        if self.use_moe:
            layer_idx = int(prefix.split(".")[-1])
            if 'per_layer_experts_blocks' in config.moe_config:
                assert config.moe_config['per_layer_experts_blocks'] != None
                self.num_experts = config.moe_config['per_layer_experts_blocks'][layer_idx]
            elif 'moe_num_experts' in config.moe_config:
                assert config.moe_config['moe_num_experts'] != None
                self.num_experts = config.moe_config['moe_num_experts']
            else:
                raise ValueError(f'per_layer_experts_blocks or moe_num_experts must in config.moe_config')
            self.mlp = YuanMoeLayer(config, layer_idx, self.num_experts,
                                    quant_config=quant_config, prefix=prefix)
        else:
            self.mlp = YuanMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
        self.input_layernorm = VLLM_RMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)
        self.post_attention_layernorm = VLLM_RMSNorm(config.hidden_size,
                                                     eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        pre_lf_indexs: torch.Tensor,
        out_lf_indexs: torch.Tensor,
        input_lf_loc: torch.Tensor,
        out_lf_loc: torch.Tensor,
        inputs_loc: torch.Tensor,
        outputs_loc: torch.Tensor,
        use_yarn: bool=False,
        yarn_scale_factor: float=1.0,
        attn_factor: float=1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions,
            hidden_states,
            rotary_pos_emb,
            pre_lf_indexs,
            out_lf_indexs,
            input_lf_loc,
            out_lf_loc,
            inputs_loc,
            outputs_loc,
            use_yarn=use_yarn,
            yarn_scale_factor=yarn_scale_factor,
            attn_factor=attn_factor
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states 


@support_torch_compile
class YuanModel(nn.Module):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
 
        self.vocab_size = config.vocab_size + lora_vocab
        rotary_percent = getattr(config, "rotary_percent", 1.0)
        attention_projection_size = getattr(config, 'attention_projection_size', config.hidden_size)
        rotary_dim = getattr(config, "rotary_dim", attention_projection_size // config.num_attention_heads)
        if rotary_percent < 1.0:
            rotary_dim = rotary_dim * rotary_percent
        self.use_yarn = getattr(config, "use_yarn", False)
        rotary_interleaved = getattr(config, "rotary_interleaved", False)
        rotary_base = getattr(config, "rotary_base", 500000)
        seq_len_interpolation_factor = getattr(config, "seq_len_interpolation_factor", None)
        self.yarn_scale_factor = getattr(config, "yarn_scale_factor", 128)
        max_position_embeddings = getattr(config, "max_position_embeddings", 4096)
        self.attn_factor = getattr(config, "attn_factor", 1.0)
        scaled_max_position_embeddings = getattr(config, "scaled_max_position_embeddings", max_position_embeddings)
        self.torch_dtype = getattr(config, "torch_dtype", torch.bfloat16)
        self.use_moe = getattr(config, "use_moe", False)

        if self.use_yarn:
            self.rotary_emb = YuanYaRNScaledRotaryEmbedding(
                rotary_dim,
                max_position_embeddings=scaled_max_position_embeddings,
                scale=self.yarn_scale_factor,
                original_max_position_embeddings=max_position_embeddings,
                attn_factor=self.attn_factor,
                dtype=self.torch_dtype
            )
            self.seq_len = scaled_max_position_embeddings
        else:
            self.rotary_emb = YuanRotaryEmbedding(rotary_dim,
                                                  base=rotary_base, 
                                                  dtype=self.torch_dtype, 
                                                  rotary_interleaved=rotary_interleaved, 
                                                  seq_len_interpolation_factor=seq_len_interpolation_factor)
            self.seq_len = max_position_embeddings
        self.rotary_pos_emb = self.rotary_emb(self.seq_len)
         
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: YuanDecoderLayer(config, quant_config=quant_config, cache_config=cache_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = VLLM_RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        pre_lf_indexs: torch.Tensor,
        out_lf_indexs: torch.Tensor,
        input_lf_loc: torch.Tensor,
        out_lf_loc: torch.Tensor,
        inputs_loc: torch.Tensor,
        outputs_loc: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                self.rotary_pos_emb,
                pre_lf_indexs,
                out_lf_indexs,
                input_lf_loc,
                out_lf_loc,
                inputs_loc,
                outputs_loc,
                self.use_yarn,
                self.yarn_scale_factor,
                self.attn_factor
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
            })
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (".bias", "_bias", ".k_scale", "_k_scale",
                           ".v_scale", "_v_scale", ".weight_scale",
                           "_weight_scale", ".input_scale", "_input_scale")

        for name, loaded_weight in weights:
            if "rotary_pos_emb" in name:
                continue
            if is_pp_missing_parameter(name, self):
                continue

            if self.use_moe and 'experts' in name:
                pattern = r'(layers\.\d+\.mlp\.experts\.w\d+)\.\d+\.([^.]+)$'
                param_name = re.sub(pattern, r'\1_\2', name)
                if "w1" in param_name:
                    param_name = param_name.replace("w1", "w13")
 
                # Skip loading extra parameters for GPTQ/modelopt models.
                if param_name.endswith(ignore_suffixes) and param_name not in params_dict:
                    print(f'{param_name} not in params_dict')
                    continue
                if is_pp_missing_parameter(param_name, self):
                    print(f'pp_missing: {param_name} not in params_dict')
                    continue

                layer_id = int(name.split(".")[1])
                expert_id = int(name.split(".")[-2])
                if 'per_layer_experts_blocks' in self.config.moe_config:
                    assert self.config.moe_config['per_layer_experts_blocks'] != None
                    num_experts = self.config.moe_config['per_layer_experts_blocks'][layer_id]
                elif 'moe_num_experts' in self.config.moe_config:
                    assert self.config.moe_config['moe_num_experts'] != None
                    num_experts = self.config.moe_config['moe_num_experts']
                else:
                    raise ValueError(f'per_layer_experts_blocks or moe_num_experts must in config.moe_config')
                assert expert_id < num_experts, f"num_experts {num_experts} must less num_experts {num_experts}"

                param = params_dict[param_name]
                if "w1" in param_name:
                    weight_loader = param.weight_loader
                    gate, up = loaded_weight.chunk(2)
                    weight_loader(param, gate, param_name, "w1", expert_id)
                    weight_loader(param, up, param_name, "w3", expert_id)
                elif "w2" in param_name:
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, param_name, "w2", expert_id)
                loaded_params.add(param_name)
            elif 'conv1' in name and "bias" not in name:
                param_name = name.replace("conv1.", "conv1_")
                if param_name not in params_dict:
                    print(f'{name} not in params_dict')
                    continue
                param = params_dict[param_name]
                param_data = param.data
                weight_data_1 = loaded_weight[:,:,0,0].permute(1, 0).chunk(tp_size, dim=0)[tp_rank]
                weight_data_2 = loaded_weight[:,:,1,0].permute(1, 0).chunk(tp_size, dim=0)[tp_rank]
                param_data[:,:param_data.shape[1] // 2].copy_(weight_data_1)
                param_data[:,param_data.shape[1] // 2:].copy_(weight_data_2)
                loaded_params.add(param_name)
            elif 'conv1' in name and  "bias" in name:
                param_name = name.replace("conv1.", "conv1_")
                if param_name not in params_dict:
                    print(f'{name} not in params_dict')
                    continue
                param = params_dict[param_name]
                param_data = param.data
                param_data.copy_(loaded_weight)
                loaded_params.add(param_name)
            elif 'conv2' in name and "bias" not in name:
                param_name = name.replace("conv2.", "conv2_")
                if param_name not in params_dict:
                    print(f'{name} not in params_dict')
                    continue
                param = params_dict[param_name]
                param_data = param.data
                weight_data_1 = loaded_weight[:,:,0,0].permute(1, 0).chunk(tp_size, dim=0)[tp_rank]
                weight_data_2 = loaded_weight[:,:,1,0].permute(1, 0).chunk(tp_size, dim=0)[tp_rank]
                param_data[:,:param_data.shape[1] // 2].copy_(weight_data_1)
                param_data[:,param_data.shape[1] // 2:].copy_(weight_data_2)
                loaded_params.add(param_name)
            elif 'conv2' in name and  "bias" in name:
                param_name = name.replace("conv2.", "conv2_")
                if param_name not in params_dict:
                    print(f'{name} not in params_dict')
                    continue
                param = params_dict[param_name]
                param_data = param.data
                param_data.copy_(loaded_weight)
                loaded_params.add(param_name)
            else:
                if name not in params_dict:
                    print(f'{name} not in params_dict')
                    continue
                param = params_dict[name]
                param_data = param.data
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        return loaded_params


class YuanForCausalLM(nn.Module, SupportsPP):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config
        self.model = YuanModel(vllm_config=vllm_config,
                               prefix=maybe_prefix(prefix, "model"))
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head_dtype = vllm_config.model_config.lm_head_dtype 
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config else lora_config.lora_vocab_padding_size,
                quant_config=quant_config,
                params_dtype=self.lm_head_dtype,
            )
        else:
            self.lm_head = PPMissingLayer()

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        
        self.start_layer, self.end_layer = get_pp_indices(
            self.config.num_hidden_layers,
            get_pp_group().rank_in_group,
            get_pp_group().world_size
        )

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        if envs.VLLM_USE_V1:
            self.len_kv_cache = 1
        else:
            self.len_kv_cache = get_current_vllm_config().parallel_config.pipeline_parallel_size
        self.cache_config = vllm_config.cache_config
        if vllm_config.model_config.enforce_eager:
            self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        else:
            self.max_num_seqs = vllm_config.scheduler_config.cuda_graph_sizes[0]
        self.full_cuda_graph = vllm_config.compilation_config.full_cuda_graph

        inputs_len = vllm_config.scheduler_config.max_num_batched_tokens
        lf_len = inputs_len // self.cache_config.block_size + self.max_num_seqs
        self.pre_lf_indexs = torch.zeros(lf_len, dtype=torch.long, device="cuda")
        self.out_lf_indexs = torch.zeros(2*lf_len, dtype=torch.long, device="cuda")
        self.input_lf_loc = torch.zeros(lf_len, dtype=torch.long, device="cuda")
        self.out_lf_loc = torch.zeros(2*lf_len, dtype=torch.long, device="cuda")
        self.inputs_loc = torch.zeros(inputs_len, dtype=torch.long, device="cuda")
        self.outputs_loc = torch.zeros(inputs_len, dtype=torch.long, device="cuda")

    def get_lf_index(self, input_ids):
        attn_metadata = get_forward_context().attn_metadata
        lf_len = self.max_num_seqs if input_ids.shape[0] >= self.max_num_seqs else input_ids.shape[0]
        if attn_metadata is None:
            self.input_lf_len = lf_len
            self.input_len = input_ids.shape[0]
            if self.cache_config.enable_prefix_caching:
                self.output_lf_len = lf_len * 2
            else:
                self.output_lf_len = lf_len
            return


        if isinstance(attn_metadata, dict):
            attn_metadata = list(attn_metadata.values())[0]

        if not isinstance(attn_metadata, FlashAttentionMetadata): 
            assert False, f"Now not support {type(attn_metadata)}!"

        if isinstance(attn_metadata, FlashAttentionMetadata) and attn_metadata:
            # v1: use backend flashattn
            seq_lens = attn_metadata.seq_lens
            block_table = attn_metadata.block_table
            indices_1 = torch.clamp_min((seq_lens - 2) // self.cache_config.block_size, 0)
            pre_indices = torch.gather(block_table, dim=1, index=indices_1.long().unsqueeze(1)).squeeze()
            pre_indices = pre_indices.view(pre_indices.numel())
            indices_2 = (seq_lens - 1) // self.cache_config.block_size
            lf_indices = torch.gather(block_table, dim=1, index=indices_2.long().unsqueeze(1)).squeeze()
            lf_indices = lf_indices.view(lf_indices.numel())
            num_reqs = seq_lens.shape[0]
            # in cudagraph mode, prefill inputs_ids will padding with 0
            if attn_metadata.max_query_len == 1 and not self.full_cuda_graph:
                # decode
                padding = input_ids.shape[0] - num_reqs
                input_len = input_ids.shape[0]
                input_lf_loc_list = [x for x in range(0, 2*num_reqs, 2)]
                out_lf_loc_list = [x for x in range(1, 2*num_reqs, 2)]
                inputs_loc_list = [x for x in range(1, 2*num_reqs, 2)]
                outputs_loc_list = [x for x in range(0, 2*num_reqs, 2)]
                if padding > 0:
                    input_lf_loc_list.extend([-1 for _ in range(padding)])
                    out_lf_loc_list.extend([0 for _ in range(padding)])
                    inputs_loc_list.extend([2*num_reqs for _ in range(padding)])
                    outputs_loc_list.extend([0 for _ in range(padding)])
                self.input_lf_loc[:input_len].copy_(torch.tensor(input_lf_loc_list)) 
                self.out_lf_loc[:input_len].copy_(torch.tensor(out_lf_loc_list)) 
                self.pre_lf_indexs[:num_reqs].copy_(pre_indices)
                self.pre_lf_indexs[num_reqs:input_len].fill_(-1)
                self.out_lf_indexs[:num_reqs].copy_(lf_indices)
                self.out_lf_indexs[num_reqs:input_len].fill_(-1)
                self.inputs_loc[:input_len].copy_(torch.tensor(inputs_loc_list)) 
                self.outputs_loc[:input_len].copy_(torch.tensor(outputs_loc_list))
                self.input_lf_len = input_len
                self.output_lf_len = input_len
                self.input_len = input_len
            else:
                padding = input_ids.shape[0] - attn_metadata.num_actual_tokens
                query_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
                context_lens_tensor = seq_lens - query_lens
                context_lens_tensor_list = context_lens_tensor.tolist()
                query_lens_list = query_lens.tolist()
                seq_lens_list = seq_lens.tolist()
                input_lf_loc_list = []
                out_lf_loc_list = []
                inputs_loc_list = []
                outputs_loc_list = []
                if sum(query_lens_list) == 0:
                    query_lens_list = [1 for _ in range(input_ids.shape[0])]
                    padding = 0
                for i, l in enumerate(query_lens_list):
                    if self.cache_config.enable_prefix_caching and l > 1:
                        start = self.cache_config.block_size - context_lens_tensor_list[i] % self.cache_config.block_size
                        list_t = [i + j + sum(query_lens_list[:i]) \
                            for j in range(start, l, self.cache_config.block_size)]
                        out_lf_loc_list.extend(list_t)
                    input_lf_loc_list.append(sum(query_lens_list[:i])+i)
                    out_lf_loc_list.append(i + sum(query_lens_list[:i+1]))
                    inputs_loc_list.extend([x+input_lf_loc_list[i] for x in range(1, l+1)])
                    outputs_loc_list.extend([x+input_lf_loc_list[i] for x in range(l)])
                if padding > 0:
                    input_lf_loc_list.extend([-1 for _ in range(padding)])
                    out_lf_loc_list.extend([0 for _ in range(padding)])
                    inputs_loc_list.extend([inputs_loc_list[-1] + 1 for _ in range(padding)])
                    outputs_loc_list.extend([0 for _ in range(padding)])
                
                self.input_lf_len = len(input_lf_loc_list)
                self.output_lf_len = len(out_lf_loc_list)
                self.input_len = input_ids.shape[0]
                self.input_lf_loc[:self.input_lf_len].copy_(torch.tensor(input_lf_loc_list))
                self.out_lf_loc[:self.output_lf_len].copy_(torch.tensor(out_lf_loc_list))
                self.inputs_loc[:self.input_len].copy_(torch.tensor(inputs_loc_list))
                self.outputs_loc[:self.input_len].copy_(torch.tensor(outputs_loc_list))
                lf_prefill_indices_list = []
                for i in range(num_reqs):
                    if self.cache_config.enable_prefix_caching:
                        # prefix_cache
                        if query_lens_list[i] == 1:
                            # chunked prefill
                            lf_prefill_indices_list.append(lf_indices[i:i+1])
                            continue

                        if context_lens_tensor_list[i] != 0:
                            if context_lens_tensor_list[i] % self.cache_config.block_size == 0:
                                x = context_lens_tensor_list[i] // self.cache_config.block_size
                                pre_indices[i] = block_table[i][x-1]
                            else:
                                x = context_lens_tensor_list[i] // self.cache_config.block_size
                                pre_indices[i] = block_table[i][x]
                            sub_block_table_len = (seq_lens_list[i] - 1) // self.cache_config.block_size + 1
                            lf_prefill_indices_list.append(block_table[i][x:sub_block_table_len].flatten())
                            continue
                        else:
                            sub_block_table_len = (seq_lens_list[i] - 1) // self.cache_config.block_size + 1
                            lf_prefill_indices_list.append(block_table[i][:sub_block_table_len].flatten())
                            for layer_index in range(self.model.start_layer, self.model.end_layer):
                                lf1_caches = self.model.layers[layer_index].self_attn.lf_gate.lf1_caches
                                lf2_caches = self.model.layers[layer_index].self_attn.lf_gate.lf2_caches
                                lf1_caches[lf_prefill_indices_list[-1], ...].zero_()
                                lf2_caches[lf_prefill_indices_list[-1], ...].zero_()
                    if context_lens_tensor_list[i] == 0:
                        for layer_index in range(self.model.start_layer, self.model.end_layer):
                            lf1_caches = self.model.layers[layer_index].self_attn.lf_gate.lf1_caches
                            lf2_caches = self.model.layers[layer_index].self_attn.lf_gate.lf2_caches
                            lf1_caches[pre_indices[i], ...].zero_()
                            lf2_caches[pre_indices[i], ...].zero_()

                if self.cache_config.enable_prefix_caching:
                    lf_indices = torch.cat(lf_prefill_indices_list)
                    self.input_lf_loc[self.input_lf_len:lf_len].fill_(-1)
                    self.out_lf_loc[self.output_lf_len:lf_len * 2].zero_()
                    self.pre_lf_indexs[:pre_indices.shape[0]].copy_(pre_indices)
                    self.pre_lf_indexs[pre_indices.shape[0]:lf_len].fill_(-1)
                    self.out_lf_indexs[:lf_indices.shape[0]].copy_(lf_indices)
                    self.out_lf_indexs[lf_indices.shape[0]:lf_len * 2].fill_(-1)
                    self.input_lf_len = lf_len
                    self.output_lf_len = lf_len * 2
                else:
                    self.pre_lf_indexs[:pre_indices.shape[0]].copy_(pre_indices)
                    self.out_lf_indexs[:lf_indices.shape[0]].copy_(lf_indices)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[IntermediateTensors, torch.Tensor]:
        hidden_states = self.model(
                input_ids, positions,
                self.pre_lf_indexs[:self.input_lf_len],
                self.out_lf_indexs[:self.output_lf_len],
                self.input_lf_loc[:self.input_lf_len],
                self.out_lf_loc[:self.output_lf_len],
                self.inputs_loc[:self.input_len],
                self.outputs_loc[:self.input_len],
                intermediate_tensors,
                inputs_embeds,
        )

        return hidden_states

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states.to(self.lm_head_dtype))
        return logits

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
        })

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
