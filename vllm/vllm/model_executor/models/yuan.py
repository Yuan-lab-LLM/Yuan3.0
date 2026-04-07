"""Inference-only Yuan model compatible with HuggingFace weights."""
import os
import re
from typing import Iterable, Optional, Tuple, Union
import math
import numpy as np
import torch
from torch import nn, Tensor

import vllm.envs as envs
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.config import CacheConfig, VllmConfig, CUDAGraphMode
from transformers.activations import ACT2FN
from vllm.model_executor.layers.attention.encoder_only_attention import Attention
from vllm.model_executor.layers.linear import  (ColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead, DEFAULT_VOCAB_PADDING_SIZE)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.distributed import (get_ep_group, get_pp_group,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_all_gather)
from vllm.distributed.utils import get_pp_indices
from .interfaces import SupportsPP, MixtureOfExperts
from .utils import (AutoWeightsLoader, PPMissingLayer,
                    is_pp_missing_parameter, make_layers,
                    maybe_prefix)

from vllm.model_executor.layers.layernorm import RMSNorm as VLLM_RMSNorm
from vllm.model_executor.layers.fused_moe import fused_topk_v2, FusedMoE
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.compilation.decorators import support_torch_compile
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
try:
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadata
except:
    pass


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


class YuanMoeLayer(nn.Module):
    def __init__(self,
                 vllm_config: VllmConfig,
                 num_experts: int,
                 prefix: str = ""):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.num_experts = num_experts
        self.top_k = config.moe_config['moe_top_k']

        if config.moe_config['router_type'] == 'attn_router':
            self.router = ParallelAttention_router(config, self.num_experts)
        else:
            self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)

        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb
        if 'per_layer_experts_blocks' in config.moe_config:
            assert config.moe_config['per_layer_experts_blocks'] != None
            self.n_logical_experts = max(config.moe_config['per_layer_experts_blocks'])
            self.n_routed_experts = self.n_logical_experts
        else:
            self.n_logical_experts = num_experts
            self.n_routed_experts = num_experts

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = (self.n_logical_experts +
                                   self.n_redundant_experts)
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        if self.enable_eplb:
            real_n_redundant_experts = self.n_redundant_experts + self.n_logical_experts - num_experts
        else:
            real_n_redundant_experts = 0
        
        self.experts = FusedMoE(num_experts=self.num_experts,
                                top_k=self.top_k,
                                hidden_size=config.hidden_size,
                                intermediate_size=config.moe_config['ffn_hidden_size'],
                                reduce_results=False,
                                renormalize=config.moe_config["norm_topk_prob"],
                                quant_config=quant_config,
                                prefix=f"{prefix}.experts",
                                is_sequence_parallel=self.is_sequence_parallel,
                                enable_eplb=self.enable_eplb,
                                num_redundant_experts=real_n_redundant_experts,
                                custom_routing_function=fused_topk_v2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        logits = self.router(hidden_states)
        final_hidden_states = self.experts(hidden_states, logits)
        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0)
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )

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
        inv_freq = (1.0 / ( self.base**(
                            torch.arange(0, self.dim, 2, dtype=torch.float32,
                            device=torch.cuda.current_device()) / self.dim)
                        )
                    ).to(torch.float32)
        
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

        self.start_layer, self.end_layer = get_pp_indices(
            config.num_hidden_layers,
            get_pp_group().rank_in_group,
            get_pp_group().world_size
        )

        if cache_config.cache_dtype == "fp8" or cache_config.cache_dtype == "int8":
            scale = 2
        else:
            scale = 1
        self.lf_cache_size_gibs = float(os.environ.get('LF_CACHE_SIZE_GIBS', 2))
        self.lf_cache_nums = int(self.lf_cache_size_gibs * 2**30 * scale / (self.embed_dim * 3 / self.tp_size * (self.end_layer - self.start_layer)))
        self.lf1_caches = torch.zeros((self.lf_cache_nums, self.embed_dim // self.tp_size), dtype=params_dtype, device="cuda")
        self.lf2_caches = torch.zeros((self.lf_cache_nums, self.embed_dim // 2 // self.tp_size), dtype=params_dtype, device="cuda")

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
        inputs_t[inputs_loc] = inputs
        inputs_t[input_lf_loc] = lf_cache
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
            kv_cache: torch.Tensor,
        ):
        assert pre_lf_indexs.shape == input_lf_loc.shape
        assert out_lf_indexs.shape == out_lf_loc.shape
        assert kv_cache.numel() == 0 or self.lf_cache_nums > kv_cache.shape[1], \
                f"please increase env LF_CACHE_SIZE_GIBS with {self.lf_cache_size_gibs * kv_cache.shape[1] / self.lf_cache_nums + 0.1:.2f}"
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
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        import torch.distributed as dist
        self.rank = dist.get_rank()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        cache_config=vllm_config.cache_config
        hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = getattr(config, 'num_kv_heads', self.total_num_heads)
        tp_size = get_tensor_model_parallel_world_size()

        attention_projection_size = getattr(config, 'attention_projection_size', config.hidden_size)
        self.attn_head_size = attention_projection_size // self.total_num_heads

        self.num_heads = (self.total_num_heads + tp_size - 1) // tp_size  
        self.num_kv_heads = max(1, (self.total_num_kv_heads + tp_size - 1) // tp_size)

        self.head_dim = config.attention_projection_size // self.total_num_heads \
            if hasattr(config, 'attention_projection_size') else hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        
        self.eps = 1e-6
        self.get_query_key = ColumnParallelLinear(
            hidden_size,
            (self.q_size + self.kv_size) * tp_size,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.get_query_key",
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.kv_size * tp_size,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.o_proj = RowParallelLinear(
            self.q_size * tp_size,
            hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
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
        kv_cache = self.attn.kv_cache[0]
        hidden_states = self.lf_gate(hidden_states, pre_lf_indexs, out_lf_indexs, input_lf_loc, out_lf_loc, inputs_loc, outputs_loc, kv_cache)

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
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.self_attn = YuanAttention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
        )
        self.use_moe = getattr(config, "use_moe", False)
        if self.use_moe:
            layer_idx = int(prefix.split(".")[-1])
            if 'per_layer_experts_blocks' in config.moe_config:
                assert config.moe_config['per_layer_experts_blocks'] != None
                num_experts = config.moe_config['per_layer_experts_blocks'][layer_idx]
                self.max_experts = max(config.moe_config['per_layer_experts_blocks'])
            elif 'moe_num_experts' in config.moe_config:
                assert config.moe_config['moe_num_experts'] != None
                num_experts = config.moe_config['moe_num_experts']
                self.max_experts = num_experts
            else:
                raise ValueError(f'per_layer_experts_blocks or moe_num_experts must in config.moe_config')
            self.mlp = YuanMoeLayer(vllm_config=vllm_config, num_experts=num_experts, prefix=prefix)
        else:
            self.mlp = YuanMLP(
                hidden_size=config.hidden_size,
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
        self.config = config
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()
 
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
            lambda prefix: YuanDecoderLayer(vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = VLLM_RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts
        self.enable_eplb = parallel_config.enable_eplb
        if 'per_layer_experts_blocks' in self.config.moe_config:
            assert self.config.moe_config['per_layer_experts_blocks'] != None
            self.max_num_experts = max(self.config.moe_config['per_layer_experts_blocks'][self.start_layer:self.end_layer])
        elif 'moe_num_experts' in self.config.moe_config:
            assert self.config.moe_config['moe_num_experts'] != None
            self.max_num_experts = self.config.moe_config['moe_num_experts']
        else:
            raise ValueError(f'per_layer_experts_blocks or moe_num_experts must in config.moe_config')

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
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
                hidden_states = self.embed_input_ids(input_ids)
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
 
                if param_name not in params_dict:
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
                if self.enable_eplb:
                    if expert_id % num_experts < self.max_num_experts - num_experts + self.num_redundant_experts:
                        for k in range(1, math.ceil((self.max_num_experts + self.num_redundant_experts) / num_experts)):
                            redundant_expert_id = k * num_experts + expert_id
                            if "w1" in param_name:
                                weight_loader = param.weight_loader
                                gate, up = loaded_weight.chunk(2)
                                weight_loader(param, gate, param_name, "w1", redundant_expert_id)
                                weight_loader(param, up, param_name, "w3", redundant_expert_id)
                            elif "w2" in param_name:
                                weight_loader = param.weight_loader
                                weight_loader(param, loaded_weight, param_name, "w2", redundant_expert_id)
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


class YuanForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):

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

        # Set MoE hyperparameters
        self.expert_weights = []

        self.moe_layers: list[FusedMoE] = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, YuanDecoderLayer)
            if isinstance(layer.mlp, YuanMoeLayer):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            raise RuntimeError("No Yuan layer found in the model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

        self.num_routed_experts_list = []
        if 'per_layer_experts_blocks' in self.config.moe_config:
            assert self.config.moe_config['per_layer_experts_blocks'] != None
            self.num_routed_experts_list = self.config.moe_config['per_layer_experts_blocks'][self.start_layer:self.end_layer]

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.cache_config = vllm_config.cache_config
        self.full_cuda_graph = \
            vllm_config.compilation_config.cudagraph_mode in [CUDAGraphMode.FULL]
        print("self.full_cuda_graph: ", self.full_cuda_graph, flush=True)
        self.backend = vllm_config.attention_config.backend

        inputs_len = vllm_config.scheduler_config.max_num_batched_tokens
        self.pre_lf_indexs = torch.zeros(inputs_len, dtype=torch.long, device="cuda")
        self.out_lf_indexs = torch.zeros(inputs_len, dtype=torch.long, device="cuda")
        self.input_lf_loc = torch.zeros(inputs_len, dtype=torch.long, device="cuda")
        self.out_lf_loc = torch.zeros(inputs_len, dtype=torch.long, device="cuda")
        self.inputs_loc = torch.zeros(inputs_len, dtype=torch.long, device="cuda")
        self.outputs_loc = torch.zeros(inputs_len, dtype=torch.long, device="cuda")

    def get_lf_index(self, input_ids):
        attn_metadata = get_forward_context().attn_metadata
        lf_len = input_ids.shape[0]
        if attn_metadata is None:
            self.input_lf_len = lf_len
            self.input_len = lf_len
            self.output_lf_len = lf_len

            self.pre_lf_indexs[:lf_len].fill_(-1)
            self.out_lf_indexs[:lf_len].fill_(-1)
            self.input_lf_loc[:lf_len].fill_(-1)
            self.out_lf_loc[:lf_len].zero_()
            self.inputs_loc[:lf_len].zero_()
            self.outputs_loc[:lf_len].zero_()
            return

        if isinstance(attn_metadata, dict):
            attn_metadata = list(attn_metadata.values())[0]

        if self.backend == AttentionBackendEnum.FLASHINFER \
                and isinstance(attn_metadata, FlashInferMetadata):
            # v1: use backend flashinfer
            block_table = attn_metadata.block_table_tensor
            seq_lens = attn_metadata.seq_lens
            padding_reqs = (seq_lens == 0).sum().item()
            num_reqs = seq_lens.shape[0] - padding_reqs
            input_len = input_ids.shape[0]
            max_query_len = attn_metadata.max_q_len,
            num_actual_tokens = attn_metadata.num_actual_tokens - padding_reqs
            query_lens = attn_metadata.qo_indptr_gpu[1:] - attn_metadata.qo_indptr_gpu[:-1]
        elif isinstance(attn_metadata, FlashAttentionMetadata) \
                or isinstance(attn_metadata, TritonAttentionMetadata):
            # v1: use backend flashattn
            seq_lens = attn_metadata.seq_lens
            padding_reqs = (seq_lens == 0).sum().item()
            num_reqs = seq_lens.shape[0] - padding_reqs
            seq_lens = seq_lens[:num_reqs]
            block_table = attn_metadata.block_table[:num_reqs]
            input_len = input_ids.shape[0]
            max_query_len = attn_metadata.max_query_len
            num_actual_tokens = attn_metadata.num_actual_tokens
            query_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
        else:
            assert False, f"Now not support {type(attn_metadata)}!"

        seq_lens = seq_lens[:num_reqs]
        block_table = block_table[:num_reqs]
        query_lens = query_lens[:num_reqs]
        indices_1 = torch.clamp_min((seq_lens - 2) // self.cache_config.block_size, 0)
        pre_indices = torch.gather(block_table, dim=1, index=indices_1.long().unsqueeze(1)).squeeze()
        pre_indices = pre_indices.view(pre_indices.numel())
        indices_2 = (seq_lens - 1) // self.cache_config.block_size
        lf_indices = torch.gather(block_table , dim=1, index=indices_2.long().unsqueeze(1)).squeeze()
        lf_indices = lf_indices.view(lf_indices.numel())
        # in cudagraph mode, prefill inputs_ids will padding with 0
        if max_query_len == 1:
            # decode
            padding = input_len - num_reqs
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
            padding = input_len - num_actual_tokens
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

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

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

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        for layer_idx, layer in enumerate(self.moe_layers):
            # Register the expert weights.
            self.expert_weights.append(layer.get_expert_weights())
            layer.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = (num_physical_experts -
                                      self.num_logical_experts)
        for _, layer in enumerate(self.model.layers):
            if isinstance(layer.mlp, YuanMoeLayer):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
