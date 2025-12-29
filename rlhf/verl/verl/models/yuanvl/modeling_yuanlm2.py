# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Yuan model."""
import math
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch
import torch.utils.checkpoint
from torch import einsum, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from verl.models.configs.configuration_yuan import YuanConfig
from einops import rearrange
from transformer_engine.pytorch import RMSNorm
import copy
try:
    import grouped_gemm as gg
except ImportError:
    gg = None
try:
    from flash_attn import flash_attn_varlen_func as flash_attn_unpadded_func
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_unpadded_func = None


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "YuanConfig"


class YuanRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, dtype=torch.float32, rotary_interleaved=False, seq_len_interpolation_factor=None):
        super().__init__()
        self.base = base
        self.dim = dim
        self.rotary_interleaved = rotary_interleaved
        self.seq_len_interpolation_factor = seq_len_interpolation_factor

    def get_rotary_seq_len(
            self,
            inference_param=None,
            transformer_input: torch.Tensor=None,
            transformer_config=None,
    ):
        if inference_param is not None:
            rotary_seq_len = inference_param.max_sequence_length
        else:
            rotary_seq_len = transformer_input.size[0]
        if transformer_config.sequence_parallel:
            rotary_seq_len *= transformer_config.tensor_model_parallel_size
        
        return rotary_seq_len

    def forward(self, max_seq_len, offset=0):

        """Forward pass of RoPE embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): _description_. Defaults to 0.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        inv_freq = (1.0 / ( self.base**(torch.arange(0, self.dim, 2, dtype=torch.float32, device=torch.cuda.current_device()) / self.dim))).to(torch.float32)
        
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


def _rotate_half(x, rotary_interleaved):
    """huggingface version
    change sign so the last dimension becomes [-odd, +even]
    
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x_new = torch.stack((-x2, x1), dim=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)

def apply_rotary_pos_emb(t, freqs, position_ids, rotary_interleaved=False):

    rot_dim = freqs.shape[-1]
    freqs = freqs[position_ids]
    freqs = freqs.view(t.shape[1],freqs.shape[1],freqs.shape[2],freqs.shape[4]).transpose(0,1)
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t_type = t.dtype
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)

    return torch.cat((t, t_pass), dim=-1)

class LocalizedFiltering(torch.nn.Module):
    """
    Mega's Exponential Moving Average layer, largely left unmodified from the original repo with the exception of
    variable names and moving away from the stateful representation of incremental decoding state. See
    "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(self, hidden_size, lf_conv2d_group, lf_conv2d_num_pad):
        super().__init__()

        self.embed_dim = hidden_size
        self.lf_conv2d_group = lf_conv2d_group
        self.lf_conv2d_num_pad = lf_conv2d_num_pad
        if self.lf_conv2d_num_pad == 1:
            self.training = True
        self.conv1 = torch.nn.Conv2d(self.embed_dim, self.embed_dim // 2, (2, 1), stride=(1, 1), padding=(self.lf_conv2d_num_pad, 0), groups=self.lf_conv2d_group)
        self.conv2 = torch.nn.Conv2d(self.embed_dim // 2, self.embed_dim, (2, 1), stride=(1, 1), padding=(self.lf_conv2d_num_pad, 0), groups=self.lf_conv2d_group)
        self.output_layernorm = RMSNorm(self.embed_dim, eps=1e-6)

    def _train_forward(self, inputs):
        inputs = inputs.transpose(0,1)
        seq_len, bsz, embed_dim = inputs.size()
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Unexpected embedding dimension received: input is {embed_dim}, model expects {self.embed_dim}"
            )
        residual = inputs

        inputs = inputs.view(seq_len, 1, bsz, embed_dim).permute(2, 3, 0, 1)
        output1 = self.conv1(inputs)
        output1 = output1[:, :, :seq_len, :]

        output2 = self.conv2(output1)
        output2 = output2[:, :, :seq_len, :].permute(2, 3, 0, 1).contiguous()
        output2 = output2.view(seq_len, bsz, embed_dim)
        assert output2.shape == residual.shape

        torch.cuda.set_device(output2.device)
        lf_output = self.output_layernorm(output2 + residual)
        lf_output = lf_output.transpose(0,1)
        return lf_output

    def _inference_forward(self, inputs, before_hidden_states):

        if before_hidden_states is None:
            residual = inputs
            seq_len, bsz, embed_dim = inputs.size()

            inputs = inputs.view(seq_len, 1, bsz, embed_dim).permute(2, 3, 0, 1)

            pad_zero1 = torch.zeros(bsz, embed_dim, 1, 1).to(inputs)
            inputs = torch.cat((pad_zero1, inputs), dim=2).contiguous()
            output1 = self.conv1(inputs)

            pad_zero2 = torch.zeros(bsz, embed_dim // 2, 1, 1).to(output1)
            output1 = torch.cat((pad_zero2, output1), dim=2).contiguous()
            output2 = self.conv2(output1)

            output2 = output2.permute(2, 3, 0, 1).contiguous()

            output2 = output2.view(seq_len, bsz, embed_dim)

            assert output2.shape == residual.shape

            lf_output = self.output_layernorm(output2 + residual)

        else:
            residual = inputs

            seq_len, bsz, embed_dim = inputs.size()
            seq_len_before, _, _ = before_hidden_states.size()

            assert seq_len == 1 and seq_len_before == 2

            inputs = torch.cat((before_hidden_states, inputs), dim=0)
            inputs = inputs.view(3, 1, bsz, embed_dim).permute(2, 3, 0, 1)

            output1 = self.conv1(inputs)
            output2 = self.conv2(output1)
            output2 = output2.view(1, bsz, embed_dim)

            assert output2.shape == residual.shape

            lf_output = self.output_layernorm(output2 + residual)

        return lf_output


    def forward(
        self,
        inputs,
        before_hidden_states = None,
    ) -> torch.Tensor:
        # assert self.lf_conv2d_num_pad == 1
        if self.training:
            lf_output = self._train_forward(inputs)
        else:
            lf_output = self._inference_forward(inputs, before_hidden_states)

        return lf_output


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class YuanRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        YuanRMSNorm is equivalent to LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# flash attn
class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        assert flash_attn_unpadded_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((i.is_cuda for i in (q,k,v)))

        batch_size, seqlen_q = q.shape[1], q.shape[0]
        seqlen_k = k.shape[0]
        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)
        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q
            is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q
            dropout_p = self.dropout_p
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=q.device) 
            dropout_p = 0

        output = flash_attn_unpadded_func(q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k, dropout_p, softmax_scale=self.softmax_scale, causal=is_causal)

        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output

class ParallelAttention_router(nn.Module):
    def __init__(self, config):
        super(ParallelAttention_router, self).__init__()
        layer_number=0
        self.layer_number = max(1, layer_number)
        
        self.hidden_size = config.hidden_size
        self.projection_size = config.moe_config['moe_num_experts']
        
        self.num_attention_router_heads = config.moe_config['num_attention_router_heads']
        self.hidden_size_per_attention_head = config.max_position_embeddings // self.num_attention_router_heads
        self.query_key_value = nn.Linear(self.hidden_size, self.projection_size*3, bias=False)

    def forward(self, hidden_states, attention_mask=None, enc_position_ids=None,
                encoder_output=None, inference_params=None,
                rotary_pos_emb=None):
        is_first_step = False
        before_hidden_states = None
        
        mixed_x_layer = self.query_key_value(hidden_states)
        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, self.projection_size, -1)
        b, s, z = query_layer.shape

        # use fp32 router
        query_layer = query_layer.float().view(b,s,z,1)
        key_layer = key_layer.float().view(b,s,z,1)
        value_layer = value_layer.float().view(b,s,z,1)

        attn_weights = torch.matmul(query_layer, key_layer.transpose(2, 3))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_layer)
        router_output = attn_output.view(-1, z)
        return router_output 

class YuanExpertMLP(nn.Module):
    def __init__(self, config):
        super(YuanExpertMLP, self).__init__()
        self.gated_linear_unit = config.moe_config['gated_linear_unit']
        self.ffn_hidden_size = config.ffn_hidden_size


        if self.gated_linear_unit:
            self.w1 = nn.Linear(config.hidden_size, self.ffn_hidden_size*2, bias=False)
              
        else:
            self.w1 = nn.Linear(config.hidden_size, self.ffn_hidden_size, bias=False)
            
        self.act_fn = ACT2FN[config.hidden_act]
        self.w2 = nn.Linear(self.ffn_hidden_size, config.hidden_size, bias=False)
        

    def forward(self, x):
        x = self.w1(x)
        if self.gated_linear_unit:
            x = torch.chunk(x, 2, dim=-1)
            x = self.act_fn(x[0]) * x[1]
        else:
            x = self.act_fn(x)  
        x = self.w2(x)  
        return x
    


class YuanMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str
    ):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.act_fn(self.up_proj(x)))
        

class YuanAttention(nn.Module):
    """Localized Filtering-based Attention 'YUAN 2.0: A Large Language Model with Localized Filtering-based Attention' paper"""

    def __init__(self, config: YuanConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.lf_conv2d_group = config.lf_conv2d_group
        self.lf_conv2d_num_pad = config.lf_conv2d_num_pad

        try:
            self.attention_projection_size = config.attention_projection_size
        except:
            self.attention_projection_size = None
        
        if self.attention_projection_size is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = self.attention_projection_size // self.num_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.causal_mask = config.causal_mask
        self.attn_mask_type = config.attn_mask_type
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        self.use_flash_attention = config.use_flash_attention
        try:
            self.use_shareqk = config.use_shareqk
        except Exception as e:
            self.use_shareqk=False
        self.dropout = 0.0
        self.attention_projection_size = config.attention_projection_size
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        if self.use_shareqk:
            self.qk_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.qk_weight = nn.Parameter(torch.Tensor(2, self.hidden_size))
            self.qk_bias = nn.Parameter(torch.Tensor(2, self.hidden_size))
        else:
            self.lf_gate = LocalizedFiltering(self.hidden_size, self.lf_conv2d_group, self.lf_conv2d_num_pad)
            self.get_query_key = nn.Linear(self.hidden_size, 2 * self.attention_projection_size, bias=False)
        self.core_attention = FlashSelfAttention(causal=True, attention_dropout=config.attn_dropout, softmax_scale=self.softmax_scale)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_k: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        q_len, bsz, _ = hidden_states.size()
        hidden_states = hidden_states
        is_first_step = False
        if use_cache:
            if past_key_value is None:
                before_hidden_states = None
                is_first_step = True
                if q_len > 1:
                    inference_hidden_states_memory = hidden_states[-2:, :, :]
                else:
                    inference_hidden_states_memory = torch.cat((torch.zeros_like(hidden_states), hidden_states), dim=0)
            else:
                before_hidden_states = past_key_value[2]
                inference_hidden_states_memory = torch.cat((before_hidden_states[-1:, :, :], hidden_states), dim=0)
        value_states = self.v_proj(hidden_states).view(q_len, bsz, self.num_heads, self.head_dim)
        if self.use_shareqk:
            qk_states = self.qk_proj(hidden_states).view(q_len, bsz, self.num_heads*self.head_dim)
            query_key = qk_states.unsqueeze(2) * self.qk_weight + self.qk_bias
            query_states, key_states = torch.unbind(query_key, dim=2)

            query_states = query_states.view(q_len, bsz, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(q_len, bsz, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            hidden_states = self.lf_gate(hidden_states, before_hidden_states)
            mixed_qk_layer = self.get_query_key(hidden_states)
            new_tensor_shape = mixed_qk_layer.size()[:-1] + (self.num_heads, 2 * self.head_dim)
            mixed_qk_layer = mixed_qk_layer.view(*new_tensor_shape)
            (query_states, key_states) = torch.split(mixed_qk_layer, self.head_dim, dim=-1)

        
        kv_seq_len = key_states.shape[1]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[1]
        
        # duplicate the pos_emb for self attention
        if rotary_pos_emb is not None:
            if position_ids.shape[1] == 1:
                q_seq_start = position_ids[0,-1]
                q_seq_end = q_seq_start + 1
                k_seq_end = q_seq_end
            else:
                q_seq_start = 0
                q_seq_end = q_seq_start+key_states.shape[0]
                k_seq_end = q_seq_end
            
            rotary_pos_shape = rotary_pos_emb.shape
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = ((rotary_pos_emb,) * 2)
            q_pos_emb, k_pos_emb = rotary_pos_emb
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=0)
            value_states = torch.cat([past_key_value[1], value_states], dim=0)
        past_key_value = (key_states, value_states, inference_hidden_states_memory) if use_cache else None
        query_states = apply_rotary_pos_emb(query_states, q_pos_emb, position_ids)
        key_states = apply_rotary_pos_emb(key_states, k_pos_emb, position_ids_k)

        attn_weights = None
        attn_output = self.core_attention(query_states, key_states, value_states)
        q_len, bsz, _, _ = attn_output.shape
        attn_output = attn_output.reshape(q_len, bsz, -1)
        
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value

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
        unpermuted_local_bias = None
        output_total = unpermuted_local_hidden
        output_bias_total = unpermuted_local_bias

        if self.router_topk > 1:
            global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1]
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

class GroupedMLP(nn.Module):
    """An efficient implementation of the Experts layer using CUTLASS GroupedGEMM.

    This class is designed to execute multiple experts in parallel, thereby maximizing computational efficiency.
    """

    def __init__(self, num_experts: int, config: YuanConfig):
        super().__init__()
        self.num_experts = num_experts
        self.config = config

        def glu(x):
            x = torch.chunk(x, 2, dim=-1)
            return torch.nn.functional.silu(x[0]) * x[1]

        self.activation_func = glu
        self.ffn_hidden_size = config.ffn_hidden_size
        fc1_output_size_per_partition = self.ffn_hidden_size * 2
        fc2_input_size = self.ffn_hidden_size
        
        self.w1 = nn.ModuleList([nn.Linear(self.config.hidden_size, self.ffn_hidden_size * 2, bias=False) for _ in range(num_experts)])
        self.w2 = nn.ModuleList([nn.Linear(self.ffn_hidden_size, self.config.hidden_size, bias=False) for _ in range(num_experts)])
    def forward(self, permuted_hidden_states, tokens_per_expert):
        torch.cuda.set_device(permuted_hidden_states.device)
        permuted_hidden_states = permuted_hidden_states

        fc2_outputs = []
        start_idx = 0
        for i in range(self.num_experts):
            if tokens_per_expert[i] == 0:
                continue
            end_idx = start_idx + tokens_per_expert[i]
            # Use custom attributes for each expert's Linear layers
            
            fc1_output = self.w1[i](permuted_hidden_states[start_idx:end_idx])
            intermediate_parallel = self.activation_func(fc1_output)
            fc2_output = self.w2[i](intermediate_parallel)
            fc2_outputs.append(fc2_output)
            start_idx = end_idx
        fc2_output = torch.cat(fc2_outputs, dim=0)
        return fc2_output

class YuanMoeLayer(nn.Module):
    def __init__(self, config:YuanConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.moe_config['moe_num_experts']
        self.top_k = config.moe_config['moe_top_k']
        self.norm_topk_prob = config.moe_config['norm_topk_prob']
        self.hidden_size = config.hidden_size
        
        expert_indices_offset = (0)

        self.router = ParallelAttention_router(config)
        self.token_dispatcher = MoEDroplessTokenDispatcher(self.num_experts, config=self.config)
        self.experts = GroupedMLP(self.num_experts, self.config)

    def routing(self, logits: torch.Tensor) -> torch.Tensor:
        top_logits, indices = torch.topk(logits, k=self.top_k, dim=1)
        scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)
        return scores, indices

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        logits = self.router(hidden_states)
        scores, indices = self.routing(logits)
        scores = scores.to(hidden_states.dtype)
        (dispatched_input, tokens_per_expert, scores, indices, global_local_map, ) = self.token_dispatcher.token_permutation(hidden_states, scores, indices)
        expert_output = self.experts(dispatched_input, tokens_per_expert)
        output = self.token_dispatcher.token_unpermutation(expert_output, scores, indices, global_local_map)
        return output

class YuanDecoderLayer(nn.Module):
    def __init__(self, config: YuanConfig, num_layer):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = YuanAttention(config=config)
        self.num_layer = num_layer
        
        if config.moe_config['moe_num_experts'] > 0:
            self.mlp = YuanMoeLayer(config)
        else:
            self.mlp = YuanMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
        
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_k: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        torch.cuda.set_device(hidden_states.device)
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_ids_k=position_ids_k,
            past_key_value=past_key_value,
            rotary_pos_emb=rotary_pos_emb,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = residual + hidden_states.permute(1, 0, 2)

        # Fully Connected
        residual = hidden_states
        torch.cuda.set_device(hidden_states.device)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)# .to('cuda:1')
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
    
        return outputs


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
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

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

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, YuanModel):
            module.gradient_checkpointing = value


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

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
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
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
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
        
        #TODO: control it by config
        self.eod_token = config.eod_token
        self.reset_attention_mask = config.reset_attention_mask
        self.reset_position_ids = config.reset_position_ids
        self.max_position_embeddings = config.max_position_embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([YuanDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        self.seq_length = config.max_position_embeddings
        rotary_dim = config.hidden_size // config.num_attention_heads
        if config.rotary_percent < 1.0:
            rotary_dim = int(rotary_dim * config.rotary_percent)
        self.rotary_pos_emb = YuanRotaryEmbedding(rotary_dim, base=config.rotary_base, dtype=config.torch_dtype)


    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def _prepare_decoder_attention_mask_training(self, input_id, inputs_embeds, eod_token, reset_mask_flag ,reset_attention_mask=True, reset_position_ids=True):
    
        micro_batch_size, seq_length = input_id.size()
        
        attention_mask = torch.tril(torch.ones(
            (micro_batch_size, seq_length, seq_length), device=inputs_embeds.device)).view(
                micro_batch_size, 1, seq_length, seq_length)    
                
        position_ids = torch.arange(seq_length, dtype=torch.long,
                                    device=inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_id)
               
        if reset_position_ids:
            position_ids = position_ids.clone()
        
        if reset_position_ids or reset_attention_mask:
            # Loop through the batches:
            for b in range(micro_batch_size):

                # Find indecies where EOD token is.
                eod_index = position_ids[b, input_id[b] == eod_token]

                # Detach indecies from positions if going to modify positions.
                if reset_position_ids:
                    eod_index = eod_index.clone()
                # Loop through EOD indecies:
                prev_index = 0
                for j in range(eod_index.size()[0]):
                    i = eod_index[j]
                    # Mask attention loss.
                    if reset_attention_mask:
                        attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                    # Reset positions.
                    if reset_position_ids:
                        position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                        prev_index = i + 1
                        
        inverted_mask = 1 - attention_mask   
        output_attn_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min)
        if reset_mask_flag:
            output_attn_mask = output_attn_mask[:,:,-1:,:]
        return output_attn_mask, position_ids

    @add_start_docstrings_to_model_forward(YUAN_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast, torch.Tensor]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids1 = copy.deepcopy(input_ids)
        reset_mask_flag = False
        if past_key_values:
            input_ids = input_ids
            input_ids = input_ids[:,-1:]
            if use_cache:
                reset_mask_flag = True
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = input_ids
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds.transpose(0,1)
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

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

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            pass
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).transpose(0,1)
        
        if self.training or self.reset_position_ids:
            attention_mask, _ = self._prepare_decoder_attention_mask_training(input_ids1, inputs_embeds, self.eod_token, reset_mask_flag, self.reset_attention_mask, self.reset_position_ids)
        else: 
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
                )
                attention_mask = self._prepare_decoder_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        '''
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                transformer_input=inputs_embeds
            )
        '''
        rotary_pos_emb = self.rotary_pos_emb(self.max_position_embeddings)

        hidden_states = inputs_embeds
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        position_ids = position_ids.cpu()
        position_ids_k = position_ids_k.cpu()
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_ids_k=position_ids_k,
                    past_key_value=past_key_value,
                    rotary_pos_emb=rotary_pos_emb,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = hidden_states
        torch.cuda.set_device(hidden_states.device)
        hidden_states = self.norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class YuanForCausalLM(YuanPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        '''
        self.eod_token = config.eod_token
        self.sep_token = config.sep_token
        self.use_loss_mask = config.use_loss_mask
        self.model = YuanModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        '''
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model = YuanModel(config)
        # self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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

    def get_loss_mask(self, input_ids, labels, eod_token, sep_token):
        micro_batch_size, seq_length = input_ids.size()
        loss_mask = torch.ones(input_ids.size(), dtype=torch.float, device=input_ids.device)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids) 


        """modify loss_mask to only calculate the loss of the answer (separated with [SEP])"""

        for b in range(micro_batch_size):
            eod_indexs = position_ids[b, input_ids[b] == eod_token]
            sep_indexs = position_ids[b, input_ids[b] == sep_token]

            if len(eod_indexs) == 0 or len(sep_indexs) == 0:
                loss_mask[b] = 1.0
            else:
                if eod_indexs[0] > sep_indexs[0]:
                    loss_mask[b, 0:sep_indexs[0]] = 0

                    if len(eod_indexs) == len(sep_indexs):
                        for ii, eod_index in enumerate(eod_indexs):
                            start_index = eod_index
                            if ii == (len(sep_indexs) - 1):
                                stop_index = seq_length
                            else:
                                stop_index = sep_indexs[ii + 1]
                            loss_mask[b, start_index:stop_index] = 0.0
                    else:
                        if len(eod_indexs) > len(sep_indexs):
                            loss_mask[b,:] = 1.0
                        else:
                            for ii, eod_index in enumerate(eod_indexs):
                                start_index = eod_index
                                stop_index = sep_indexs[ii + 1]

                                loss_mask[b, start_index:stop_index] = 0.0

                elif eod_indexs[0] < sep_indexs[0]:

                    if len(eod_indexs) == len(sep_indexs):
                        for ii, eod_index in enumerate(eod_indexs):
                            start_index = eod_index
                            stop_index = sep_indexs[ii]
                            loss_mask[b, start_index:stop_index] = 0.0

                    else:
                        if len(eod_indexs) < len(sep_indexs):
                            loss_mask[b,:] = 1.0
                        else:
                            for ii, eod_index in enumerate(eod_indexs):
                                start_index = eod_index
                                if ii >= len(sep_indexs):
                                    stop_index = seq_length
                                else:
                                    stop_index = sep_indexs[ii]
                                loss_mask[b, start_index:stop_index] = 0.0

        loss_mask[input_ids == eod_token] = 1.0
        return loss_mask
    @add_start_docstrings_to_model_forward(YUAN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, YuanForCausalLM

        >>> model = YuanForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0].transpose(0,1)
        
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            if self.use_loss_mask:
                loss_mask = self.get_loss_mask(input_ids, labels, self.eod_token, self.sep_token)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            if self.use_loss_mask:
                loss_fct = CrossEntropyLoss(reduction='none')
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = torch.sum(loss * loss_mask) / loss_mask.sum()
            else:
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


@add_start_docstrings(
    """
    The Yuan Model transformer with a sequence classification head on top (linear layer).

    [`YuanForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    YUAN_START_DOCSTRING,
)
class YuanForSequenceClassification(YuanPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = YuanModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(YUAN_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )



