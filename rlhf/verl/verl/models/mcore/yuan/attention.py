from abc import ABC
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import torch
from torch import Tensor


from megatron.core import parallel_state, tensor_parallel, mpu
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_with_cos_sin,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.attention import Attention, SelfAttention
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
)
from megatron.core.utils import deprecate_inference_params, divide


from verl.utils.megatron_utils import TransformerConfig
try:
    from einops import rearrange
except ImportError:
    rearrange = None

@dataclass
class YuanSelfAttentionSubmodules:
    """
    Configuration class for specifying the submodules of a self-attention.
    """
    linear_qkv: Union[ModuleSpec, type] = None
    linear_qk: Union[ModuleSpec, type] = None
    linear_v: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None
    lf_gate: Union[ModuleSpec, type] = None

@dataclass
class LocalizedFilteringSubmodules:
    conv1: Union[ModuleSpec, type] = None
    conv2: Union[ModuleSpec, type] = None
    output_layernorm: Union[ModuleSpec, type] = None

class LocalizedFiltering(MegatronModule, ABC):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: LocalizedFilteringSubmodules
    ):
        super().__init__(config=config)

        self.config = config

        self.embed_dim = self.config.hidden_size
        self.lf_conv2d_group = self.config.lf_conv2d_group
        self.lf_conv2d_num_pad = self.config.lf_conv2d_num_pad
        self.lf_conv2d_add_bias = self.config.lf_conv2d_add_bias                                                                    
        self.sequence_parallel = self.config.sequence_parallel
        if get_cuda_rng_tracker().is_initialized():
            with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
                self.conv1 = build_module(
                    submodules.conv1,
                    in_channels=self.embed_dim,
                    out_channels=self.embed_dim // 2,
                    kernel_size=(2, 1),
                    stride=(1, 1),
                    padding=(self.lf_conv2d_num_pad, 0),
                    groups=self.lf_conv2d_group,
                    bias=self.lf_conv2d_add_bias
                )
        else:
            self.conv1 = build_module(
                submodules.conv1,
                in_channels=self.embed_dim,
                out_channels=self.embed_dim // 2,
                kernel_size=(2, 1),
                stride=(1, 1),
                padding=(self.lf_conv2d_num_pad, 0),
                groups=self.lf_conv2d_group,
                bias=self.lf_conv2d_add_bias
            )
        if get_cuda_rng_tracker().is_initialized():
            with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
                self.conv2 = build_module(
                    submodules.conv2,
                    in_channels=self.embed_dim // 2,
                    out_channels=self.embed_dim,
                    kernel_size=(2, 1),
                    stride=(1, 1),
                    padding=(self.lf_conv2d_num_pad, 0),
                    groups=self.lf_conv2d_group,
                    bias=self.lf_conv2d_add_bias
                )
        else:
            self.conv2 = build_module(
                submodules.conv2,
                in_channels=self.embed_dim // 2,
                out_channels=self.embed_dim,
                kernel_size=(2, 1),
                stride=(1, 1),
                padding=(self.lf_conv2d_num_pad, 0),
                groups=self.lf_conv2d_group,
                bias=self.lf_conv2d_add_bias
            )
        setattr(self.conv1.weight, 'sequence_parallel', self.sequence_parallel)
        setattr(self.conv2.weight, 'sequence_parallel', self.sequence_parallel)
        self.output_layernorm = build_module(
                submodules.output_layernorm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon
        )
        setattr(self.output_layernorm.weight, 'sequence_parallel', self.sequence_parallel)

    def _train_forward(self, inputs):
        seq_len, bsz, embed_dim = inputs.size()
        if embed_dim != self.embed_dim:
            raise ValueError(f"Unexpected embedding dimension received: input is {embed_dim}, model expects {self.embed_dim}")
        residual = inputs
        inputs = inputs.view(seq_len, 1, bsz, embed_dim).permute(2, 3, 0, 1)
        output1 = self.conv1(inputs)
        output1 = output1[:, :, :seq_len, :]
        output2 = self.conv2(output1)
        output2 = output2[:, :, :seq_len, :].permute(2, 3, 0, 1).contiguous()
        output2 = output2.view(seq_len, bsz, embed_dim)
        assert output2.shape == residual.shape
        lf_output = self.output_layernorm(output2 + residual)
        return lf_output

    def forward(
        self,
        inputs,
        before_hidden_states=None
    ) -> torch.Tensor:
        lf_output = self._train_forward(inputs)
        return lf_output



class YuanSelfAttention(Attention):
    """Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: YuanSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
        )
        '''
        self.get_value = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            self.kv_projection_size,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False
        )

        self.get_query_key = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            self.query_projection_size + self.kv_projection_size,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False)
        '''
        self.query_projection_size = config.kv_channels * config.num_attention_heads
        self.kv_projection_size = config.kv_channels * config.num_key_value_heads
        self.linear_qk = build_module(
            submodules.linear_qk,
            self.config.hidden_size,
            self.query_projection_size + self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qk',
        )
        self.linear_v = build_module(
            submodules.linear_v,
            self.config.hidden_size,
            self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='v',
        )

        self.lf_gate = build_module(
            submodules.lf_gate,
            config=config
        )
        '''
        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
        )

        self.checkpoint_core_attention = config.recompute_granularity == 'selective'

        if self.use_flash_attn:
            self.core_attention_flash = FlashSelfAttention(
                causal=True, attention_dropout=self.flash_attn_drop
            )

        # Output.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )
        '''


    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:

            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:

            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value

    def run_realtime_tests(self):
        """Performs a consistency check.

        This function makes sure that tensors across devices are the same during an experiment.
        This is often not guaranteed to be so because of silent hardware failures (eg, memory
        corruption loading a checkpoint, network traffic corruption encountered during
        data transmission).

        (TODO) In the future, more tensors should be checked across the training run and
        checked every X iterations. This is left for future work. Equality of tensors is probably
        not required; transmitting hashes is sufficient."""

        if not self.config.qk_layernorm:
            return

        # check that all tensor parallel and data parallel ranks have the same
        # Q & K layernorm parameters.
        rank = get_data_parallel_rank()
        inputs = torch.stack(
            [
                self.q_layernorm.weight.data,
                self.q_layernorm.bias.data,
                self.k_layernorm.weight.data,
                self.k_layernorm.bias.data,
            ]
        )
        dp_list = [torch.empty_like(inputs) for _ in range(get_data_parallel_world_size())]
        dp_list[rank] = inputs
        torch.distributed.all_gather(dp_list, inputs, group=get_data_parallel_group())

        def _compare(srcs, tgts, names, parallelism):
            assert len(srcs) == len(tgts) == len(names)
            for src, tgt, name in zip(srcs, tgts, names):
                assert torch.all(src == tgt), (
                    f"Discrepancy between {name} in {parallelism} ranks {i} and {rank}. "
                    f"Diff: {torch.norm(src - tgt)}"
                )

        for i, dp in enumerate(dp_list):
            q_w, q_b, k_w, k_b = torch.unbind(dp)
            _compare(
                [q_w, q_b, k_w, k_b],
                [
                    self.q_layernorm.weight.data,
                    self.q_layernorm.bias.data,
                    self.k_layernorm.weight.data,
                    self.k_layernorm.bias.data,
                ],
                ["q_w", "q_b", "k_w", "k_b"],
                "DP",
            )

        rank = get_tensor_model_parallel_rank()
        tp_list = [torch.empty_like(inputs) for _ in range(get_tensor_model_parallel_world_size())]
        tp_list[rank] = inputs
        torch.distributed.all_gather(tp_list, inputs, group=get_tensor_model_parallel_group())


        for i, tp in enumerate(tp_list):
            q_w, q_b, k_w, k_b = torch.unbind(tp)
            _compare(
                [q_w, q_b, k_w, k_b],
                [
                    self.q_layernorm.weight.data,
                    self.q_layernorm.bias.data,
                    self.k_layernorm.weight.data,
                    self.k_layernorm.bias.data,
                ],
                ["q_w", "q_b", "k_w", "k_b"],
                "TP",
            )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform a forward pass through the attention module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) Attention output and bias.

        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if inference_context and inference_context.is_dynamic_batching():
            assert (
                flash_decode_and_prefill_kernel is not None
            ), "Internal use only: install package `nvidia_chunked_flash_attn`."

        # hidden_states: [sq, b, h]
        if self.config.flash_decode and not self.training and inference_context is not None:
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================

        # This branch only runs in the decode phase of flash decoding and returns after the linear
        # projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
        if (
            self.config.flash_decode
            and inference_context is not None
            and inference_context.is_decode_only()
            and not self.training
            and rotary_pos_cos is not None
        ):
            assert self.layer_number in inference_context.key_value_memory_dict
            assert inference_context.sequence_len_offset is not None
            inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[
                self.layer_number
            ]
            output = self.flash_decode(
                sequence_len_offset=sequence_len_offset,
                query_layer=query,
                key_layer=key,
                value_layer=value,
                inference_key_memory=inference_key_memory,
                inference_value_memory=inference_value_memory,
                rotary_cos=rotary_pos_cos,
                rotary_sin=rotary_pos_sin,
            )
            out = output.transpose(0, 1).contiguous()
            context_layer = out.view(out.size(0), out.size(1), -1)
            output, bias = self.linear_proj(context_layer)
            return output, bias

        query, key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_context,
            query,
            key,
            value,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if q_pos_emb is not None:
                # TODO VIJAY: simplify
                if inference_context is None or inference_context.is_static_batching():
                    query = apply_rotary_pos_emb(
                        query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q
                    )
                else:
                    query = inference_context.apply_rotary_emb_query(
                        query, q_pos_emb, self.config, cu_seqlens_q
                    )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv
                )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        # ==================================
        # core attention computation
        # ==================================

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                # Static batching attention kernel.
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    attn_mask_type=attn_mask_type,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )

            else:
                # Dynamic batching attention kernel.
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

                core_attn_out = self.flash_decode_and_prefill(
                    q, k, v, max_seqlen_q, max_seqlen_k, cu_query_lengths, cu_kv_lengths
                )
                core_attn_out = core_attn_out.squeeze(0).unsqueeze(1)
                core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.linear_proj(core_attn_out)

        return output, bias


