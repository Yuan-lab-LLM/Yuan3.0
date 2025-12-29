from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union
import torch
from torch import Tensor

from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
)
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core import parallel_state
from megatron.core.transformer.moe.router import TopKRouter, Router
from megatron.core.transformer.moe.moe_layer import BaseMoELayer
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.module import MegatronModule
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.utils import attention_mask_func

from verl.utils.megatron_utils import TransformerConfig

@dataclass
class YuanMLPSubmodules:
    """MoE Layer Submodule spec"""
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


@dataclass
class YuanMoESubmodules:
    """MoE Layer Submodule spec"""
    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class DotProductAttention(MegatronModule):
    """
    Region where selective activation recomputation is applied.
    This region is memory intensive but less compute intensive which
    makes activation checkpointing more efficient for LLMs (20B+).
    See Reducing Activation Recomputation in Large Transformer Models: https://arxiv.org/abs/2205.05198 for more details.

    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config

        assert (
            self.config.context_parallel_size == 1
        ), "Context parallelism is only supported by TEDotProductAttention!"

        assert (
            self.config.window_size is None
        ), "Sliding Window Attention is only supported by TEDotProductAttention!"

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused for now

        # Per attention head and per partition values.
        self.hidden_size_per_partition = 0
        coeff = None
        self.norm_factor = 1.0
        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.config.fp16,
            input_in_bf16=self.config.bf16,
            attn_mask_type=self.attn_mask_type,
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
            mask_func=attention_mask_func,
            softmax_in_fp32=self.config.attention_softmax_in_fp32,
            scale=coeff,
        )
        self.attention_dropout = torch.nn.Dropout(attention_dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        assert packed_seq_params is None, (
            "Packed sequence is not supported by DotProductAttention."
            "Please use TEDotProductAttention instead."
        )

        self.hidden_size_per_partition = query.size(2)
        output_size = (
            query.size(1),
            query.size(2),
            query.size(0),
            key.size(0),
        )
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu",
        )
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query.transpose(0, 1),  # [b * np, sq, hn]
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )
        attention_scores = matmul_result.view(*output_size)
        attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)
        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)
        output_size = (
            value.size(1),
            value.size(2),
            query.size(0),
            value.size(3),
        )
        value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        context = torch.bmm(attention_probs, value.transpose(0, 1))
        context = context.view(*output_size)
        context = context.permute(2, 0, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)


class ParallelAttention_router(MegatronModule):
    def __init__(
        self,
        config,
        layer_number: int = 0,
        num_moe_experts: int = None

    ):
        super(ParallelAttention_router, self).__init__(config)
        self.layer_number = max(1, layer_number)
        self.hidden_size = config.hidden_size
        assert num_moe_experts != None, f'num_moe_experts must be give in ParallelAttention_router'
        projection_size = num_moe_experts
        self.weight = torch.nn.Parameter(
            torch.empty((3 * projection_size, config.hidden_size))
        )
        if config.perform_initialization:
            if get_cuda_rng_tracker().is_initialized():
                with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
                    config.init_method(self.weight)
        else:
            config.init_method(self.weight)
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)

        self.core_attention = DotProductAttention(config=config,
                                                  layer_number=self.layer_number,
                                                  attn_mask_type=AttnMaskType.padding,
                                                  attention_type=None,
                                                  attention_dropout=0)
        self.checkpoint_core_attention = config.recompute_granularity == 'selective'

    def forward(self, hidden_states, attention_mask=None, enc_position_ids=None,
                encoder_output=None, inference_params=None,
                rotary_pos_emb=None):
        mixed_x_layer = torch.nn.functional.linear(hidden_states, self.weight)

        (query_layer,
         key_layer,
         value_layer) = torch.split(mixed_x_layer, mixed_x_layer.size(-1) // 3, dim=-1)

        seq_length = query_layer.size(0)
        batch_size = query_layer.size(1)
        expert_num = query_layer.size(2)

        query_layer = query_layer.transpose(0, 2).contiguous().view(expert_num, batch_size, seq_length, 1)
        key_layer = key_layer.transpose(0, 2).contiguous().view(expert_num, batch_size, seq_length, 1)
        value_layer = value_layer.transpose(0, 2).contiguous().view(expert_num, batch_size, seq_length, 1)

        context_layer = self.core_attention(
                query_layer.float(), key_layer.float(), value_layer.float(), None)
        router_output = context_layer.transpose(0, 2).contiguous()
        return router_output

class YuanTopKRouter(TopKRouter):
    """Base Router class"""

    def __init__(self, config: TransformerConfig, num_total_experts = None) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super().__init__(config, num_total_experts)
        self.config = config
        self.num_experts = num_total_experts if num_total_experts != None else config.num_moe_experts
        self.moe_aux_loss_func = None
        self.layer_number = None
        

        assert num_total_experts != None, f'num_total_experts must be give in YuanTopKRouter'
        # Initialize the gate weights.
        # TODO: Add support for GPU initialization, which requires updating the golden values.
        if self.config.use_attention_router:
            self.attention_router = ParallelAttention_router(
                                        config,
                                        num_moe_experts=num_total_experts
                                    )
        else:
            self.weight = torch.nn.Parameter(
                torch.empty((self.num_experts, self.config.hidden_size), dtype=torch.float32)
            )
            if config.perform_initialization:
                config.init_method(self.weight)
            self.weight.data = self.weight.data.to(dtype=config.params_dtype)
            setattr(self.weight, 'sequence_parallel', config.sequence_parallel)
            # If calculate per token loss, we need to scale up moe aux loss by the number of tokens.
            # So we need to know if the model is configured to calculate per token loss.
            self.calculate_per_token_loss = self.config.calculate_per_token_loss

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == 'cpu':
            # move weights to GPU
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        # Convert to specified datatype for routing computation if enabled
        router_dtype = input.dtype
        if self.config.moe_router_dtype == 'fp32':
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == 'fp64':
            router_dtype = torch.float64

        if self.use_attention_router:
            logits = self.attention_router(input)
        else:
            logits = torch.nn.functional.linear(input, self.weight)
        return logits

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        if not self.use_attention_router:
            self._maintain_float32_expert_bias()

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)
        if self.use_attention_router:
            logits = logits.view(-1, self.num_experts)

        scores, routing_map = self.routing(logits)

        if self.use_attention_router:
            scores = scores.to(input.dtype)

        return scores, routing_map

class YuanMoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """
    def __init__(
        self,
        config: TransformerConfig,
        submodules: YuanMoESubmodules = None,
        layer_number: int = None,
    ):
        self.submodules = submodules
        super(YuanMoELayer, self).__init__(config=config, layer_number=layer_number)

        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        if self.config.per_layer_experts_blocks != None:
            assert self.config.per_layer_experts_blocks[layer_number-1] % self.expert_parallel_size == 0, f'sssssssssssssss {self.config.per_layer_experts_blocks} {layer_number - 1}'
            self.num_moe_experts = self.config.per_layer_experts_blocks[layer_number-1]

        elif self.config.num_moe_experts != None:
            assert self.config.num_moe_experts % self.expert_parallel_size == 0
            self.num_moe_experts = self.config.num_moe_experts
        else:
            raise ValueError(f"must given config.moe_config['per_layer_experts_blocks'] or config.num_moe_experts")

        self.num_local_experts = self.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.num_moe_experts, self.local_expert_indices))
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None

        self.layer_number = layer_number

        # self.num_total_experts = self.num_moe_experts

        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective' and "moe" in config.recompute_modules
        )

        self.router = YuanTopKRouter(config=self.config, num_total_experts=self.num_moe_experts)

        self.experts = build_module(self.submodules.experts, self.num_local_experts, self.config)

        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.num_moe_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}")
        '''
        self.mean_num_local_tokens_per_expert = torch.zeros(self.config.num_moe_experts, dtype=torch.int32, device='cuda')
        self.acc_size = args.global_batch_size // (args.data_parallel_size * args.micro_batch_size) * 2
        self.acc_num = 0
        '''

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(hidden_states):
            probs, routing_map = self.router(hidden_states)
            (dispatched_input, tokens_per_expert) = (
                self.token_dispatcher.token_permutation(hidden_states, probs, routing_map)
            )
            expert_output, mlp_bias = self.experts(
                dispatched_input, tokens_per_expert
            )
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias

    def set_layer_number(self, layer_number: int):
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)
