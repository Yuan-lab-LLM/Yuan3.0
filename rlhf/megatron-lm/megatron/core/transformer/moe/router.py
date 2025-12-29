# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import torch

from megatron.core import parallel_state
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
    sequence_load_balancing_loss_func,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
    z_loss_func,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType

from megatron.core import parallel_state, tensor_parallel
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.utils import divide
from torch import Tensor

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

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
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
        # [b, np, sq, sk]
        output_size = (
            query.size(1),
            query.size(2),
            query.size(0),
            key.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use simple strides
        # to extract the queries.
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu",
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query.transpose(0, 1),  # [b * np, sq, hn]
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value.size(1),
            value.size(2),
            query.size(0),
            value.size(3),
        )

        # change view [sk, b * np, hn]
        value = value.view(value.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))

        # change view [b, np, sq, hn]
        context = context.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)

        return context

class ParallelAttention_router(MegatronModule):
    def __init__(self, config, num_experts, layer_number=0,):
        super(ParallelAttention_router, self).__init__(config)
        self.layer_number = max(1, layer_number)
        self.hidden_size = config.hidden_size
        projection_size = num_experts
        
        # Initialize the gate weights.
        self.weight = torch.nn.Parameter(
                torch.empty((3 * projection_size, self.config.hidden_size), dtype=torch.float32)
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
        
        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
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

class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(self, config: TransformerConfig, num_total_experts) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super().__init__(config)
        self.config = config
        self.use_attention_router = self.config.use_attention_router
        self.num_experts = num_total_experts
        self.moe_aux_loss_func = None
        self.layer_number = None

        # Initialize the gate weights.
        # TODO: Add support for GPU initialization, which requires updating the golden values.
        if self.use_attention_router:
            self.attention_router = ParallelAttention_router(config, self.num_experts)
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
        if self.use_attention_router:
            logits = self.attention_router(input)
        else:
            if self.weight.device.type == 'cpu':
                # move weights to GPU
                self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        # Convert to specified datatype for routing computation if enabled
            router_dtype = input.dtype
            if self.config.moe_router_dtype == 'fp32':
                router_dtype = torch.float32
            elif self.config.moe_router_dtype == 'fp64':
                router_dtype = torch.float64
            logits = torch.nn.functional.linear(input.to(router_dtype), self.weight.to(router_dtype))
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mapping.
        """
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        raise NotImplementedError("Forward function not implemented.")

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the router."""
        self.layer_number = layer_number


class TopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(
        self,
        config: TransformerConfig,
        num_total_experts
    ) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config, num_total_experts=num_total_experts)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.score_function = self.config.moe_router_score_function
        self.input_jitter = None

        self.enable_expert_bias = self.config.moe_router_enable_expert_bias
        if self.enable_expert_bias:
            self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(self.num_experts, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                'expert_bias', torch.zeros(self.num_experts, dtype=torch.float32)
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None

    def _maintain_float32_expert_bias(self):
        """
        Maintain the expert bias in float32.

        When using bf16/fp16, the expert bias gets converted to lower precision in Float16Module.
        We keep it in float32 to avoid routing errors when updating the expert_bias.
        """
        if hasattr(self, 'expert_bias') and self.expert_bias is not None:
            if self.expert_bias.dtype != torch.float32:
                self.expert_bias.data = self.expert_bias.data.to(torch.float32)

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mask.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = _sinkhorn_activation(logits)
        else:
            logits = _sinkhorn_activation(logits)
            _, indices = torch.topk(logits, k=self.topk, dim=1)
        map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
        scores = logits * map
        return scores, map

    def compute_routing_scores_for_aux_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute routing scores based on the score function.

        Args:
            logits (torch.Tensor): The logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            torch.Tensor: The normalized routing scores.
        """
        if self.score_function == "softmax":
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        elif self.score_function == "sigmoid":
            scores = torch.sigmoid(logits)
            scores = (
                scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.topk > 1 else scores
            )
        else:
            raise ValueError(f"Invalid score_function: {self.score_function}")
        return scores

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply auxiliary loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mask of token to experts assignment.
        """
        probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
        )

        if self.training and torch.is_grad_enabled():
            # Apply auxiliary load balancing loss
            # Skip auxiliary loss calculations when using torch.no_grad() or checkpointing.
            scores = self.compute_routing_scores_for_aux_loss(logits)
            aux_loss_func = partial(
                switch_load_balancing_loss_func,
                probs=scores,
                tokens_per_expert=tokens_per_expert,
                topk=self.topk,
            )
            probs = self.apply_load_balancing_loss(
                activation=probs, load_balancing_loss_func=aux_loss_func
            )
        return probs, routing_map

    def seq_aux_loss_load_balancing(self, logits: torch.Tensor, bsz: int, seq_length: int):
        """Apply sequence-auxiliary loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor after gating, shape: [num_tokens, num_experts].
            bsz (int): The batch size.
            seq_length (int): The sequence length.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mask of token to experts assignment.
        """

        probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
        )

        if self.training and torch.is_grad_enabled():
            # Apply sequence-auxiliary load balancing loss
            scores = self.compute_routing_scores_for_aux_loss(logits)
            aux_loss_func = partial(
                sequence_load_balancing_loss_func,
                probs=scores,
                routing_map=routing_map,
                batch_size=bsz,
                seq_length=seq_length,
                topk=self.topk,
            )
            probs = self.apply_load_balancing_loss(
                activation=probs, load_balancing_loss_func=aux_loss_func
            )

        return probs, routing_map

    def apply_load_balancing_loss(
        self, activation: torch.Tensor, load_balancing_loss_func: Callable
    ):
        """Calculate auxiliary loss, attach gradient function to activation and add to logging."""
        moe_aux_loss_coeff = self.config.moe_aux_loss_coeff
        if moe_aux_loss_coeff == 0:
            return activation
        sequence_partition_group = None
        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            sequence_partition_group = parallel_state.get_context_parallel_group()
            moe_aux_loss_coeff /= parallel_state.get_tensor_model_parallel_world_size()
        elif parallel_state.get_tensor_and_context_parallel_world_size() > 1:
            sequence_partition_group = parallel_state.get_tensor_and_context_parallel_group()

        aux_loss = load_balancing_loss_func(
            moe_aux_loss_coeff=moe_aux_loss_coeff, sequence_partition_group=sequence_partition_group
        )
        save_to_aux_losses_tracker(
            "load_balancing_loss",
            aux_loss / moe_aux_loss_coeff,
            self.layer_number,
            self.config.num_layers,
            reduce_group=sequence_partition_group,
        )
        if self.calculate_per_token_loss:
            # Scale the aux_loss by the number of tokens.
            # The expected final scaling for aux_loss gradients is 1/(num_micro_batches * dp_size).
            # After commit 02648000, Megatron started using the number of total tokens to scale
            # gradients under the argument of calculate_per_token_loss,
            # which scales both the main_loss gradient and aux_loss gradient by
            # 1/(num_local_tokens * dp_size * num_micro_batches) in finalize_model_grads function.
            # To correct this scaling, we need to scale the aux_loss by num_local_tokens here.
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss * activation.shape[0])
        else:
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training and torch.is_grad_enabled():
            # Skip Z loss calculations when using torch.no_grad() or checkpointing.
            moe_z_loss_coeff = (
                self.config.moe_z_loss_coeff
                / parallel_state.get_tensor_and_context_parallel_world_size()
            )
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            scale_up = 1.0
            if self.calculate_per_token_loss:
                # The expected final scaling for z_loss gradients is
                # 1/(num_micro_batches * dp_size).
                # After commit 02648000, Megatron started using the number of total tokens
                # to scale gradients under the argument of calculate_per_token_loss,
                # which scales both the main_loss gradient and z_loss gradient by
                # 1/(num_local_tokens * dp_size * num_micro_batches) in finalize_model_grads().
                # To correct this scaling, we need to scale the z_loss by num_local_tokens here.
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss * logits.shape[0])
            else:
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            save_to_aux_losses_tracker(
                "z_loss", z_loss / moe_z_loss_coeff, self.layer_number, self.config.num_layers
            )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.num_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        if self.routing_type == "sinkhorn":
            scores, routing_map = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "aux_loss":
            scores, routing_map = self.aux_loss_load_balancing(logits)
        elif self.routing_type == "seq_aux_loss":
            scores, routing_map = self.seq_aux_loss_load_balancing(logits, bsz, seq_length)
        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            scores, routing_map, _ = topk_softmax_with_capacity(
                logits,
                self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                drop_policy=self.config.moe_token_drop_policy,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.config.moe_router_num_groups,
                group_topk=self.config.moe_router_group_topk,
                scaling_factor=self.config.moe_router_topk_scaling_factor,
                deterministic_mode=self.config.deterministic_mode,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
            )
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")
        # Prevent extra local tokens accumulation on evaluation or activation recomputation
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)

        return scores, routing_map

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self._maintain_float32_expert_bias()

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)

        scores, routing_map = self.routing(logits)

        return scores, routing_map
