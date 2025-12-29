from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)

from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory

@dataclass
class ImageMLPSubmodules:
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None
    imagemlp_layernorm: Union[ModuleSpec, type] = None

class ImageMLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """                                                                                                                             
    def __init__(
        self,
        config: TransformerConfig,
        submodules: ImageMLPSubmodules,
        input_size: int = None,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.input_size = input_size if input_size != None else self.config.hidden_siz
        '''
        if self.training:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            assert tp_size == 1, 'yuanvl can not support tensor parallel'
        '''

        ffn_hidden_size = self.config.ffn_hidden_size

        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc1',
        )

        self.activation_func = self.config.activation_func

        self.imagemlp_layernorm = build_module(
            submodules.imagemlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

    def forward(self, hidden_states, per_token_scale=None):
        """Perform the forward pass through the MLP block."""
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        if self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                else:
                    raise ValueError("Only support fusion of swiglu with per_token_scale in MLP.")
            else:

                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(intermediate_parallel, bias_parallel)
                    else:
                        assert self.config.add_bias_linear is True
                        intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
                if self.config.gated_linear_unit:

                    def glu(x):
                        x = torch.chunk(x, 2, dim=-1)
                        return self.config.activation_func(x[0]) * x[1]

                    intermediate_parallel = glu(intermediate_parallel)
                else:
                    intermediate_parallel = self.activation_func(intermediate_parallel)

            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)

        # [s, b, h]
        output, output_bias = self.linear_fc2(intermediate_parallel)

        output = self.imagemlp_layernorm(output)

        if output_bias is not None:
            output = output + output_bias

        return output

    # pylint: disable=missing-function-docstring
    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = {}
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(f'{prefix}{name}.', sharded_offsets, metadata)
            if self.config.gated_linear_unit and name == 'linear_fc1':
                for k, v in sub_sd.items():
                    if k in (f'{prefix}{name}.weight', f'{prefix}{name}.bias'):
                        sub_sd[k] = apply_swiglu_sharded_factory(v, sharded_offsets)
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict
