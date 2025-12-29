import warnings
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.cuda_graphs import CudaGraphManager
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import deprecate_inference_params, is_te_min_version, make_viewless_tensor

from verl.utils.megatron_utils import TransformerConfig
@dataclass
class YuanTransformerLayerSubmodules:
    """
    Configuration class for specifying the submodules of a transformer layer.

    This class defines the structure and default implementations for various
    components of a transformer layer, allowing for flexible customization
    of the layer's architecture.

    Args:
        input_layernorm (Union[ModuleSpec, type]): Specification for the input layer normalization.
        self_attention (Union[ModuleSpec, type]): Specification for the self-attention mechanism.
        self_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after self-attention.
        pre_cross_attn_layernorm (Union[ModuleSpec, type]): Specification for the layer
            normalization before cross-attention.
        cross_attention (Union[ModuleSpec, type]): Specification for the cross-attention mechanism.
        cross_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after cross-attention.
        pre_mlp_layernorm (Union[ModuleSpec, type]): Specification for the layer normalization
            before the MLP.
        mlp (Union[ModuleSpec, type]): Specification for the MLP in Dense layer.
        mlp_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after the MLP.
        sharded_state_dict_keys_map (Dict[str, str]): Mapping for sharded tensor keys to be applied
            in the `sharded_state_dict` method.
    """

    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)
