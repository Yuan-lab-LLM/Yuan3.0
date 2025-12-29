# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

# use mcore transformer config to initialize the model
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from torch import nn

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec, get_gpt_mtp_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)

# from verl.models.mcore.yuan_vl.imagemlp import ImageMLPSubmodules, ImageMLP
from verl.models.mcore.yuan.layer import YuanTransformerLayerSubmodules

from verl.models.mcore.yuan.attention import (
    LocalizedFilteringSubmodules,
    YuanSelfAttentionSubmodules,
    LocalizedFiltering,
    YuanSelfAttention
)
from verl.models.mcore.yuan.moe import YuanMoESubmodules, YuanMLPSubmodules, YuanMoELayer
from .config_converter import PretrainedConfig, TransformerConfig
try:
    from megatron.core.transformer.custom_layers.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings
    from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm
    warnings.warn(f'Apex is not installed. Falling back to Torch LayerNorm')
    LNImpl = WrappedTorchLayerNorm


class BaseModelInitializer(ABC):
    """Base class for model initializers."""

    def __init__(self, tfconfig: TransformerConfig, hf_config: PretrainedConfig):
        self.tfconfig = tfconfig
        self.hf_config = hf_config

    @abstractmethod
    def get_transformer_layer_spec(self):
        """Get the transformer layer specification.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_layer_specs.py"""
        pass

    def get_rope_scaling_args(self) -> dict:
        """Get rope scaling args."""
        rope_scaling_args = {}
        if "rope_scaling" in self.hf_config:
            if self.hf_config.rope_scaling is not None:
                # assert self.hf_config.rope_scaling["type"] == "linear", "only linear scaling is supported for now"
                rope_scaling_args["seq_len_interpolation_factor"] = self.hf_config.rope_scaling["factor"]
        return rope_scaling_args

    def initialize(
        self,
        pre_process: bool = True,
        post_process: bool = True,
        share_embeddings_and_output_weights: bool = False,
        value: bool = False,
        **extra_kwargs,
    ) -> GPTModel:
        """Initialize a GPT model with the given configuration.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_model.py

        Args:
            pre_process (bool): include embedding layer.
            post_process (bool): including an output layer.
            share_embeddings_and_output_weights (bool): input embeddings and output logit weights are shared.
            value (bool): add an extra linear layer for classification or regression.

        Returns:
            GPTModel: An initialized GPT model instance
        """
        transformer_layer_spec = self.get_transformer_layer_spec()
        rope_scaling_args = self.get_rope_scaling_args()
        mtp_block_spec = extra_kwargs.get("mtp_block_spec", None)
        model = GPTModel(
            config=self.tfconfig,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type="rope",
            rotary_base=self.hf_config.rope_theta if hasattr(self.hf_config, 'rope_theta') else self.hf_config.rotary_base,
            **rope_scaling_args,
            mtp_block_spec=mtp_block_spec,
        )

        if post_process and value:
            from verl.models.llama.megatron.layers.parallel_linear import LinearForLastLayer

            model.output_layer = LinearForLastLayer(
                input_size=self.tfconfig.hidden_size, output_size=1, config=self.tfconfig
            )

        return model


class DenseModel(BaseModelInitializer):
    """Initializer for dense models like Llama and Qwen2."""

    def get_transformer_layer_spec(self):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        return get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True)


class Qwen2MoEModel(BaseModelInitializer):
    """Initializer for Qwen2 MoE models."""

    def get_transformer_layer_spec(self):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True)

        # Patch layer spec for shared experts
        for i in range(len(transformer_layer_spec.layer_specs)):
            transformer_layer_spec.layer_specs[i].submodules.mlp.submodules.shared_experts.params["gate"] = True

        return transformer_layer_spec

    def initialize(self, **kwargs):
        # Qwen default freeze_moe_router: true
        model = super().initialize(**kwargs)
        freeze_moe_router = kwargs.get("freeze_moe_router", True)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                layer.mlp.router.weight.requires_grad = False
        return model

class YuanMoEModel(BaseModelInitializer):
    """Initializer for Yuan MoE models."""

    def _get_lfa_module_spec(self) -> ModuleSpec:
        return ModuleSpec(
                   module=LocalizedFiltering,
                   submodules=LocalizedFilteringSubmodules(
                       conv1=torch.nn.Conv2d,
                       conv2=torch.nn.Conv2d,
                       output_layernorm=TENorm
                   )
    
               )

    def _get_moe_module_spec(
        self,
        use_te: bool = True,
        num_experts: int = None,
    ) -> ModuleSpec:

        expert_submodule = YuanMLPSubmodules(
            linear_fc1=TEColumnParallelGroupedLinear,
            linear_fc2=TERowParallelGroupedLinear
        )
        experts = ModuleSpec(
                      module=TEGroupedMLP,
                      submodules=expert_submodule
                  )
        # MoE module spec
        moe_module_spec = ModuleSpec(
            module=YuanMoELayer,
            submodules=YuanMoESubmodules(
                    experts=experts
                )
        )
        return moe_module_spec

    def get_transformer_layer_spec(
        self,
        num_experts: int = None,
    ) -> ModuleSpec:

        lf_gate = self._get_lfa_module_spec()

        mlp = self._get_moe_module_spec(
            use_te=True,
            num_experts=num_experts,
        )

        return ModuleSpec(
            module=TransformerLayer,
            submodules=YuanTransformerLayerSubmodules(
                input_layernorm=TENorm,
                self_attention=ModuleSpec(
                    module=YuanSelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=YuanSelfAttentionSubmodules(
                        linear_qkv=IdentityOp,
                        linear_qk=TEColumnParallelLinear,
                        linear_v=TEColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                        q_layernorm=IdentityOp,
                        k_layernorm=IdentityOp,
                        lf_gate=lf_gate,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=TENorm,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        ) 

    def initialize(
        self,
        pre_process: bool = True,
        post_process: bool = True,
        share_embeddings_and_output_weights: bool = False,
        value: bool = False,
        **extra_kwargs,
    ) -> GPTModel:
        """Initialize a GPT model with the given configuration.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_model.py

        Args:
            pre_process (bool): include embedding layer.
            post_process (bool): including an output layer.
            share_embeddings_and_output_weights (bool): input embeddings and output logit weights are shared.
            value (bool): add an extra linear layer for classification or regression.

        Returns:
            GPTModel: An initialized GPT model instance
        """
        transformer_layer_spec = self.get_transformer_layer_spec()
        rope_scaling_args = self.get_rope_scaling_args()
        mtp_block_spec = extra_kwargs.get("mtp_block_spec", None)
        model = GPTModel(
            config=self.tfconfig,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=not self.hf_config.tie_word_embeddings,
            position_embedding_type="rope",
            rotary_base=self.hf_config.rope_theta if hasattr(self.hf_config, 'rope_theta') else self.hf_config.rotary_base,
            rotary_percent=self.hf_config.rotary_percent,
            parallel_output=True,
            **rope_scaling_args,
            mtp_block_spec=mtp_block_spec,
        )

        if post_process and value:
            from verl.models.llama.megatron.layers.parallel_linear import LinearForLastLayer

            model.output_layer = LinearForLastLayer(
                input_size=self.tfconfig.hidden_size, output_size=1, config=self.tfconfig
            )

        return model

class MixtralModel(BaseModelInitializer):
    """Initializer for Mixtral models."""

    def get_transformer_layer_spec(self):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True)
        return transformer_layer_spec

    def initialize(self, **kwargs):
        model = super().initialize(**kwargs)
        freeze_moe_router = kwargs.get("freeze_moe_router", False)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                layer.mlp.router.weight.requires_grad = False
        return model


class Qwen3MoEModel(BaseModelInitializer):
    """Initializer for Qwen3 MoE models."""

    def get_transformer_layer_spec(self):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True)
        return transformer_layer_spec

    def initialize(self, **kwargs):
        # Qwen default freeze_moe_router: true
        model = super().initialize(**kwargs)
        freeze_moe_router = kwargs.get("freeze_moe_router", True)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                layer.mlp.router.weight.requires_grad = False
        return model


class DeepseekV3Model(BaseModelInitializer):
    """Initializer for DeepseekV3 models."""

    def get_transformer_layer_spec(self):
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True)
        return transformer_layer_spec

    def get_rope_scaling_args(self) -> dict:
        """Get rope scaling args."""
        rope_scaling_args = {}
        return rope_scaling_args

    def initialize(
        self,
        **kwargs,
    ):
        freeze_moe_router = kwargs.get("freeze_moe_router", True)
        if freeze_moe_router:
            self.tfconfig.moe_router_load_balancing_type = "none"
        # MTP
        if self.tfconfig.mtp_num_layers is not None:
            transformer_layer_spec = self.get_transformer_layer_spec()
            mtp_block_spec = get_gpt_mtp_block_spec(self.tfconfig, transformer_layer_spec, use_transformer_engine=True)
            kwargs["mtp_block_spec"] = mtp_block_spec

        model = super().initialize(**kwargs)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                if hasattr(layer.mlp, "router"):
                    layer.mlp.router.weight.requires_grad = False
        return model


class Qwen25VLModel(BaseModelInitializer):
    """Initializer for Qwen2.5 VL models."""

    def get_transformer_layer_spec(self):
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True)
        return transformer_layer_spec

    def initialize(
        self,
        pre_process=None,
        post_process=None,
        share_embeddings_and_output_weights=False,
        value=False,
        **extra_kwargs,
    ):
        tfconfig = self.tfconfig
        hf_config = self.hf_config
        # Qwen2_5_VLForConditionalGeneration
        from copy import deepcopy

        transformer_layer_spec = self.get_transformer_layer_spec()

        from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
        from megatron.core.models.gpt.moe_module_specs import MLPSubmodules
        from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec

        from .qwen2_5_vl import Qwen2_5VLModel, get_vision_model_config, get_vision_projection_config

        vision_transformer_config = get_vision_model_config(deepcopy(tfconfig))
        vision_transformer_config.pipeline_model_parallel_size = 1
        vision_transformer_config.first_pipeline_num_layers = None

        vision_projection_config = get_vision_projection_config(
            deepcopy(tfconfig),
            vision_transformer_config.hidden_size,
            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        )
        vision_projection_layer_spec = MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        )
        vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()

        qwen25_vl_model = Qwen2_5VLModel(
            language_transformer_config=tfconfig,
            language_transformer_layer_spec=transformer_layer_spec,
            language_vocab_size=hf_config.vocab_size,
            language_max_sequence_length=hf_config.max_position_embeddings,
            vision_transformer_config=vision_transformer_config,
            vision_transformer_layer_spec=vision_transformer_layer_spec,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_layer_spec,
            vision_projection_type="mlp",
            language_rotary_base=hf_config.rope_theta,
            pre_process=pre_process,
            post_process=post_process,
            add_decoder=True,
            add_encoder=True,
            parallel_output=True,
            language_share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        )

        if post_process and value:
            from verl.models.llama.megatron.layers.parallel_linear import LinearForLastLayer

            qwen25_vl_model.language_model.output_layer = LinearForLastLayer(
                input_size=tfconfig.hidden_size, output_size=1, config=tfconfig
            )

        return qwen25_vl_model

class YuanVLChatModel(BaseModelInitializer):
    """Initializer for YuanVLChatModel models."""
    def get_transformer_layer_spec(self):
        '''
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for YuanVLModel"
        assert self.tfconfig.use_lf_gate, f"YuanVLModel only supported use_lfa, but got {self.tfconfig.use_lf_gate}"
        assert self.tfconfig.use_yuanvl, f"YuanVLModel only supported use_yuanvl, but got {self.tfconfig.use_yuanvl}"
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True)
        '''
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
        tfconfig = self.tfconfig
        hf_config = self.hf_config
        use_apex_rmsnorm = hf_config.use_apex_rmsnorm if hasattr(hf_config, 'use_apex_rmsnorm') else False
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        num_experts=tfconfig.num_moe_experts,
                        moe_grouped_gemm=tfconfig.moe_grouped_gemm,
                        qk_layernorm=False,
                        use_lf_gate=hf_config.llm_config.use_lf_gate,
                        use_apex_rmsnorm=use_apex_rmsnorm,
                        multi_latent_attention=False,
                        moe_use_legacy_grouped_gemm=False)
        return transformer_layer_spec

    '''
    def _get_imagemlp_layer_with_transformer_engine_spec(
        use_apex_rmsnorm: bool = False,
    ) -> ModuleSpec:

        return ModuleSpec(
            module=ImageMLP,
            submodules=ImageMLPSubmodules(
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
                imagemlp_layernorm=TENorm,
            ),
        )
    '''

    def initialize(
        self,
        pre_process=None,
        post_process=None,
        share_embeddings_and_output_weights=False,
        value=False,
        **extra_kwargs,
    ):
        tfconfig = self.tfconfig
        hf_config = self.hf_config
        # YuanVLChatModel

        # transformer_layer_spec = self.get_transformer_layer_spec()

        '''
        from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
        from megatron.core.models.gpt.moe_module_specs import MLPSubmodules
        from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
        from .yuan_vl import YuanVLModel, get_vision_model_config, get_vision_projection_config

        vision_transformer_config = get_vision_model_config(deepcopy(tfconfig))
        vision_transformer_config.pipeline_model_parallel_size = 1
        vision_transformer_config.first_pipeline_num_layers = None

        vision_projection_config = get_vision_projection_config(
            deepcopy(tfconfig),
            vision_transformer_config.hidden_size,
            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        )
        vision_projection_layer_spec = MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        )
        vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()
        '''
        use_apex_rmsnorm = hf_config.use_apex_rmsnorm if hasattr(hf_config, 'use_apex_rmsnorm') else False
        yuanvl_use_te_imagemlp = hf_config.yuanvl_use_te_imagemlp if hasattr(hf_config, 'yuanvl_use_te_imagemlp') else False
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_decoder_block_spec,
            get_gpt_layer_local_spec,
            get_gpt_mtp_block_spec,
            get_imagemlp_layer_with_transformer_engine_spec,
        )
        transformer_layer_spec = self.get_transformer_layer_spec()
        imagemlp_layer_spec = get_imagemlp_layer_with_transformer_engine_spec(
                                  use_apex_rmsnorm=use_apex_rmsnorm,
                                  yuanvl_use_te_imagemlp=yuanvl_use_te_imagemlp)
        mtp_block_spec = None
        '''
        yuan_vl_model = YuanVLModel(
            language_transformer_config=tfconfig,
            language_transformer_layer_spec=transformer_layer_spec,
            language_vocab_size=hf_config.vocab_size,
            language_max_sequence_length=hf_config.max_position_embeddings,
            vision_transformer_config=vision_transformer_config,
            vision_transformer_layer_spec=vision_transformer_layer_spec,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_layer_spec,
            vision_projection_type="mlp",
            language_rotary_base=hf_config.rope_theta,
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=False,
            language_share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        )
        '''
        model = GPTModel(
            config=tfconfig,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=tfconfig.padded_vocab_size,
            max_sequence_length=tfconfig.max_position_embeddings,
            imagemlp_layer_spec=imagemlp_layer_spec,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=False,
            parallel_output=True,
            share_embeddings_and_output_weights=hf_config.llm_config.tie_word_embeddings,
            position_embedding_type='rope',
            rotary_percent=tfconfig.rotary_percent,
            rotary_base=tfconfig.rotary_base,
            rope_scaling=tfconfig.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
        )

        if post_process and value:
            from verl.models.llama.megatron.layers.parallel_linear import LinearForLastLayer

            model.output_layer = LinearForLastLayer(
                input_size=tfconfig.hidden_size, output_size=1, config=tfconfig
            )
        return model
