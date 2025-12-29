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

# online convert mcore weight to pure huggingface weight, no any fusion
# including format conversion and name mapping
# not including resharding
from optparse import Option
from typing import Union, List, Optional
import torch
from transformers import PretrainedConfig

# from megatron.core.transformer import TransformerConfig
from verl.utils.megatron_utils import TransformerConfig

class McoreToHFWeightConverterBase:
    def __init__(self, hf_config: PretrainedConfig, mcore_config: TransformerConfig):
        self.hf_config = hf_config
        self.mcore_config = mcore_config

    def convert_param(self, name: str, params_one_group: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class McoreToHFWeightConverterDense(McoreToHFWeightConverterBase):
    def _convert_attention_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # 'decoder.layers.0.self_attention.linear_proj.weight'
        # 'decoder.layers.0.self_attention.linear_qkv.layer_norm_weight'
        # 'decoder.layers.0.self_attention.linear_qkv.weight'
        # 'decoder.layers.0.self_attention.linear_qkv.bias'
        layer_number = name.split(".")[2]
        convert_names = []
        if "self_attention.linear_qkv.bias" in name or "self_attention.linear_qkv.weight" in name:
            param_type = name.split(".")[-1]
            assert param_type == "bias" or param_type == "weight"
            convert_names.append(f"model.layers.{layer_number}.self_attn.q_proj.{param_type}")
            convert_names.append(f"model.layers.{layer_number}.self_attn.k_proj.{param_type}")
            convert_names.append(f"model.layers.{layer_number}.self_attn.v_proj.{param_type}")
            assert len(params) == 3
        elif "self_attention.linear_proj.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.self_attn.o_proj.weight")
            assert len(params) == 1
        elif "self_attention.linear_qkv.layer_norm_weight" in name:
            convert_names.append(f"model.layers.{layer_number}.input_layernorm.weight")
            assert len(params) == 1
        elif "self_attention.q_layernorm.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.self_attn.q_norm.weight")
            assert len(params) == 1
        elif "self_attention.k_layernorm.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.self_attn.k_norm.weight")
            assert len(params) == 1
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names, params

    def _convert_mlp_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # 'decoder.layers.0.mlp.linear_fc1.layer_norm_weight'
        # 'decoder.layers.0.mlp.linear_fc1.weight'
        # 'decoder.layers.0.mlp.linear_fc2.weight'
        layer_number = name.split(".")[2]
        convert_names = []
        if "mlp.linear_fc1.weight" in name:
            # split gate_proj and up_proj
            convert_names.append(f"model.layers.{layer_number}.mlp.gate_proj.weight")
            convert_names.append(f"model.layers.{layer_number}.mlp.up_proj.weight")
            assert len(params) == 2
        elif "mlp.linear_fc1.layer_norm_weight" in name:
            convert_names.append(f"model.layers.{layer_number}.post_attention_layernorm.weight")
            assert len(params) == 1
        elif "mlp.linear_fc2.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.mlp.down_proj.weight")
            assert len(params) == 1
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names, params

    def convert_param(self, name: str, params_one_group: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        direct_name_mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }
        if name in direct_name_mapping:
            return [direct_name_mapping[name]], [params_one_group[0]]

        if "self_attention" in name:
            return self._convert_attention_param(name, params_one_group)
        elif "mlp" in name:
            return self._convert_mlp_param(name, params_one_group)
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")


class McoreToHFWeightConverterQwen2Moe(McoreToHFWeightConverterDense):
    def _convert_mlp_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # 'decoder.layers.0.pre_mlp_layernorm.weight',
        # 'decoder.layers.0.mlp.router.weight',
        # 'decoder.layers.0.mlp.shared_experts.gate_weight',
        # 'decoder.layers.0.mlp.shared_experts.linear_fc1.weight',
        # 'decoder.layers.0.mlp.shared_experts.linear_fc2.weight'
        # moe1
        # 'decoder.layers.0.mlp.experts.linear_fc1.weight0',
        # 'decoder.layers.0.mlp.experts.linear_fc1.weight1',
        # 'decoder.layers.0.mlp.experts.linear_fc1.weight2',
        # 'decoder.layers.0.mlp.experts.linear_fc1.weight3',
        # moe2
        # 'decoder.layers.0.mlp.experts.linear_fc2.weight0',
        # 'decoder.layers.0.mlp.experts.linear_fc2.weight1',
        layer_number = name.split(".")[2]
        convert_names = []
        if "pre_mlp_layernorm" in name:
            convert_names.append(f"model.layers.{layer_number}.post_attention_layernorm.weight")
            assert len(params) == 1
        elif "mlp.router.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.mlp.gate.weight")
            assert len(params) == 1
        elif "shared_experts.gate_weight" in name:
            convert_names.append(f"model.layers.{layer_number}.mlp.shared_expert_gate.weight")
            assert len(params) == 1
        elif "shared_experts.linear_fc1.weight" in name:  # split gate_proj and up_proj
            convert_names.append(f"model.layers.{layer_number}.mlp.shared_expert.gate_proj.weight")
            convert_names.append(f"model.layers.{layer_number}.mlp.shared_expert.up_proj.weight")
            assert len(params) == 2
        elif "shared_experts.linear_fc2.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.mlp.shared_expert.down_proj.weight")
            assert len(params) == 1
        elif "mlp.experts.linear_fc1" in name:  # split gate_proj and up_proj
            expert_id = name.split("weight")[-1]
            convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight")
            convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight")
            assert len(params) == 2
        elif "mlp.experts.linear_fc2" in name:
            expert_id = name.split("weight")[-1]
            convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight")
            assert len(params) == 1
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names, params


class McoreToHFWeightConverterQwen2_5_VL(McoreToHFWeightConverterDense):
    def convert_param(self, name: str, params_one_group: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        direct_name_mapping = {
            "language_model.embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "language_model.decoder.final_layernorm.weight": "model.norm.weight",
            "language_model.output_layer.weight": "lm_head.weight",
            "vision_model.patch_embed.proj.weight": "visual.patch_embed.proj.weight",
            "vision_model.decoder.final_layernorm.weight": "visual.merger.ln_q.weight",
            "vision_model.projection.encoder.linear_fc1.weight": "visual.merger.mlp.0.weight",
            "vision_model.projection.encoder.linear_fc1.bias": "visual.merger.mlp.0.bias",
            "vision_model.projection.encoder.linear_fc2.weight": "visual.merger.mlp.2.weight",
            "vision_model.projection.encoder.linear_fc2.bias": "visual.merger.mlp.2.bias",
        }
        if name in direct_name_mapping:
            return [direct_name_mapping[name]], [params_one_group[0]]

        if "self_attention" in name:
            return self._convert_attention_param(name, params_one_group)
        elif "mlp" in name:
            return self._convert_mlp_param(name, params_one_group)
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")

    def _convert_attention_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        model_type, _, _, layer_number = name.split(".")[:4]

        convert_names = []
        if model_type == "language_model":
            name_map_after_layer = {
                "self_attention.linear_qkv.bias": [
                    "self_attn.q_proj.bias",
                    "self_attn.k_proj.bias",
                    "self_attn.v_proj.bias",
                ],
                "self_attention.linear_qkv.weight": [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ],
                "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
                "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
            }
            name_after_layer = ".".join(name.split(".")[-3:])
            mapped_name = name_map_after_layer.get(name_after_layer)
            if isinstance(mapped_name, list):
                assert len(params) == len(mapped_name)
                for one in mapped_name:
                    convert_names.append(f"model.layers.{layer_number}.{one}")
            else:
                assert len(params) == 1
                convert_names.append(f"model.layers.{layer_number}.{mapped_name}")
        elif model_type == "vision_model":
            name_map_after_layer = {
                "self_attention.linear_proj.weight": "attn.proj.weight",
                "self_attention.linear_proj.bias": "attn.proj.bias",
                "self_attention.linear_qkv.layer_norm_weight": "norm1.weight",
            }
            name_after_layer = ".".join(name.split(".")[-3:])
            mapped_name = name_map_after_layer.get(name_after_layer, None)
            if mapped_name is None:
                assert "linear_qkv" in name_after_layer
                assert len(params) == 3
                new_param = torch.cat(params, dim=0)
                params = [new_param]
                if "bias" in name_after_layer:
                    convert_names.append(f"visual.blocks.{layer_number}.attn.qkv.bias")
                else:
                    convert_names.append(f"visual.blocks.{layer_number}.attn.qkv.weight")
            else:
                assert len(params) == 1
                convert_names.append(f"visual.blocks.{layer_number}.{mapped_name}")
        else:
            raise NotImplementedError(f"Unsupported model type: {model_type}")
        return convert_names, params

    def _convert_mlp_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        model_type, _, _, layer_number = name.split(".")[:4]

        convert_names = []
        if model_type == "language_model":
            name_map_after_layer = {
                "mlp.linear_fc1.weight": ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
                "mlp.linear_fc1.bias": ["mlp.gate_proj.bias", "mlp.up_proj.bias"],
                "mlp.linear_fc2.weight": "mlp.down_proj.weight",
                "mlp.linear_fc2.bias": "mlp.down_proj.bias",
                "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
            }
            name_after_layer = ".".join(name.split(".")[-3:])
            mapped_name = name_map_after_layer.get(name_after_layer)
            if isinstance(mapped_name, list):
                assert len(params) == len(mapped_name)
                for one in mapped_name:
                    convert_names.append(f"model.layers.{layer_number}.{one}")
            else:
                assert len(params) == 1
                convert_names.append(f"model.layers.{layer_number}.{mapped_name}")

        elif model_type == "vision_model":
            name_map_after_layer = {
                "mlp.linear_fc1.weight": ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
                "mlp.linear_fc1.bias": ["mlp.gate_proj.bias", "mlp.up_proj.bias"],
                "mlp.linear_fc2.weight": "mlp.down_proj.weight",
                "mlp.linear_fc2.bias": "mlp.down_proj.bias",
                "mlp.linear_fc1.layer_norm_weight": "norm2.weight",
            }
            name_after_layer = ".".join(name.split(".")[-3:])
            mapped_name = name_map_after_layer.get(name_after_layer)
            if isinstance(mapped_name, list):
                assert len(params) == len(mapped_name)
                for one in mapped_name:
                    convert_names.append(f"visual.blocks.{layer_number}.{one}")
            else:
                assert len(params) == 1
                convert_names.append(f"visual.blocks.{layer_number}.{mapped_name}")
        else:
            raise NotImplementedError(f"Unsupported model type: {model_type}")
        return convert_names, params


class McoreToHFWeightConverterDpskv3(McoreToHFWeightConverterBase):
    def _convert_attention_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # mcore
        # 'decoder.layers.0.input_layernorm.weight'
        # 'decoder.layers.0.self_attention.linear_proj.weight'
        # 'decoder.layers.0.self_attention.linear_q_proj.weight'
        # 'decoder.layers.0.self_attention.linear_kv_down_proj.weight'
        # 'decoder.layers.0.self_attention.linear_kv_up_proj.layer_norm_weight'
        # 'decoder.layers.0.self_attention.linear_kv_up_proj.weight'
        # 'decoder.layers.0.self_attention.linear_q_down_proj.weight'
        # 'decoder.layers.0.self_attention.linear_q_up_proj.weight'
        # 'decoder.layers.0.self_attention.linear_q_up_proj.layer_norm_weight'
        # hf
        # 'model.layers.0.input_layernorm.weight'
        # 'model.layers.0.self_attn.o_proj.weight'
        # 'model.layers.0.self_attn.q_proj.weight'
        # 'model.layers.0.self_attn.kv_a_proj_with_mqa.weight'
        # 'model.layers.0.self_attn.kv_a_layernorm.weight'
        # 'model.layers.0.self_attn.kv_b_proj.weight'
        # 'model.layers.0.self_attn.q_a_proj.weight'
        # 'model.layers.0.self_attn.q_b_proj.weight'
        # 'model.layers.0.self_attn.q_a_layernorm.weight'
        name_map_after_layer = {
            "input_layernorm.weight": "input_layernorm.weight",
            "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
            "self_attention.linear_q_proj.weight": "self_attn.q_proj.weight",
            "self_attention.linear_kv_down_proj.weight": "self_attn.kv_a_proj_with_mqa.weight",
            "self_attention.linear_kv_up_proj.layer_norm_weight": "self_attn.kv_a_layernorm.weight",
            "self_attention.linear_kv_up_proj.weight": "self_attn.kv_b_proj.weight",
            "self_attention.linear_q_down_proj.weight": "self_attn.q_a_proj.weight",
            "self_attention.linear_q_up_proj.weight": "self_attn.q_b_proj.weight",
            "self_attention.linear_q_up_proj.layer_norm_weight": "self_attn.q_a_layernorm.weight",
        }
        assert len(params) == 1
        convert_names = []
        layer_number = name.split(".")[2]
        name_after_layer = name.split(f".{layer_number}.")[1]
        convert_names.append(f"model.layers.{layer_number}.{name_map_after_layer[name_after_layer]}")
        return convert_names, params

    def _convert_mlp_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # mcore dense
        # 'decoder.layers.0.mlp.linear_fc1.layer_norm_weight'
        # 'decoder.layers.0.mlp.linear_fc2.weight'
        # 'decoder.layers.0.mlp.linear_fc1.weight'
        #       ---
        # 'decoder.layers.1.mlp.shared_experts.linear_fc1.weight'
        #       ---
        # 'decoder.layers.1.mlp.shared_experts.linear_fc2.weight'
        # hf dense
        # 'model.layers.0.post_attention_layernorm.weight'
        # 'model.layers.0.mlp.down_proj.weight'
        # 'model.layers.0.mlp.gate_proj.weight'
        # 'model.layers.0.mlp.up_proj.weight'
        # 'model.layers.1.mlp.shared_experts.gate_proj.weight'
        # 'model.layers.1.mlp.shared_experts.up_proj.weight'
        # 'model.layers.1.mlp.shared_experts.down_proj.weight'

        # mcore moe
        # 'decoder.layers.1.pre_mlp_layernorm.weight'
        # 'decoder.layers.1.mlp.router.weight'
        # 'decoder.layers.1.mlp.router.expert_bias'
        # 'decoder.layers.1.mlp.experts.linear_fc1.weight0'
        #       ---
        # 'decoder.layers.1.mlp.experts.linear_fc2.weight0'
        # hf moe
        # 'model.layers.1.post_attention_layernorm.weight'
        # 'model.layers.1.mlp.gate.weight'
        # 'model.layers.1.mlp.gate.e_score_correction_bias'
        # 'model.layers.1.mlp.experts.0.gate_proj.weight'
        # 'model.layers.1.mlp.experts.0.up_proj.weight'
        # 'model.layers.1.mlp.experts.0.down_proj.weight'

        name_map_after_layer = {
            "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
            "mlp.linear_fc2.weight": "mlp.down_proj.weight",
            "mlp.shared_experts.linear_fc2.weight": "mlp.shared_experts.down_proj.weight",
            "mlp.linear_fc1.weight": ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
            "mlp.shared_experts.linear_fc1.weight": [
                "mlp.shared_experts.gate_proj.weight",
                "mlp.shared_experts.up_proj.weight",
            ],
            "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
            "mlp.router.weight": "mlp.gate.weight",
            "mlp.router.expert_bias": "mlp.gate.e_score_correction_bias",
        }
        convert_names = []
        layer_number = name.split(".")[2]
        name_after_layer = name.split(f".{layer_number}.")[1]
        if name_after_layer in name_map_after_layer:
            mapped_name = name_map_after_layer[name_after_layer]
            if isinstance(mapped_name, list):
                assert len(params) == len(mapped_name)
                for one in mapped_name:
                    convert_names.append(f"model.layers.{layer_number}.{one}")
            else:
                assert len(params) == 1
                convert_names.append(f"model.layers.{layer_number}.{mapped_name}")
        else:
            if "mlp.experts.linear_fc1.weight" in name:
                expert_id = name.split("weight")[-1]
                convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight")
                convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight")
                assert len(params) == 2
            elif "mlp.experts.linear_fc2.weight" in name:
                expert_id = name.split("weight")[-1]
                convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight")
                assert len(params) == 1
            else:
                raise NotImplementedError(f"Unsupported parameter name: {name}")

        return convert_names, params

    def _convert_mtp_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        assert self.mcore_config.mtp_num_layers == 1, "only support one mtp layer for now"
        assert self.mcore_config.num_layers == 61, "only support 61 layers for now"
        direct_name_mapping = {
            "mtp.layers.0.enorm.weight": "model.layers.61.enorm.weight",
            "mtp.layers.0.hnorm.weight": "model.layers.61.hnorm.weight",
            "mtp.layers.0.eh_proj.weight": "model.layers.61.eh_proj.weight",
            "mtp.layers.0.final_layernorm.weight": "model.layers.61.shared_head.norm.weight",
        }
        if name in direct_name_mapping:
            return [direct_name_mapping[name]], [params[0]]
        assert "mtp.layers.0.transformer_layer" in name, "only support transformer layer for now"
        # use proxy name to convert
        proxy_name = name.replace("mtp.layers.0.transformer_layer", "decoder.layers.61")
        if "self_attention" in proxy_name or "input_layernorm.weight" in proxy_name:
            convert_names, params = self._convert_attention_param(proxy_name, params)
        elif "mlp" in proxy_name:
            convert_names, params = self._convert_mlp_param(proxy_name, params)
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names, params

    def convert_param(self, name: str, params_one_group: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        direct_name_mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }
        if name in direct_name_mapping:
            return [direct_name_mapping[name]], [params_one_group[0]]
        if "mtp" in name:
            return self._convert_mtp_param(name, params_one_group)
        elif "self_attention" in name or "input_layernorm.weight" in name:
            return self._convert_attention_param(name, params_one_group)
        elif "mlp" in name:
            return self._convert_mlp_param(name, params_one_group)
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")


class McoreToHFWeightConverterMixtral(McoreToHFWeightConverterDense):
    def _convert_mlp_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # decoder.layers.0.mlp.router.weight
        # decoder.layers.0.mlp.experts.linear_fc1.weight0 - weight7
        # decoder.layers.0.mlp.experts.linear_fc2.weight0 - weight7

        layer_number = name.split(".")[2]
        convert_names = []
        if "pre_mlp_layernorm" in name:
            convert_names.append(f"model.layers.{layer_number}.post_attention_layernorm.weight")
        elif "mlp.router.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.block_sparse_moe.gate.weight")
        elif "mlp.experts.linear_fc1.weight" in name:
            expert_id = name.split("weight")[-1]
            convert_names.append(f"model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w1.weight")
            convert_names.append(f"model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w3.weight")
        elif "mlp.experts.linear_fc2.weight" in name:
            expert_id = name.split("weight")[-1]
            convert_names.append(f"model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w2.weight")
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names, params


class McoreToHFWeightConverterQwen3Moe(McoreToHFWeightConverterDense):
    def _convert_mlp_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # qwen3 moe no share expert

        # 'decoder.layers.0.pre_mlp_layernorm.weight',
        # 'decoder.layers.0.mlp.router.weight',
        # moe1
        # 'decoder.layers.0.mlp.experts.linear_fc1.weight0',
        # 'decoder.layers.0.mlp.experts.linear_fc1.weight1',
        # 'decoder.layers.0.mlp.experts.linear_fc1.weight2',
        # 'decoder.layers.0.mlp.experts.linear_fc1.weight3',
        # moe2
        # 'decoder.layers.0.mlp.experts.linear_fc2.weight0',
        # 'decoder.layers.0.mlp.experts.linear_fc2.weight1',
        layer_number = name.split(".")[2]
        convert_names = []
        if "pre_mlp_layernorm" in name:
            convert_names.append(f"model.layers.{layer_number}.post_attention_layernorm.weight")
            assert len(params) == 1
        elif "mlp.router.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.mlp.gate.weight")
            assert len(params) == 1
        elif "mlp.experts.linear_fc1" in name:  # split gate_proj and up_proj
            expert_id = name.split("weight")[-1]
            convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight")
            convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight")
            assert len(params) == 2
        elif "mlp.experts.linear_fc2" in name:
            expert_id = name.split("weight")[-1]
            convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight")
            assert len(params) == 1
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names, params


class McoreToHFWeightConverterYuan(McoreToHFWeightConverterDense):
    def convert_param(self, name: str, params_one_group: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        direct_name_mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }
        if name in direct_name_mapping:
            return [direct_name_mapping[name]], [params_one_group[0]]

        if "self_attention" in name or "input_layernorm" in name:
            return self._convert_attention_param(name, params_one_group)
        elif "mlp" in name:
            return self._convert_mlp_param(name, params_one_group)
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")

    def _convert_attention_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        layer_number = name.split(".")[2]
        convert_names = []
        if "input_layernorm" in name:
            param_type = name.split(".")[-1]
            assert param_type == "weight"
            convert_names.append(f"model.layers.{layer_number}.input_layernorm.{param_type}")
            assert len(params) == 1
        elif "self_attention.linear_qk" in name:
            param_type = name.split(".")[-1]
            assert param_type == "bias" or param_type == "weight"
            convert_names.append(f"model.layers.{layer_number}.self_attn.get_query_key.{param_type}")
            assert len(params) == 1
        elif "self_attention.linear_v" in name:
            param_type = name.split(".")[-1]
            assert param_type == "weight"
            convert_names.append(f"model.layers.{layer_number}.self_attn.v_proj.{param_type}")
            assert len(params) == 1
        elif "self_attention.lf_gate.conv1" in name or "self_attention.lf_gate.conv2" in name:
            conv_name = name.split(".")[-2]
            param_type = name.split(".")[-1]
            assert param_type in ["bias", "weight"]
            convert_names.append(f"model.layers.{layer_number}.self_attn.lf_gate.{conv_name}.{param_type}")
            assert len(params) == 1
        elif "self_attention.lf_gate.output_layernorm" in name:
            param_type = name.split(".")[-1]
            assert param_type == "weight"
            convert_names.append(f"model.layers.{layer_number}.self_attn.lf_gate.output_layernorm.{param_type}")
            assert len(params) == 1
        elif "self_attention.linear_proj.weight" in name:
            param_type = name.split(".")[-1]
            assert param_type == "weight"
            convert_names.append(f"model.layers.{layer_number}.self_attn.o_proj.{param_type}")
            assert len(params) == 1
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names, params

    def _convert_mlp_param(self, name: str, params: list[torch.Tensor], num_experts: Optional[Union[int, List[int]]] = None) -> tuple[list[str], list[torch.Tensor]]:
        layer_number = name.split(".")[2]
        convert_names = []
        if "pre_mlp_layernorm" in name:
            convert_names.append(f"model.layers.{layer_number}.post_attention_layernorm.weight")
            assert len(params) == 1
        elif "attention_router" in name:
            # attention router
            convert_names.append(f"model.layers.{layer_number}.mlp.router.query_key_value.weight")
        elif "mlp.router.weight" in name:
            param_type = name.split(".")[-1]
            assert param_type == "weight"
            convert_names.append(f"model.layers.{layer_number}.mlp.router.{param_type}")
            assert len(params) == 1
        elif "mlp.experts.linear_fc1.weight" in name:
            exp_id = name.split('.weight')[-1]
            convert_names.append(f"model.layers.{layer_number}.mlp.experts.w1.{exp_id}.weight")
            assert len(params) == 2, f'{name} length {len(params)}'
            params = [torch.cat(params, dim=0)]
        elif "mlp.experts.linear_fc2.weight" in name:
            exp_id = name.split('.weight')[-1]
            convert_names.append(f"model.layers.{layer_number}.mlp.experts.w2.{exp_id}.weight")
            assert len(params) == 1
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names, params

        
class McoreToHFWeightConverterYuanVL(McoreToHFWeightConverterDense):
    def convert_param(self, name: str, params_one_group: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # mcore
        # imagemlp.linear_fc1.weight
        # imagemlp.linear_fc2.weight
        # imagemlp.imagemlp_layernorm.weight
        # embedding.word_embeddings.weight
        # decoder.layers.0.input_layernorm.weight
        # decoder.layers.0.self_attention.linear_proj.weight
        # decoder.layers.0.self_attention.linear_qk.weight
        # decoder.layers.0.self_attention.linear_v.weight
        # decoder.layers.0.self_attention.lf_gate.conv1.weight
        # decoder.layers.0.self_attention.lf_gate.conv1.bias
        # decoder.layers.0.self_attention.lf_gate.conv2.weight
        # decoder.layers.0.self_attention.lf_gate.conv2.bias
        # decoder.layers.0.self_attention.lf_gate.output_layernorm.weight
        # decoder.layers.0.pre_mlp_layernorm.weight
        # decoder.layers.0.mlp.router.attention_router.weight
        # decoder.layers.0.mlp.experts.linear_fc1.weight0
        # decoder.layers.0.mlp.experts.linear_fc1.weight1
        # decoder.layers.0.mlp.experts.linear_fc1.weight2
        # decoder.layers.0.mlp.experts.linear_fc1.weight3
        # decoder.layers.0.mlp.experts.linear_fc2.weight0
        # decoder.layers.0.mlp.experts.linear_fc2.weight1
        # decoder.layers.0.mlp.experts.linear_fc2.weight2
        # decoder.layers.0.mlp.experts.linear_fc2.weight3
        # decoder.final_layernorm.weight
        # output_layer.weight
        # hf
        # 'language_model.model.layers.0.input_layernorm.weight'
        # 'language_model.model.layers.0.self_attn.get_query_key.weight'
        # 'language_model.model.layers.0.self_attn.v_proj.weight'
        # 'language_model.model.layers.0.self_attn.o_proj.weight'
        # 'language_model.model.layers.0.self_attn.lf_gate.conv1.weight'
        # 'language_model.model.layers.0.self_attn.lf_gate.conv1.bias'
        # 'language_model.model.layers.0.self_attn.lf_gate.conv2.weight'
        # 'language_model.model.layers.0.self_attn.lf_gate.conv2.bias'
        # 'language_model.model.layers.0.self_attn.lf_gate.output_layernorm.weight'
        # 'language_model.model.layers.0.post_attention_layernorm.weight'
        # "language_model.model.layers.0.mlp.router.query_key_value.weight"
        # "language_model.model.layers.0.mlp.experts.w1.0.weight"
        # "language_model.model.layers.0.mlp.experts.w1.1.weight"
        # "language_model.model.norm.weight"
        # "language_model.lm_head.weight"
        direct_name_mapping = {
            "embedding.word_embeddings.weight": "language_model.model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "language_model.model.norm.weight",
            "output_layer.weight": "language_model.lm_head.weight",
            "imagemlp.linear_fc2.weight": "imagemlp.down_proj.weight",
            "imagemlp.imagemlp_layernorm.weight": "imagemlp_layernorm.weight",
        }
        imagemlp_linear_fc1 = {
            "imagemlp.linear_fc1.weight": ["imagemlp.up_proj.weight", "imagemlp.gate_proj.weight"]
        }
        if name in direct_name_mapping:
            return [direct_name_mapping[name]], [params_one_group[0]]
        if name in imagemlp_linear_fc1:
            assert len(params_one_group) == 1, f'imagemlp_linear_fc1 len must be 1 {len(params_one_group)}, {params_one_group[0].shape}'
            imagemlp_up_proj, imagemlp_gate_proj = torch.chunk(params_one_group[0], 2, dim=0)
            return imagemlp_linear_fc1["imagemlp.linear_fc1.weight"], [imagemlp_up_proj, imagemlp_gate_proj]

        if "self_attention" in name or "input_layernorm" in name:
            return self._convert_attention_param(name, params_one_group)
        elif "mlp" in name or "post_attention_layernorm" in name:
            return self._convert_mlp_param(name, params_one_group)
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")

    def _convert_attention_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        layer_number = name.split(".")[2]
        convert_names = []
        if "input_layernorm" in name:
            param_type = name.split(".")[-1]
            assert param_type == "weight"
            convert_names.append(f"language_model.model.layers.{layer_number}.input_layernorm.{param_type}")
            assert len(params) == 1
        elif "self_attention.linear_qk" in name:
            param_type = name.split(".")[-1]
            assert param_type == "bias" or param_type == "weight"
            convert_names.append(f"language_model.model.layers.{layer_number}.self_attn.get_query_key.{param_type}")
            assert len(params) == 1
        elif "self_attention.linear_v" in name:
            param_type = name.split(".")[-1]
            assert param_type == "weight"
            convert_names.append(f"language_model.model.layers.{layer_number}.self_attn.v_proj.{param_type}")
            assert len(params) == 1
        elif "self_attention.lf_gate.conv1" in name or "self_attention.lf_gate.conv2" in name:
            conv_name = name.split(".")[-2]
            param_type = name.split(".")[-1]
            assert param_type == "bias" or param_type == "weight"
            convert_names.append(f"language_model.model.layers.{layer_number}.self_attn.lf_gate.{conv_name}.{param_type}")
            assert len(params) == 1
        elif "self_attention.lf_gate.output_layernorm" in name:
            param_type = name.split(".")[-1]
            assert param_type == "weight"
            convert_names.append(f"language_model.model.layers.{layer_number}.self_attn.lf_gate.output_layernorm.{param_type}")
            assert len(params) == 1
        elif "self_attention.linear_proj.weight" in name:
            param_type = name.split(".")[-1]
            assert param_type == "weight"
            convert_names.append(f"language_model.model.layers.{layer_number}.self_attn.o_proj.{param_type}")
            assert len(params) == 1
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names, params

    def _convert_mlp_param(self, name: str, params: list[torch.Tensor], num_experts: Optional[Union[int, List[int]]] = None) -> tuple[list[str], list[torch.Tensor]]:
        layer_number = name.split(".")[2]
        convert_names = []
        if "pre_mlp_layernorm" in name:
            convert_names.append(f"language_model.model.layers.{layer_number}.post_attention_layernorm.weight")
            assert len(params) == 1
        elif "mlp.router.attention_router.weight" in name:
            # attention router
            convert_names.append(f"language_model.model.layers.{layer_number}.mlp.router.query_key_value.weight")
        elif "mlp.router.weight" in name:
            convert_names.append(f"language_model.model.layers.{layer_number}.mlp.router.weight")
            assert len(params) == 1
        elif "mlp.experts.linear_fc1.weight" in name:
            exp_id = name.split('.weight')[-1]
            convert_names.append(f"language_model.model.layers.{layer_number}.mlp.experts.w1.{exp_id}.weight")
            assert len(params) == 1, f'{name} length {len(params)}'
        elif "mlp.experts.linear_fc2.weight" in name:
            exp_id = name.split('.weight')[-1]
            convert_names.append(f"language_model.model.layers.{layer_number}.mlp.experts.w2.{exp_id}.weight")
            assert len(params) == 1
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names, params
