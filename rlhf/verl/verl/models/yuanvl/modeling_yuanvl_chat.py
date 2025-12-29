# --------------------------------------------------------
# YuanVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import (Any, Callable, Iterable, List, Literal, Mapping, Optional,
                    Set, Tuple, Type, TypedDict, Union)

import torch.utils.checkpoint
import transformers
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from transformer_engine.pytorch import RMSNorm
from transformers.activations import ACT2FN

from verl.models.configs.configuration_yuanvl import YuanVLChatConfig
from verl.models.yuanvl.conversation import get_conv_template
from verl.models.yuanvl.modeling_intern_vit import InternVisionModel, has_flash_attn
from verl.models.yuanvl.modeling_yuanlm2 import YuanForCausalLM
from verl.models.yuanvl.utils import flatten_bn, merge_multimodal_embeddings

logger = logging.get_logger(__name__)


class InternVLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """
    Shape: `(batch_size, 1 + num_patches, num_channels, height, width)`

    Note that `num_patches` may be different for each batch, in which case
    the data is passed as a list instead of a batched tensor.
    """
    patches_per_image: List[int]
    """
    List of number of total patches for each image in the batch.
    """


class InternVLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: Any # in vllm vision this is a NestedTensors
    """
    A tensor of shape `(num_images, total_image_feature_size, hidden_size)`
    or a list of tensors of shape `(total_image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


InternVLImageInputs = Union[InternVLImagePixelInputs,
                            InternVLImageEmbeddingInputs]


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

class YuanImageMLP(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_size, bias=False)

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
        
        self.act_fn = ACT2FN[hidden_act]
    
    @torch.compile
    def swiglu(self, y_1, y_2):
        return self.act_fn(y_1) * y_2
    
    def forward(self, x):
        x1 = self.up_proj(x)
        x2 = self.gate_proj(x)
        x3 = self.swiglu(x1, x2)
        x = self.down_proj(x3)
        return x

class YuanVLChatModel(PreTrainedModel):
    config_class = YuanVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    _no_split_modules = ['InternVisionModel', 'YuanDeocderLayer']

    def __init__(self, config: YuanVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'YuanForCausalLM':
                self.language_model = YuanForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        self.pixel_unshuffle = torch.nn.PixelUnshuffle(downscale_factor=2)
        layernorm_epsilon = config.llm_config.rms_norm_eps

        self.imagemlp_input_hiddensize = int(config.vision_config.hidden_size / self.downsample_ratio ** 2)
        self.imagemlp_ffn_hidden_size = config.llm_config.ffn_hidden_size

        self.imagemlp = YuanImageMLP(self.imagemlp_input_hiddensize, self.imagemlp_ffn_hidden_size,
                     output_size=config.llm_config.hidden_size, hidden_act="silu")
        self.imagemlp_layernorm = RMSNorm(config.llm_config.hidden_size, eps=layernorm_epsilon)

        self.img_context_token_id = config.img_context_token_id
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    def _validate_pixel_values(self,
                                data: Union[torch.Tensor, List[torch.Tensor]]
                                ) -> Union[torch.Tensor, List[torch.Tensor]]:
        
        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)
            if actual_dims != expected_dims:
                expected_expr = (expected_dims)
                raise ValueError("The expected shape of pixel values in each batch element "
                                 f" is {expected_expr}. You supplied {tuple(d.shape)}.")
            # data的数据类型可以是tensor，也可以是List[tensor]
            # 从这一段上来看，image tensor的个数为 imbs*num_images
        for d in data:
            _validate_shape(d)
        return data



    def _parse_and_validate_image_input(self, 
                                        pixel_values: List[torch.Tensor] = None, 
                                        image_token_id: torch.Tensor = None, 
                                        image_embeds: torch.Tensor = None,
                                        ) -> Optional[InternVLImagePixelInputs]:
        # 没有图像数据
        if pixel_values is None and image_embeds is None:
            return None
        
        # 传入数据有image_embeds
        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return InternVLImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )
        
        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")
            patches_per_image = []
            # bsz/request循环
            for request_pixel_values in pixel_values:
                # 每个request的images循环
                patches_per_image.append(request_pixel_values.shape[0])

            # We need to flatten (B, N, P) to (B*N*P)
            # so we call flatten_bn twice.
            # (total_patches, 3, h, w)
            return InternVLImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(flatten_bn(pixel_values)),
                patches_per_image=patches_per_image)
        raise AssertionError("This line should be unreachable")
    
    def _process_image_input(
            self,
            image_input: InternVLImageInputs,
    ) -> Tuple[torch.Tensor] :
        if image_input["type"] == "image_embeds":
            return image_input["data"]
        assert self.vision_model is not None
        # (total_patches, tokens_per_image, llm_config.hidden_size)
        image_embeds = self.extract_feature(image_input["data"])

        patches_per_image = image_input["patches_per_image"]

        # Only one image in the current batch
        # bsz=1的情况，直接返回image_embeds
        if len(patches_per_image) == 1:
            # 返回一个tensor，[1, num_patches*256, text_config.hidden_size]
            image_embeds = image_embeds.view(-1, self.config.llm_config.hidden_size).unsqueeze(1)
            return image_embeds
        # NOTE: Image embeddings are split into separate tensors for each image
        # by the size of each embedding.
        # feature_size 每个patch 256个token位置
        feature_size = image_embeds.shape[1]
        # (total_image_tokens, llm_config.hidden_size)
        image_embeds = image_embeds.view(-1, self.config.llm_config.hidden_size)
        image_feature_sizes = [num_patches * feature_size for num_patches in patches_per_image]
        # 切分后得到一个Tuple，元组每个元胞表示一个image的image_embed, [num_patches * 256, llm_config.hidden_size]
        image_embeds = image_embeds.split(image_feature_sizes)

        return image_embeds
        

    
    def get_multimodal_embeddings(self,
                                  pixel_values: Optional[List[torch.Tensor]] = None,
                                  image_token_id: Optional[List[torch.Tensor]] = None,
                                  image_embeds: Optional[List[torch.Tensor]] = None,
                                  image_input: InternVLImageInputs = None,
                                  ):
        image_input = self._parse_and_validate_image_input(pixel_values, image_token_id, image_embeds)
        if image_input is None:
            return None
        
        # image_input: (total_patches, 3, h, w)
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings
    
    def get_input_embeddings(
            self,
            input_ids: torch.Tensor,
            multimodal_embeddings: Optional[torch.Tensor]
            ) -> torch.Tensor:
        # 生成 token_embeddings
        inputs_embeds = self.language_model.model.get_input_embeddings(input_ids)
        # 将image embed放到img_context_token_id的位置
        if multimodal_embeddings is not None:
            assert self.img_context_token_id is not None
            # input_ids: torch.Tensor
            # inputs_embeds: torch.Tensor 
            # multimodal_embeddings: torch.Tensor
            # placeholder_token_id: img_context_token_id
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.img_context_token_id)
        return inputs_embeds
    
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: torch.Tensor = None,
            position_ids: torch.LongTensor = None,
            past_key_values: List[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[List[torch.Tensor]] = None,
            image_token_id: Optional[List[torch.Tensor]] = None,
            image_embeds: Optional[List[torch.Tensor]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            # (images, patches * token_per_image)
            vision_embeddings = self.get_multimodal_embeddings(pixel_values, image_token_id, image_embeds)
            # (tokens, hidden_size)
            inputs_embeds = self.get_input_embeddings(input_ids, vision_embeddings).permute(1, 0, 2)
            input_ids = None
        
        hidden_states = self.language_model.model(input_ids, attention_mask, position_ids, past_key_values, 
                                                  inputs_embeds, labels, use_cache, output_attentions, 
                                                  output_hidden_states, return_dict)

        return hidden_states

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    # Internvl vision
    def extract_feature(self, pixel_values):
        # pixel_values: (imbs * num_image, ic, ih, iw)
        pixel_values = pixel_values.to(torch.bfloat16)
        output = self.vision_model(pixel_values=pixel_values)
        vit_embeds=output[0]
        # vit_embeds: (imbs * num_images, h*w, vit_dim)
        vit_embeds = vit_embeds[:, 1:, :]

        '''h = w = int(vit_embeds.shape[1]**0.5)
        # vit_embeds: (imbs * num_images, vit_dim, h, w)
        vit_embeds = vit_embeds.view(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        '''
        pn, phw, pc = vit_embeds.shape
        ph = pw = int(phw**0.5)
        vit_embeds = vit_embeds.view(pn, ph, pw, pc).permute(0, 3, 1, 2)
        vit_embeds = self.pixel_unshuffle(vit_embeds)
        pn, pc, ph, pw = vit_embeds.shape
        vit_embeds = vit_embeds.view(pn, pc, ph * pw).permute(0, 2, 1)
        num_images, cvs, chs = vit_embeds.shape
        vit_embeds = vit_embeds.reshape(1, -1, vit_embeds.shape[-1]).permute(1, 0, 2)
        vit_embeds = self.imagemlp(vit_embeds)
        vit_embeds = self.imagemlp_layernorm(vit_embeds)
        vit_embeds = vit_embeds.view(num_images, cvs, -1)
        return vit_embeds
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> torch.LongTensor:

        
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.get_multimodal_embeddings(pixel_values)
                inputs_embeds = self.get_input_embeddings(input_ids, vit_embeds)
                input_ids = None
        
        
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            position_ids=position_ids,
            max_length=8192,
            use_cache=True,
        )


        return outputs
