# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from transformers import AutoConfig, LlamaConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING

from .configuration_intern_vit import InternVisionConfig
from .configuration_yuan import YuanConfig

logger = logging.get_logger(__name__)


class YuanVLChatConfig(PretrainedConfig):
    model_type = 'yuanvl'
    is_composition = True
    sub_configs = {"llm_config": YuanConfig, "vision_config": InternVisionConfig}  # 声明子配置类型

    def __init__(
            self,
            vision_config=None,
            llm_config=None,
            use_backbone_lora=0,
            use_llm_lora=0,
            select_layer=-1,
            force_image_size=None,
            downsample_ratio=0.5,
            template=None,
            dynamic_image_size=False,
            use_thumbnail=False,
            tie_word_embeddings=False,
            ps_version='v1',
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            img_context_token_id=77188,** kwargs):

        # 初始化视觉子配置（确保为InternVisionConfig实例）
        if vision_config is None:
            # 输入为None时，直接实例化InternVisionConfig（而非字典）
            self.vision_config = InternVisionConfig(architectures=['InternVisionModel'])
            logger.info('vision_config is None. Initializing InternVisionConfig with default values.')
        elif isinstance(vision_config, dict):
            # 输入为字典时，用from_dict实例化
            self.vision_config = InternVisionConfig.from_dict(vision_config)
        else:
            # 输入已为实例时直接使用
            self.vision_config = vision_config

        # 初始化LLM子配置（确保为YuanConfig实例）
        if llm_config is None:
            # 输入为None时，直接实例化YuanConfig（而非字典）
            self.llm_config = YuanConfig(architectures=['YuanForCausalLM'])
            self.llm_config.tie_word_embeddings = tie_word_embeddings  # 显式设置属性
            logger.info('llm_config is None. Initializing YuanConfig with default values.')
        elif isinstance(llm_config, dict):
            # 输入为字典时，用from_dict实例化
            self.llm_config = YuanConfig.from_dict(llm_config)
            self.llm_config.tie_word_embeddings = tie_word_embeddings
        else:
            # 输入已为实例时直接使用，并同步tie_word_embeddings
            self.llm_config = llm_config
            self.llm_config.tie_word_embeddings = tie_word_embeddings

        # 其他属性初始化
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.img_context_token_id = img_context_token_id
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings  # 同步LLM的配置

        # 日志输出
        logger.info(f'vision_select_layer: {self.select_layer}')
        logger.info(f'ps_version: {self.ps_version}')
        logger.info(f'min_dynamic_patch: {self.min_dynamic_patch}')
        logger.info(f'max_dynamic_patch: {self.max_dynamic_patch}')

        super().__init__(**kwargs)

    @classmethod
    def from_sub_model_configs(
        cls,
        vision_config: InternVisionConfig,
        llm_config: YuanConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`YuanVLChatConfig`] (or a derived class) from bark sub-models configuration.

        Returns:
            [``YuanVLChatConfig``]: An instance of a configuration object
        """
        return cls(
            vision_config=vision_config.to_dict(),
            llm_config=llm_config.to_dict(),
            **kwargs,
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].
        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['dynamic_image_size'] = self.dynamic_image_size
        output['use_thumbnail'] = self.use_thumbnail
        output['ps_version'] = self.ps_version
        output['min_dynamic_patch'] = self.min_dynamic_patch
        output['max_dynamic_patch'] = self.max_dynamic_patch

        return output
