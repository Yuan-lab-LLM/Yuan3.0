"""Inference-only YuanVL model compatible with HuggingFace weights."""
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence

from typing import (Iterable, Literal, Mapping, Optional,
                    Annotated, Tuple, TypedDict, Union, TypeVar)
from PIL import Image

import torch
import torch.nn as nn
from transformers import BatchFeature, PretrainedConfig, TensorType

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.multimodal.inputs import (MultiModalDataDict, NestedTensors,
                                    MultiModalFieldConfig, MultiModalKwargsItems)
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                   ImageSize, MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.tokenizers.protocol import TokenizerLike
from transformers.activations import ACT2FN
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.sequence import IntermediateTensors
from vllm.logger import init_logger

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.intern_vit import (InternVisionModel,
                                                   InternVisionPatchModel)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.yuan import YuanForCausalLM
from transformers import CLIPImageProcessor
from vllm.utils.platform_utils import (
    is_pin_memory_available,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape


from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP, MixtureOfExperts


BOS = '<BOS>'
SEP = '<sep>'
IMG_START = '<IMAGE>'
IMG_END = '</IMAGE>'
IMG_CONTEXT = '<pad>'

MAX_IMAGE_FEATURE_SIZE_WIDTH = 3000
MAX_IMAGE_FEATURE_SIZE_HEIGHT = 500

logger = init_logger(__name__)


class YuanVLImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - bnp: Batch size * number of images * (1 + num_patches)
        - c: Number of channels (3)
        - h: Height of each image patch
        - w: Width of each image patch
    """

    type: Literal["pixel_values"]
    pixel_values_flat: Annotated[torch.Tensor, TensorShape("bnp", 3, "h", "w")]
    num_patches: Annotated[torch.Tensor, TensorShape("bn")]


class YuanVLImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - n: Number of images
        - f: Total image feature size
        - h: Hidden size (must match the hidden size of language model backbone)
    """
    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor | list[torch.Tensor], TensorShape("n", "f", "h")]


YuanVLImageInputs = Union[YuanVLImagePixelInputs,
                          YuanVLImageEmbeddingInputs]


def _convert_to_rgb(image):
    return image.convert('RGB')


# copied from https://huggingface.co/OpenGVLab/InternVL2-1B
def build_transform(input_size: int):
    process_dict = {
                    "crop_size": 448,
                    "do_center_crop": 'true',
                    "do_normalize": 'true',
                    "do_resize": 'true',
                    "feature_extractor_type": "CLIPFeatureExtractor",
                    "image_mean": [
                      0.485,
                      0.456,
                      0.406
                    ],
                    "image_std": [
                      0.229,
                      0.224,
                      0.225
                    ],
                    "resample": 3,
                    "size": 448
                  }
    trans = CLIPImageProcessor.from_dict(process_dict)
    return trans


# copied from https://huggingface.co/OpenGVLab/InternVL2-1B
def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
    threshold: float = 1.0,
) -> tuple[int, int]:
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        size_diff_length = abs(
            (
                (ratio[0]*image_size + ratio[1]*image_size)-(width+height)
            ) / (width+height)
        )
        # if ratio_diff < best_ratio_diff:
        if ratio_diff < best_ratio_diff and size_diff_length <= threshold:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def resolve_yuanvl_min_max_num(
    *,
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: bool,
    use_thumbnail: bool,
) -> tuple[int, int]:
    min_dynamic_patch = min_dynamic_patch if dynamic_image_size else 1
    max_dynamic_patch = max_dynamic_patch if dynamic_image_size else 1

    if use_thumbnail and max_dynamic_patch != 1:
        max_dynamic_patch += 1

    return min_dynamic_patch, max_dynamic_patch


def get_yuanvl_target_ratios(
    min_num: int,
    max_num: int,
) -> list[tuple[int, int]]:
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1)
                     for j in range(1, n + 1) if min_num <= i * j <= max_num}
    return sorted(target_ratios, key=lambda x: x[0] * x[1])


def calculate_yuanvl_targets(
    *,
    orig_width: int,
    orig_height: int,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
    min_num: int = 1,
    max_num: int = 9,
    threshold: float = 1.0,
) -> tuple[int, int, int]:
    aspect_ratio = orig_width / orig_height
    '''
    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        width=orig_width,
        height=orig_height,
        image_size=image_size,
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # add thumbnail image if num_blocks != 1
    if use_thumbnail and blocks != 1:
        blocks += 1
    print(use_thumbnail, blocks, target_width, target_height)
    return blocks, target_width, target_height
    '''
    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size, threshold)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    # print(use_thumbnail, blocks, target_width, target_height)
    return blocks, target_width, target_height


def dynamic_preprocess_yuanvl(
    image: Image.Image,
    *,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
    min_num: int = 1,
    max_num: int = 9,
    threshold: float = 1.0,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the number of blocks without thumbnail
    blocks, target_width, target_height = calculate_yuanvl_targets(
        orig_width=orig_width,
        orig_height=orig_height,
        target_ratios=target_ratios,
        image_size=image_size,
        use_thumbnail=False,
        min_num=min_num,
        max_num=max_num,
        threshold=threshold,
    )

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
def image_to_pixel_values_yuanvl(
    image: Image.Image,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
    threshold: float = 1.0,
) -> torch.Tensor:
    target_ratios = get_yuanvl_target_ratios(min_num, max_num)

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess_yuanvl(
        image,
        target_ratios=target_ratios,
        image_size=input_size,
        use_thumbnail=use_thumbnail,
        min_num=min_num,
        max_num=max_num,
        threshold=threshold,
    )

    pixel_values = torch.stack([transform(image, return_tensors='pt').pixel_values.squeeze(0) for image in images])
    return pixel_values


class BaseYuanVLProcessor(ABC):
    """
    The code to insert image tokens is based on:
    https://huggingface.co/OpenGVLab/InternVL2-1B/blob/main/modeling_internvl_chat.py#L252
    """

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: TokenizerLike,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        image_size: int = config.vision_config.image_size
        patch_size: int = config.vision_config.patch_size

        if min_dynamic_patch is None:
            min_dynamic_patch = config.min_dynamic_patch
        assert isinstance(min_dynamic_patch, int)

        if dynamic_image_size is None:
            dynamic_image_size = config.dynamic_image_size
        assert isinstance(dynamic_image_size, bool)

        if max_dynamic_patch is None:
            max_dynamic_patch = config.max_dynamic_patch
        assert isinstance(max_dynamic_patch, int)

        self.num_image_token = int(
            (image_size // patch_size)**2 * (config.downsample_ratio**2))
        self.image_size = image_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail: bool = config.use_thumbnail

    @property
    @abstractmethod
    def image_token_id(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_image_repl(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> PromptUpdateDetails[str]:
        raise NotImplementedError

    # 需要判断这个函数是否有必要
    def resolve_min_max_num(
        self,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        use_thumbnail: Optional[bool] = None,
    ) -> tuple[int, int]:
        min_dynamic_patch = (self.min_dynamic_patch if min_dynamic_patch
                             is None else min_dynamic_patch)
        max_dynamic_patch = (self.max_dynamic_patch if max_dynamic_patch
                             is None else max_dynamic_patch)
        dynamic_image_size = (self.dynamic_image_size if dynamic_image_size
                              is None else dynamic_image_size)
        use_thumbnail = (self.use_thumbnail
                         if use_thumbnail is None else use_thumbnail)

        return resolve_yuanvl_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
        )

    def resolve_target_ratios(
        self,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        use_thumbnail: Optional[bool] = None,
    ) -> list[tuple[int, int]]:
        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
        )

        return get_yuanvl_target_ratios(min_num, max_num)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        target_ratios = self.resolve_target_ratios(
            use_thumbnail=False,  # Applied in calculate_targets
        )

        num_patches, _, _ = calculate_yuanvl_targets(
            orig_width=image_width,
            orig_height=image_height,
            image_size=self.image_size,
            target_ratios=target_ratios,
            use_thumbnail=self.use_thumbnail,
        )
        if self.use_thumbnail and num_patches != 1:
            num_patches += 1
        return num_patches * self.num_image_token

    def _images_to_pixel_values_lst(
        self,
        images: list[Image.Image],
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> list[torch.Tensor]:
        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=False,  # Applied in image_to_pixel_values
        )

        return [
            image_to_pixel_values_yuanvl(
                image,
                input_size=self.image_size,
                min_num=min_num,
                max_num=max_num,
                use_thumbnail=self.use_thumbnail,
            ) for image in images
        ]

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        if len(images) == 0:
            image_inputs = {}
        else:
            pixel_values_lst = self._images_to_pixel_values_lst(
                images,
                min_dynamic_patch=min_dynamic_patch,
                max_dynamic_patch=max_dynamic_patch,
                dynamic_image_size=dynamic_image_size,
            )
            image_inputs: dict[str, NestedTensors] = {
                "pixel_values_flat":
                torch.cat(pixel_values_lst),
                "image_num_patches":
                torch.tensor([len(item) for item in pixel_values_lst]),
            }

            for pixel_values in pixel_values_lst:
                num_patches = pixel_values.shape[0]
                feature_size = num_patches * self.num_image_token

                image_repl = self.get_image_repl(feature_size, num_patches)
                text = [t.replace('<image>', image_repl.full, 1) for t in text]

        text_inputs = self.tokenizer(text)

        return BatchFeature(
            {
                **text_inputs,
                **image_inputs,
            },
            tensor_type=return_tensors,
        )


class YuanVLProcessor(BaseYuanVLProcessor):

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[IMG_CONTEXT]

    def get_image_repl(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> PromptUpdateDetails[str]:
        repl_features = IMG_CONTEXT * feature_size
        repl_full = BOS + IMG_START + repl_features + IMG_END

        return PromptUpdateDetails.select_text(repl_full, IMG_CONTEXT)


class BaseYuanVLProcessingInfo(BaseProcessingInfo):

    @abstractmethod
    def get_hf_processor(
        self,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> BaseYuanVLProcessor:
        raise NotImplementedError

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[BaseYuanVLProcessor],
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()

        return processor.get_num_image_tokens(
            image_width=image_width,
            image_height=image_height,
        )

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            processor=None,
        )

    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_hf_processor()

        base_size = processor.image_size
        target_ratios = processor.resolve_target_ratios()

        largest_feature_size, largest_feature_pinpoint = 0, None
        for wr, hr in target_ratios:
            width, height = base_size * wr, base_size * hr

            feat_size = self.get_num_image_tokens(
                image_width=width,
                image_height=height,
                processor=processor,
            )
            if feat_size > largest_feature_size:
                largest_feature_size = feat_size
                largest_feature_pinpoint = ImageSize(width=width,
                                                     height=height)

        if largest_feature_size == 0 or largest_feature_pinpoint is None:
            raise ValueError("Cannot have a largest feature size of 0!")

        return largest_feature_pinpoint


_I = TypeVar("_I", bound=BaseYuanVLProcessingInfo)


class YuanVLDummyInputsBuilder(BaseDummyInputsBuilder[_I]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        return "<image>" * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        num_images = mm_counts.get("image", 0)
        image_overrides = mm_options.get("image")

        return {
            "image":
            self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class YuanVLMultiModalProcessor(BaseMultiModalProcessor[_I]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        image_token_id = hf_processor.image_token_id

        # Since there may be extra tokens in the feature placeholders,
        # we need to pass the image token ID to the model to select the
        # tokens to merge from the vision encoder outputs
        processed_outputs["image_token_id"] = torch.tensor(image_token_id)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_num_patches = hf_inputs.get("image_num_patches", torch.empty(0))
        num_images = len(image_num_patches)

        return dict(
            pixel_values_flat=MultiModalFieldConfig.flat_from_sizes(
                "image", image_num_patches),
            image_num_patches=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
            image_token_id=MultiModalFieldConfig.shared("image", num_images),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        out_mm_data = out_mm_kwargs.get_data()
        if "image_num_patches" in out_mm_data:
            image_num_patches = out_mm_data["image_num_patches"]
            assert isinstance(image_num_patches, torch.Tensor)
            image_num_patches = image_num_patches.tolist()
        elif "image_embeds" in out_mm_data:
            # TODO: Use image size information in dictionary embedding inputs
            # to compute num_patches (similar to Qwen2-VL)
            image_num_patches = [None] * len(out_mm_data["image_embeds"])
        else:
            image_num_patches = []

        def get_replacement_yuanvl(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))

            if isinstance(images, ImageEmbeddingItems):
                feature_size = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                feature_size = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    processor=hf_processor,
                )

            num_patches = image_num_patches[item_idx]
            if num_patches is not None:
                assert isinstance(num_patches, int)

            return hf_processor.get_image_repl(feature_size, num_patches)

        return [
            PromptReplacement(
                modality="image",
                target="<image>",
                replacement=get_replacement_yuanvl,
            )
        ]


class YuanVLProcessingInfo(BaseYuanVLProcessingInfo):

    def get_hf_processor(
        self,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        **kwargs: object,
    ) -> YuanVLProcessor:

        if min_dynamic_patch is not None:
            kwargs["min_dynamic_patch"] = min_dynamic_patch
        if max_dynamic_patch is not None:
            kwargs["max_dynamic_patch"] = max_dynamic_patch
        if dynamic_image_size is not None:
            kwargs["dynamic_image_size"] = dynamic_image_size

        return self.ctx.init_processor(
            YuanVLProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            **kwargs,
        )


class YuanImageMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # self.up_proj = ColumnParallelLinear(hidden_size,
        self.up_proj = ReplicatedLinear(hidden_size,
                                        intermediate_size,
                                        bias=False)
        # self.gate_proj= ColumnParallelLinear(hidden_size,
        self.gate_proj = ReplicatedLinear(hidden_size,
                                          intermediate_size,
                                          bias=False)
        # self.down_proj = RowParallelLinear(intermediate_size,
        self.down_proj = ReplicatedLinear(intermediate_size,
                                          output_size,
                                          bias=False)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        x1, _ = self.up_proj(x)
        x3 = self.act_fn(x1)
        x2, _ = self.gate_proj(x)
        x, _ = self.down_proj(x2 * x3)
        return x


@MULTIMODAL_REGISTRY.register_processor(
    YuanVLMultiModalProcessor,
    info=YuanVLProcessingInfo,
    dummy_inputs=YuanVLDummyInputsBuilder)
class YuanVLChatModel(nn.Module, SupportsMultiModal, SupportsPP, MixtureOfExperts):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        # self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        # self._patch_quant_config(config, quant_config)

        self.config = config
        self.text_config = vllm_config.model_config.hf_text_config
        self.multimodal_config = multimodal_config

        self.img_context_token_id = config.img_context_token_id

        self.downsample_ratio = config.downsample_ratio
        self.imagemlp_input_hiddensize = int(config.vision_config.hidden_size / self.downsample_ratio ** 2)

        self.is_mono = False
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.vision_model = self._init_vision_model(
                config,
                quant_config=quant_config,
                is_mono=self.is_mono,
                prefix=maybe_prefix(prefix, ""),
                # prefix=maybe_prefix(prefix, "vision_model"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (self.language_model.make_empty_intermediate_tensors)

        self.pixel_unshuffle = torch.nn.PixelUnshuffle(downscale_factor=2)
        self.imagemlp_ffn_hidden_size = config.text_config.ffn_hidden_size
        self.imagemlp = YuanImageMLP(self.imagemlp_input_hiddensize,
                                     self.imagemlp_ffn_hidden_size,
                                     output_size=config.text_config.hidden_size,
                                     hidden_act=config.text_config.hidden_act)
        self.imagemlp_layernorm = RMSNorm(
            config.text_config.hidden_size,
            eps=config.text_config.rms_norm_eps
        )
        self.num_moe_layers = self.language_model.num_moe_layers
        self.num_expert_groups = self.language_model.num_expert_groups
        self.num_shared_experts = self.language_model.num_shared_experts
        self.num_logical_experts = self.language_model.num_logical_experts
        self.num_physical_experts = self.language_model.num_physical_experts
        self.num_local_physical_experts = self.language_model.num_local_physical_experts
        self.num_routed_experts = self.language_model.num_routed_experts
        self.num_redundant_experts = self.language_model.num_redundant_experts
        self.num_routed_experts_list = self.language_model.num_routed_experts_list

    def _patch_quant_config(
        self, config: PretrainedConfig, quant_config: QuantizationConfig
    ):
        # the awq models from OpenGVLab missing `modules_to_not_convert`
        # patch the quant_config to add `modules_to_not_convert` back
        if isinstance(quant_config, AWQConfig):
            text_config = config.text_config
            llm_quant_config = getattr(text_config, "quantization_config", None)
            if (not quant_config.modules_to_not_convert) and (
                llm_quant_config is not None
            ):
                quant_config.modules_to_not_convert.append("vision_model")

    def _init_vision_model(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        *,
        is_mono: bool,
        prefix: str,
    ):
        if not is_mono:
            vision_feature_layer = config.select_layer
            if vision_feature_layer < 0:
                num_hidden_layers = (
                    config.vision_config.num_hidden_layers + vision_feature_layer + 1
                )
            else:
                num_hidden_layers = vision_feature_layer + 1

            return InternVisionModel(
                config.vision_config,
                quant_config=quant_config,
                num_hidden_layers_override=num_hidden_layers,
                prefix=prefix,
            )
        else:
            return InternVisionPatchModel(config.vision_config)

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            pass
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        # pixel_values: (imbs * num_image, ic, ih, iw)
        pixel_values = pixel_values.to(torch.bfloat16)
        vit_embeds = self.vision_model(pixel_values=pixel_values)
        # vit_embeds: (imbs * num_images, h*w, vit_dim)
        vit_embeds = vit_embeds[:, 1:, :]
        h = w = int(vit_embeds.shape[1]**0.5)
        vit_embeds = vit_embeds.view(vit_embeds.shape[0], h, w, -1).permute(0, 3, 1, 2)
        vit_embeds = self.pixel_unshuffle(vit_embeds)
        pn, pc, ph, pw = vit_embeds.shape
        vit_embeds = vit_embeds.view(pn, pc, ph*pw).permute(0, 2, 1)
        num_images, cvs, chs = vit_embeds.shape
        # chs = imagemlp_input_hiddensize = vit_dim * 4
        assert self.imagemlp_input_hiddensize == chs
        # pn = b * num_images
        # (num_images*ph*pw, imbs, vit_dim)
        vit_embeds = vit_embeds.contiguous().view(-1, num_images * cvs, chs).permute(1, 0, 2).contiguous()
        # vit_embeds: (num_images * ph * pw, imbs, hidden_size)
        vit_embeds = self.imagemlp(vit_embeds)
        vit_embeds = self.imagemlp_layernorm(vit_embeds)
        # vit_embeds: (imbs, num_images * ph * pw, hidden_size)
        vit_embeds = vit_embeds.view(num_images, cvs, -1)
        return vit_embeds

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:

        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = (expected_dims)
                raise ValueError(
                    "The expected shape of pixel values in each batch element "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")
        # data的数据类型可以是tensor，也可以是List[tensor]
        # 从这一段看，image tensor的个数为 imbs * num_images
        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[YuanVLImageInputs]:
        pixel_values_flat = kwargs.pop("pixel_values_flat", None)
        image_num_patches = kwargs.pop("image_num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)
        # 没有图像数据
        if pixel_values_flat is None and image_embeds is None:
            return None

        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return YuanVLImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        image_token_id = kwargs["image_token_id"]
        if isinstance(image_token_id, torch.Tensor):
            image_token_id = image_token_id.flatten().unique().item()

        assert isinstance(image_token_id, int)
        self.img_context_token_id = image_token_id

        if pixel_values_flat is not None:
            expected_h = expected_w = self.config.vision_config.image_size
            resolve_bindings = {"h": expected_h, "w": expected_w}

            return YuanVLImagePixelInputs(
                type="pixel_values",
                pixel_values_flat=self._validate_pixel_values(pixel_values_flat),
                num_patches=image_num_patches,
                resolve_bindings=resolve_bindings,
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self,
        image_input: YuanVLImageInputs,
    ) -> Tuple[torch.Tensor]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_model is not None

        image_embeds = self.extract_feature(image_input["pixel_values_flat"])

        num_patches = image_input["num_patches"]

        # Only one image in the current batch
        if len(num_patches) == 1:
            image_embeds = image_embeds.view(
                    -1, self.text_config.hidden_size).unsqueeze(0)
            return image_embeds

        # NOTE: Image embeddings are split into separate tensors for each image
        # by the size of each embedding.
        # feature_size 每个patch 256个token位置
        feature_size = image_embeds.shape[1]
        image_embeds = image_embeds.view(-1,
                                         self.text_config.hidden_size)
        image_feature_sizes = [num_patches_ * feature_size for num_patches_ in num_patches]
        # 切分后得到一个Tuple，元组每个元组表示一个image的image_embed, [num_patches * 256, text_config.hidden_size]
        image_embeds = image_embeds.split(image_feature_sizes)

        return image_embeds

    def _set_visual_token_mask(self, input_ids: torch.Tensor) -> None:
        if self.is_mono:
            assert self.img_context_token_id is not None
            self.visual_token_mask = (input_ids == self.img_context_token_id).reshape(
                -1, 1
            )
        else:
            self.visual_token_mask = None

    def embed_multimodal(
            self, **kwargs) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        # image_input: (total_patches, 3, h, w)
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
            self._set_visual_token_mask(input_ids)

        # This is to satisfy the type checker for each overload
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ):
        self.language_model.set_eplb_state(
            expert_load_view, logical_to_physical_map, logical_replica_count
        )
        self.expert_weights = self.language_model.expert_weights

    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ):
        self.language_model.update_physical_experts_metadata(
            num_physical_experts, num_local_physical_experts
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        forward_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
        }
        hidden_states = self.language_model(**forward_kwargs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        logits = self.language_model.logits_processor(
            self.language_model.lm_head,
            hidden_states.to(self.language_model.lm_head_dtype),
        )
        return logits

    def load_weights(self, weights: Iterable[tuple[str,torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
