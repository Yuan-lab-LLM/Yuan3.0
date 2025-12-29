# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import os
import re
from collections import defaultdict
from typing import Optional

import datasets
from datasets import Features, Value, List
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from torchvision.transforms import ToTensor

import verl.utils.torch_functional as verl_F
from verl.utils.model import (
        compute_position_id_with_mask,
        compute_yuanvl_position_id_with_mask,
)

response_length_long_mapping = {
    'llm_math':16*1024,
    'llm_code':16*1024,
    'llm_choice':16*1024,
    'sql_bird':16*1024,
    'sql_spider':16*1024,
    'mllm_normal':16*1024,
    'mllm_choice':16*1024,
}

response_length_short_mapping = {
    'llm_math':4*1024,
    'llm_code':8*1024,
    'llm_choice':4*1024,
    'sql_bird':4*1024,
    'sql_spider':4*1024,
    'rag_chatqa':4*1024,
    'rag_mmtab':4*1024,
    'rag_summary':4*1024,
    'rag_ifeval':4*1024,
    'rag_frames':4*1024,
    'rag_alpaca':4*1024,
    'rag_simpleqa':4*1024,
    'llm_general':4*1024,
    'llm_tool_reward':4*1024,
    'mllm_normal':4*1024,
    'mllm_choice':4*1024,
    'mllm_knowledge':4*1024,
    'mllm_grounding':4*1024,
    'mllm_general1':4*1024,
    'mllm_general2':4*1024,
    'rag_mrag':4*1024,
    'rag_mrag_bench':4*1024,
    'rag_mmtab':4*1024
}

LLM_METHOD_FLAG = ['llm_math','llm_code','llm_choice','llm_general','rag_ifeval','rag_frames','rag_alpaca','rag_simpleqa','rag_chatqa']

logger = logging.getLogger(__name__)

def find_closest_aspect_ratio(
    aspect_ratio,
    target_ratios,
    width,
    height,
    image_size=448,
    threshold=1,
):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        
        # 拆分长计算表达式，提升可读性
        size_diff_length = abs(
            (
                (ratio[0] * image_size + ratio[1] * image_size) 
                - (width + height)
            ) / (width + height)
        )
        
        # 长条件拆分到多行，保持逻辑清晰
        if (
            ratio_diff < best_ratio_diff 
            and size_diff_length <= threshold
        ):
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    
    return best_ratio


def dynamic_preprocess(
    image,
    min_num=1,
    max_num=9,
    image_size=448,
    use_thumbnail=True,
    threshold=1
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) 
        for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) 
        for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
        threshold
    )
    
    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    assert len(processed_images) == blocks
    
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        assert self.processor.__class__.__name__ in ['CLIPImageProcessor', 'CLIPImageProcessorFast'], "check image processor"
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        self.num_tokens_per_image = int(config.clip_visual_size * config.downsample_ratio**2)
        self.max_split_tile_num = config.max_split_tile_num
        self.image_start_token = config.image_start_token
        self.image_end_token = config.image_end_token

        self.enable_thinking_flag = config.enable_thinking_flag
        self._download()
        self.pad_token_id = self.tokenizer(config.pad_token)['input_ids'][0]
        self.bos_token = config.bos_token
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        total_columns = set()
        for parquet_file in self.data_files:
            # read parquet files and cache
            try:
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file, cache_dir="~/.cache")["train"]
                dataframes.append(dataframe)
                total_columns = total_columns | { col for col in dataframe.column_names}
            except Exception as e:
                print(f"读取 parquet 文件失败: {e}")
                exit()
        for i, dataframe in enumerate(dataframes):
            missing_columns = [col for col in total_columns if col not in dataframe.column_names]
            # 以下操作只针对yuanvl的混合数据集
            if len(missing_columns) > 0:
                target_features = Features({
                    # 保留双方共有的列（类型需一致）
                    "prompt": List({
                        "content": Value("string"),
                        "role": Value("string")
                    }),
                    "reward_method": Value("string"),
                    "data_source": Value("string"),
                    "ability": Value("string"),
                    "reward_model": {
                        "ground_truth": Value("string"),
                        "style": Value("string")
                    },
                    
                    # 统一 extra_info 结构（包含 image_path，以数据集2为基准）
                    "extra_info": {
                        "answer": Value("string"),
                        "enable_thinking_flag": Value("bool"),
                        "expect_len": Value("float64"),
                        "image_path": Value("string"),  # 数据集1需补充此字段
                        "index": Value("int64"),
                        "question": Value("string"),
                        "split": Value("string"),
                        "db_id": Value("string"),
                        "question_id": Value("int64"),
                        "table": Value("string"),
                        "yuan_input": Value("string")
                    },
                    
                    # 补充双方独有的列
                    "language": Value("string"),  # 数据集2需补充此字段
                    "images": List({       # 数据集1需补充此字段
                        "bytes": Value("binary"),
                        "path": Value("string")
                    })
                })
                dataframe = dataframe.cast(Features({
                    **dataframe.features,  # 保留原有字段
                    "extra_info": target_features["extra_info"]  # 更新 extra_info 结构
                }))
                if 'images' not in dataframe.features:
                    dataframe = dataframe.add_column(
                        "images", 
                        [[] for _ in range(len(dataframe))]  # 每个样本添加空列表（符合 List 类型）
                    )
                if 'language' not in dataframe.features:
                    dataframe = dataframe.add_column(
                        "language", 
                        [""] * len(dataframe)  # 每个样本添加空字符串
                    )
                dataframe = dataframe.cast(target_features)
                dataframe = dataframe.select_columns(list(target_features.keys()))
                dataframes[i] = dataframe
        for dataframe in dataframes:
            print(dataframe, flush=True)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    if self.processor.__class__.__name__ in ['CLIPImageProcessor', 'CLIPImageProcessorFast']:
                        if image_key in doc and len(doc[image_key]) > 0:
                            texts = []
                            for message in messages:
                                if isinstance(message['content'], list):
                                    for i in range(len(message['content'])):
                                        if message['content'][i]['type'] == 'text':
                                            text = [{'content': message['content'][i]['text'], 'role': "user"}]
                                            texts.append(text)
                                else:
                                    texts.append(message)
                            assert len(texts) == 1
                            # 应用chat template，格式化对话
                            raw_prompt = self.tokenizer.apply_chat_template(
                                texts, add_generation_prompt=True, tokenize=False
                            )[0]
                            # 针对多模态数据，增加 <BOS><IMAGE><pad>......</IMAGE>
                            # 多模态数据的过滤，需要考虑图片部分的占位
                            # 通过dynamic_preprocess，实现对图片的切分
                            num_patches = 0
                            for image in doc[image_key]:
                                image_tensor = process_image(image)
                                num_patches += len(dynamic_preprocess(image_tensor))
                            raw_prompt = (
                                self.bos_token 
                                + self.image_start_token 
                                + '<pad>' * self.num_tokens_per_image * num_patches 
                                + self.image_end_token 
                                + raw_prompt
                            )
                            # tokenizer.encode 只负责对文本进行编码，不会应用chat template
                        else:
                            raw_prompt = self.tokenizer.apply_chat_template(
                                messages, add_generation_prompt=True, tokenize=False
                            )
                        raw_prompt_ids = tokenizer.encode(raw_prompt)
                        return len(raw_prompt_ids)
                    else:
                        # 非yuanvl processor，保持原有的处理方式
                        raw_prompt = self.processor.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False
                        )
                        images = [process_image(image) for image in doc[image_key]] if image_key in doc else None
                        videos = [process_video(video) for video in doc[video_key]] if video_key in doc else None
                        return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])
            else:
                def doc2len(doc) -> int:
                    return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                if example['reward_method'] in LLM_METHOD_FLAG and ('<image>' in content or '<video>' in content):
                    print(content)
                    content = content.replace('<image>','').replace('<video>','')

                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        if 'images' not in row_dict:
            row_dict['images'] = []
        if self.processor.__class__.__name__ in ['CLIPImageProcessor', 'CLIPImageProcessorFast']:
            messages = row_dict[self.prompt_key]
            for i, message in enumerate(messages):
                pattern = r'^(?:<image>)+'
                messages[i]['content'] = re.sub(pattern, '', messages[i]['content'])
        else:
            messages = self._build_messages(row_dict)
        model_inputs = {}
        enable_thinking_flag=row_dict["extra_info"]["enable_thinking_flag"]# self.enable_thinking_flag
        expect_len=int(row_dict["extra_info"]["expect_len"])
        processed_images = None
        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video
            # 输入到 rollout 的 prompts
            if self.processor.__class__.__name__ in ['CLIPImageProcessor', 'CLIPImageProcessorFast']:
                if self.image_key in row_dict:
                    raw_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                        enable_thinking=enable_thinking_flag
                    )
                else:
                    raise NotImplementedError("This function is not implemented yet.")
            else:
                raise NotImplementedError("This function is not implemented yet.")

            multi_modal_data = {}

            images = None
            processed_images = None
            if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
                if self.processor.__class__.__name__ in ['CLIPImageProcessor', 'CLIPImageProcessorFast']:
                    # images: 数据样本中的多个原始图片
                    if len(row_dict[self.image_key]) > 0:
                        processed_images = []
                        for image in row_dict[self.image_key]:
                            orig_image_tensor = process_image(image)
                            image_tensor = dynamic_preprocess(orig_image_tensor.convert("RGB"))
                            pixel_values = [self.processor(images=image, return_tensors='pt').pixel_values.squeeze(0) for image in image_tensor]
                            pixel_values = torch.stack(pixel_values)
                            processed_images.append(pixel_values)
                        images = [process_image(image) for image in row_dict.pop(self.image_key)]
                    else:
                        row_dict.pop(self.image_key)
                else:
                    images = [process_image(image) for image in row_dict.pop(self.image_key)]
                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict and row_dict.get(self.video_key, None) is not None:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            # 用于 actor/ref 模型计算log probs和训练使用的input ids
            # 需要注意，yuanvl多模态模型插入<BOS><IMAGE><pad>......</IMAGE>
            if self.processor.__class__.__name__ in ['CLIPImageProcessor', 'CLIPImageProcessorFast']:
                if 'image' in multi_modal_data:
                    template_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                        enable_thinking=enable_thinking_flag,
                    )
                    if processed_images is not None:
                        template_prefix = self.bos_token
                        for img_tensor in processed_images:
                            num_patches = len(img_tensor)
                            template_prefix = (
                                template_prefix
                                + self.image_start_token 
                                + '<pad>' * self.num_tokens_per_image * num_patches 
                                + self.image_end_token 
                            )
                        template_prompt = template_prefix + template_prompt
                else:
                    raise NotImplementedError("This function is not implemented yet.")
                try:
                    model_inputs = self.tokenizer([template_prompt], return_tensors='pt')
                except Exception as e:
                    print(f'messages: {messages}', flush=True)
                    print(f'model_inputs: {model_inputs}', flush=True)
                    print(f'template_prompt: {template_prompt}', flush=True)
                    exit()
            else:
                model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
            if processed_images is not None:
                num_tiles = [img.shape[0] for img in processed_images]
                processed_images = torch.cat(processed_images)
            else:
                num_tiles = None
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
        # 对 prompt ids 进行了补 <pad> 的操作
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        # 正常情况下 raw_prompt_ids 为 raw_prompt 的 input_ids
        # raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        if self.processor is not None and self.processor.__class__.__name__ in ['CLIPImageProcessor', 'CLIPImageProcessorFast']:
            # yuanvl 多模态数据，在输入 rollout 之前，在 raw_prompt 前面插入 <image> 用于图片占位
            if 'image' in multi_modal_data:
                if multi_modal_data['image'] is not None:
                    raw_prompt = '<image>' * len(multi_modal_data["image"]) + raw_prompt
            position_ids = compute_position_id_with_mask(attention_mask)
            raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        elif self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")
        
        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # 根据reward method进行不同推理长度的设置
        reward_method = row_dict["reward_method"]
        
        if enable_thinking_flag:
            assert reward_method in response_length_long_mapping, f"reward_method:{reward_method} not in response_length_long_mapping:{response_length_long_mapping.keys()}"
            request_response_length = response_length_long_mapping[reward_method]
        else:
            assert reward_method in response_length_short_mapping, f"reward_method:{reward_method} not in response_length_short_mapping:{response_length_short_mapping.keys()}"
            request_response_length = response_length_short_mapping[reward_method]
        
        row_dict['request_response_length'] = request_response_length
        row_dict['expect_len'] = expect_len

        if expect_len < 500 and enable_thinking_flag:
           row_dict['request_response_length'] = 1024

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        if num_tiles is not None:
            row_dict["extra_info"]['num_tiles'] = num_tiles
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        if processed_images is not None:
            row_dict["processed_images"] = [processed_images]
        else:
            row_dict["processed_images"] = None
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
