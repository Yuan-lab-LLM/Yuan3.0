# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from megatron.training import get_args, get_tokenizer
import torch.utils.data as data
from megatron.core.parallel_state import get_pipeline_model_parallel_group, get_data_parallel_group
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import math
from tqdm import tqdm
import io
from torchvision import transforms
import numpy as np
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop, ColorJitter, Grayscale

from transformers import AutoModel, CLIPImageProcessor

import pdb
from PIL import Image


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size, threshold):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        size_diff_length = abs(((ratio[0]*image_size + ratio[1]*image_size)-(width+height)) / (width+height))
        if ratio_diff < best_ratio_diff and size_diff_length <= threshold:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


# https://github.com/OpenGVLab/InternVL/blob/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/internvl/train/dataset.py#L830
def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False, threshold=1):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size, threshold)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=9):
    args = get_args()
    with Image.open(image_file) as img:
        image = img.convert('RGB')

    if args.clip_model_name == 'InternViT-448':
        image_processor = CLIPImageProcessor.from_pretrained(args.clip_download_path)

        if args.image_segment_method == 'dynamic':
            images_processed = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num, threshold=args.shape_change_threshold)
        pixel_values = [image_processor(images=image, return_tensors='pt').pixel_values.squeeze(0) for image in images_processed]

    pixel_values = torch.stack(pixel_values)
    
    return pixel_values

class BaseDataset(data.Dataset):
    _repr_indent = 4

    def __init__(
            self,
            data_paths: list,
            samples_per_file: list,
            txt_seq_length: int,
            img_transformer: Optional[Callable] = None,
            txt_transformer: Optional[Callable] = None
    ) -> None:
        self.data_paths = []
        self.samples_per_file = []
        assert (len(data_paths) == len(samples_per_file)) and len(data_paths)>0
        for data_dir, num_samples in zip(data_paths, samples_per_file):
            if isinstance(data_dir, str) or isinstance(data_dir, bytes):
                data_dir = os.path.expanduser(data_dir)
                self.data_paths.append(data_dir)
                self.samples_per_file.append(num_samples)

        self.img_transformer = img_transformer
        self.txt_transformer = txt_transformer
        self.txt_seq_length = txt_seq_length

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self) -> str:
        return ""

def make_dataset(
    data_paths: list,
    samples_per_file: list,
    pairs_per_sample: int,
    txt_seq_length: int,
    npy_full: str,
    npy_single: str
) -> List[Tuple[str, str]]:
    args = get_args()
    total_samples = sum(samples_per_file)
    bin_files = []
    samples_bin_files = []
    samples_idx_files = []
    full_idx_paths = []
    nsamples = []
    train_samples = []
    samples = []
    total_samples = 0
 
    if torch.distributed.get_rank() == 0 and not os.path.exists(npy_full):
        assert args.iteration == 0, 'npy_full must exist when iteration is not 0'
        train_samples = []
        for i, (data_dir, ns) in enumerate(zip(data_paths, samples_per_file)):
            img_txt_bin_path = '{}.{}'.format(data_dir, 'bin')
            img_txt_idx_path = '{}.{}'.format(data_dir, 'idx')
            img_txt_pairs_npy_path = '{}.{}'.format(data_dir, 'npy')
            img_txt_pairs_npy_path = os.path.join(npy_single, img_txt_pairs_npy_path.replace('/', '.'))
            print(img_txt_pairs_npy_path)
            total_samples = []

            with open(img_txt_bin_path, 'rb') as img_txt_bin_file, open(img_txt_idx_path, 'rb') as img_txt_idx_file:
                # 获取每个idx文件中的pair个数
                img_txt_idx_file.seek(0, 2)
                n_pairs = img_txt_idx_file.tell() // 8 - 1
                while True:
                    if len(total_samples)>ns*pairs_per_sample:

                        total_samples = total_samples[:ns]
                        break
                    samples = list(range(n_pairs//pairs_per_sample))
                    random.shuffle(samples)
                    total_samples.extend(samples)
            # 保存每个数据集的npy文件
            total_samples = np.array(total_samples)

            with open(img_txt_pairs_npy_path, 'wb') as f:
                np.save(f, total_samples, allow_pickle=True)
            # 合并samples
            for sample_id in tqdm(range(samples_per_file[i])):
                train_samples.append([i, sample_id])
        random.shuffle(train_samples)
        train_samples = np.array(train_samples)
        print('saving to ', npy_full)
        np.save(npy_full, train_samples, allow_pickle=True)
    counts = torch.cuda.LongTensor([1]) if torch.cuda.device_count() > 0 else torch.LongTensor([1])
    torch.distributed.all_reduce(counts, group=get_data_parallel_group())
    torch.distributed.barrier(group = get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=get_pipeline_model_parallel_group())
    torch.distributed.barrier(group = get_pipeline_model_parallel_group())
    assert os.path.exists(npy_full), f'{npy_full} must exist'
    img_txt_bin_files = []
    img_txt_idx_files = []
    img_txt_pairs_idx_files = []

    for i, (data_dir, ns) in enumerate(zip(data_paths, samples_per_file)):
        img_txt_bin_path = '{}.{}'.format(data_dir, 'bin')
        img_txt_idx_path = '{}.{}'.format(data_dir, 'idx')
        img_txt_pairs_idx_path = '{}.{}'.format(data_dir, 'npy')
        img_txt_pairs_idx_path = os.path.join(npy_single, img_txt_pairs_idx_path.replace('/', '.'))
        img_txt_bin_files.append(img_txt_bin_path)
        img_txt_idx_files.append(img_txt_idx_path)
        img_txt_pairs_idx_files.append(img_txt_pairs_idx_path)
    return img_txt_bin_files, img_txt_idx_files, img_txt_pairs_idx_files


def get_datasets_weights_and_num_samples(data_prefix,
                                         train_num_samples):
    args = get_args()
    assert len(data_prefix) % 2 == 0
    num_datasets = len(data_prefix) // 2
    weights = [0]*num_datasets
    data_path = [0]*num_datasets
    for i in range(num_datasets):
        weights[i] = float(data_prefix[2*i])
        data_path[i] = (data_prefix[2*i+1]).strip()

    # Normalize weights
    weight_sum = sum(weights)
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]

    # Add 0.5% (the 1.005 factor) so in case the bleding dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network.
    datasets_train_num_samples = []
    for weight in weights:
        datasets_train_num_samples.append(int(math.ceil(train_num_samples * weight * 1.005)))
    return data_path, weights, datasets_train_num_samples

def PadImage(image, padcolor):
    width, height = image.size
    if width == height:
        return image
    elif width > height:
        length_image = width
    else:
        length_image = height

    new_iamge = Image.new(image.mode, (length_image, length_image), padcolor)
    new_iamge.paste(image, ((height - width) // 2, 0))
    return new_iamge

def _convert_to_rgb(image):
    return image.convert('RGB')

class ImageTextDataset(BaseDataset):
    def __init__(
            self,
            data_paths: list,
            samples_per_file: list,
            txt_seq_length: int,
            full_idx_path: str,
            img_transformer: Optional[Callable] = None,
            txt_transformer: Optional[Callable] = None,
    ) -> None:
        super(ImageTextDataset, self).__init__(data_paths=data_paths, samples_per_file=samples_per_file, txt_seq_length=txt_seq_length,
                                            img_transformer=img_transformer,
                                            txt_transformer=txt_transformer)
        args = get_args()
        self.tokenizer = get_tokenizer()
        self.samples_per_file = samples_per_file
        self.pairs_per_sample = args.seq_length // txt_seq_length
        self.txt_seq_length = txt_seq_length
        self.npy_full = os.path.join(full_idx_path, 'data.npy')
        self.npy_single = full_idx_path

        self.reset_position_ids = args.reset_position_ids
        self.reset_attention_mask = args.reset_attention_mask
        self.eod_mask_loss = args.eod_mask_loss
        self.clip_visual_size = args.clip_visual_size
        
        self.bos_token, self.image_token, self.eoc_token, self.pad_token = (self.tokenizer(tok)['input_ids'][0] for tok in ['<BOS>','<IMAGE>', '<EOC>', '<pad>'])
        self.eod_token = self.tokenizer("<eod>")['input_ids'][0]
        
        self.max_split_tile_num_single_image = args.max_split_tile_num_single_image
        self.max_split_tile_num_multi_image = args.max_split_tile_num_multi_image
        print('args.max_split_tile_num_single_image:', args.max_split_tile_num_single_image)
        self.data_cut_length = args.data_cut_length
        print('args.data_cut_length:', args.data_cut_length)

        self.img_txt_bin_files, self.img_txt_idx_files, self.img_txt_pairs_idx_files = self.make_dataset(self.data_paths, self.samples_per_file, self.pairs_per_sample, txt_seq_length, self.npy_full, self.npy_single)

        self.img_txt_pairs_idx = []
        self.device = "cpu"

        if args.clip_model_name == 'InternViT-448':
            img_transforms = CLIPImageProcessor.from_pretrained(args.clip_download_path)
        else:
            print('can not support: ', args.clip_model_name)
            exit()

        for fname in self.img_txt_pairs_idx_files:
            try:
                self.img_txt_pairs_idx.append(np.load(fname, allow_pickle=True, mmap_mode='r'))
            except:
                print(fname)
                exit()
        try:
            self.total_idx = np.load(self.npy_full, allow_pickle=True, mmap_mode='r')
        except:
            print(self.npy_full)
            exit()

    @staticmethod
    def make_dataset(
        data_paths: list,
        samples_per_file: list,
        pairs_per_sample: int,
        txt_seq_length: int,
        npy_full: str,
        npy_single: str
    ) -> List[Tuple[str, int]]:
        return make_dataset(data_paths,
                            samples_per_file,
                            pairs_per_sample,
                            txt_seq_length,
                            npy_full,
                            npy_single)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        
        curr_index = index
        for x in range(len(self.total_idx)):
            try:
                fid = self.total_idx[curr_index][0]
                sid = self.total_idx[curr_index][1]

                with open(self.img_txt_bin_files[fid], 'rb') as samples_bin_file, open(self.img_txt_idx_files[fid], 'rb') as samples_idx_file:
                    sid2 = self.img_txt_pairs_idx[fid][sid]
                    samples_idx_file.seek(sid2*8, 0)

                    total_tokens = []
                    total_imgs = []
                    for i in range(self.pairs_per_sample):

                        sample_start = int.from_bytes(samples_idx_file.read(8), 'little')
                        sample_end = int.from_bytes(samples_idx_file.read(8), 'little')
                        sample_length = sample_end - sample_start

                        samples_bin_file.seek(sample_start, 0)
                        data_bytes = samples_bin_file.read(sample_length)
                        data_buffer = io.BytesIO(data_bytes)
                        sample = torch.load(data_buffer)
                        image_tensor = []
                        num_tile_pre_image = []

                        if 'image_per_sample' not in sample:
                            max_num_tile = self.max_split_tile_num_single_image
                        else:
                            max_num_tile = self.max_split_tile_num_multi_image
                        
                        if 'image' in sample:
                            if sample['image'] == None:
                                process_image_flag = False
                            else:
                                process_image_flag = True
                        else:
                            process_image_flag = False

                        if process_image_flag:

                            for image_path_out in sample['image']:
                                if isinstance(image_path_out, list):
                                    assert len(image_path_out) == 1
                                    image_path = image_path_out[0]
                                else:
                                    image_path = image_path_out

                                img = load_image(image_path, max_num=max_num_tile)
                                num_tile = img.shape[0]
                                num_tile_pre_image.append(num_tile)
                                image_tensor.append(img)

                                             
                            image_tensor = torch.cat(image_tensor, dim=0)
                            images = image_tensor.bfloat16()
                            num_tile_pre_image_tensor = torch.Tensor(num_tile_pre_image).long()

                            if 'image_per_sample' not in sample:
                                image_per_sample_tensor = torch.Tensor([1] * len(sample['image'])).long()
                            else:
                                image_per_sample_tensor = torch.Tensor(sample['image_per_sample']).long()
                        else:
                            images = torch.Tensor([0]).to(torch.bfloat16)
                            num_tile_pre_image_tensor = torch.Tensor([0]).long()
                            image_per_sample_tensor = torch.Tensor([0]).long()

                    if 'prompt' in sample:
                        if len(sample["prompt"]) > self.data_cut_length:

                            sample["prompt"] = sample["prompt"][:self.data_cut_length]
                        return {'img' : images, 'text' : torch.tensor(sample['prompt']), 'num_tile' : num_tile_pre_image_tensor, 'image_per_sample' : image_per_sample_tensor}
                    else:
                        return {'img' : images, 'text' : torch.tensor(sample['doc_ids']), 'num_tile' : num_tile_pre_image_tensor, 'image_per_sample' : image_per_sample_tensor}

                break
            except Exception as e:
                print(curr_index, fid, sid)
                print(e)
                print('ERROR_DATA')
                curr_index = 0
    def __len__(self) -> int:
        return len(self.total_idx)

def build_train_datasets(data_path, train_num_samples, txt_seq_length, full_idx_path):
    
    args = get_args()
    assert args.seq_length % txt_seq_length == 0
    # training dataset
    data_paths, train_weights, samples_per_file = get_datasets_weights_and_num_samples(data_path, train_num_samples)
    img_transformer = None
    txt_transformer = None
    
    train_data = ImageTextDataset(
        data_paths = data_paths,
        samples_per_file=samples_per_file,
        txt_seq_length = txt_seq_length,
        full_idx_path = full_idx_path,
        img_transformer = img_transformer,
        txt_transformer = txt_transformer
    )
    return train_data

