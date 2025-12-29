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

import bisect
import json
import torch
from verl.utils.megatron_utils import unwrap_model

from .util import postprocess_packed_seqs, preprocess_packed_seqs, recover_left_padding, remove_left_padding


def gptmodel_forward(
    model,
    input_ids,
    attention_mask,
    position_ids,
    sequence_parallel,
    value_model=False,
    pack_seqs=True,
    logits_processor=None,
    logits_processor_args: dict = None,
    **kwargs,
):
    """Default forward pass for GPT models with optional sequence packing."""
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process
    if pack_seqs:
        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
        )
        if post_process and logits_processor is not None:
            args = {
                k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0]
                for k, v in logits_processor_args.items()
            }
            output_dict = logits_processor(output_orig, **args)
            output = {
                k: postprocess_packed_seqs(
                    v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
                )
                for k, v in output_dict.items()
            }
        else:
            output = postprocess_packed_seqs(
                output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
            )
    else:
        assert logits_processor is None, "logits_processor is not supported for non-packed sequence"
        batch_size, sequence_length = attention_mask.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(
            input_ids, attention_mask, position_ids, sequence_parallel, pre_process=pre_process
        )
        output = model(input_ids=new_input_ids, attention_mask=new_attention_mask, position_ids=new_position_ids)
        output = recover_left_padding(
            output, new_attention_mask, attention_mask, sequence_length, post_process=post_process
        )
    if value_model and post_process:
        output = output[..., 0]
    return output


def gptmodel_forward_qwen2_5_vl(
    model,
    input_ids,
    attention_mask,
    position_ids,
    sequence_parallel,
    value_model=False,
    pack_seqs=True,
    multi_modal_inputs=None,
    logits_processor=None,
    logits_processor_args: dict = None,
    **kwargs,
):
    from megatron.core import parallel_state as mpu

    assert mpu.get_context_parallel_world_size() == 1, "qwen2_5_vl's context parallel is not accurate yet"
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process
    pixel_values = (
        multi_modal_inputs["pixel_values"].to(input_ids.device) if "pixel_values" in multi_modal_inputs else None
    )
    image_grid_thw = (
        multi_modal_inputs["image_grid_thw"].to(input_ids.device) if "image_grid_thw" in multi_modal_inputs else None
    )
    if pack_seqs:
        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        if post_process and logits_processor is not None:
            args = {
                k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0]
                for k, v in logits_processor_args.items()
            }
            output_dict = logits_processor(output_orig, **args)
            output = {
                k: postprocess_packed_seqs(
                    v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
                )
                for k, v in output_dict.items()
            }
        else:
            output = postprocess_packed_seqs(
                output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
            )
    else:
        batch_size, sequence_length = attention_mask.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(
            input_ids, attention_mask, position_ids, sequence_parallel, pre_process=pre_process
        )
        output = model(
            input_ids=new_input_ids,
            position_ids=new_position_ids,
            attention_mask=new_attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        output = recover_left_padding(
            output, new_attention_mask, attention_mask, sequence_length, post_process=post_process
        )
    if value_model and post_process:
        output = output[..., 0]
    return output


def get_ltor_masks_and_position_ids_yuanvl_train(data,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    bos_token,
                                    image_start_token,
                                    image_end_token,
                                    eod_token,
                                    sep_token,
                                    clip_visual_size,
                                    num_tile,
                                    tokenizer,
                                    pre_process,
                                    ):
    """Build masks and position id for left to right model."""
    # Extract batch size and sequence length.
    input_pad_tensor = data
    # 临时修改，避免移位带来的不匹配
    data_pad_tensor = input_pad_tensor

    micro_batch_size, seq_length_pad = data_pad_tensor.size()
    assert micro_batch_size == 1, 'now support mbs=1 only'
    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.zeros(
        (att_mask_batch, 1, 1), device=data.device).view(
            att_mask_batch, 1, 1, 1)

    if num_tile[0][0] != 0:
        # Position ids.
        position_ids = torch.arange(seq_length_pad, dtype=torch.long,
                                    device=data_pad_tensor.device)
        position_ids = position_ids.unsqueeze(0).expand_as(data_pad_tensor)

        position_ids_use = torch.zeros(data_pad_tensor.shape).to(position_ids)
        # We need to clone as the ids will be modifed based on batch index.
        if reset_position_ids:
            position_ids = position_ids.clone()

        # Loss mask.
        image_info_pad = {}

        for b in range(micro_batch_size):
            bos_index = position_ids[b, data_pad_tensor[b] == bos_token]
            sep_index = position_ids[b, data_pad_tensor[b] == sep_token]
            image_start_index = position_ids[b, data_pad_tensor[b] == image_start_token]
            image_end_index = position_ids[b, data_pad_tensor[b] == image_end_token]

            # 查找<|Assistant|>之后的<IMAGE>的索引，将<|Assistant|>之后的<IMAGE>的索引删除
            first_sep_index = sep_index[0]
            for pos_, index_ in enumerate(image_start_index.clone()):
                if index_ >= first_sep_index:
                    image_start_index = image_start_index[:pos_]
                    image_end_index = image_end_index[:pos_]
                    break
            image_end_index = image_end_index[:len(image_start_index)] #防止出现image_end_index过多的情况

            for idx in range(len(image_start_index)):
                check_num_tile = (image_end_index[idx] - image_start_index[idx]) // clip_visual_size
                assert check_num_tile == num_tile[b][idx], f'{check_num_tile} not eq {num_tile[b][idx]} {image_start_index}'

            if not len(image_start_index) == len(image_end_index) and pre_process:
                print(data_pad_tensor.tolist())
                print(tokenizer.decode(data_pad_tensor[0].tolist()))
                print(image_start_index.tolist(), num_tile[b])
                print('check mllm data, len(image_start_index) should equal len(image_end_index)')
                exit()

            image_info_pad['image_start_pos'] = image_start_index.tolist()
            image_info_pad['num_tile'] = num_tile[b]
            if not len(bos_index) == len(sep_index) and pre_process:
                print(data_pad_tensor.tolist())
                print(tokenizer.decode(data_pad_tensor[0].tolist()))


        if reset_position_ids or reset_attention_mask:
            # Loop through the batches:
            for b in range(micro_batch_size):

                # Find indecies where EOD token is.
                bos_index = position_ids[b, data_pad_tensor[b] == bos_token]
                image_end_index = position_ids[b, data_pad_tensor[b] == image_end_token]
                # Detach indecies from positions if going to modify positions.
                if reset_position_ids:
                    bos_index = bos_index.clone()
                    image_end_index = image_end_index.clone()
                # Loop through EOD indecies:
                prev_index = 0

                # TODO 确定attention_mask是否使用
                start_idx = image_end_index[-1] 
                end_idx = seq_length_pad - 1
                diff = end_idx - start_idx + 1

                position_ids_use[b][start_idx : end_idx+1] = torch.arange(diff, dtype=torch.long,
                                                                device=data_pad_tensor.device)
    else:
        # Position ids.
        position_ids = torch.arange(seq_length_pad, dtype=torch.long,
                                    device=data_pad_tensor.device)
        position_ids = position_ids.unsqueeze(0).expand_as(data_pad_tensor)

        # We need to clone as the ids will be modifed based on batch index.
        if reset_position_ids:
            position_ids = position_ids.clone()

        if reset_position_ids or reset_attention_mask:
            # Loop through the batches:
            for b in range(micro_batch_size):

                # Find indecies where EOD token is.
                eod_index = position_ids[b, data_pad_tensor[b] == eod_token]
                # Detach indecies from positions if going to modify positions.
                if reset_position_ids:
                    eod_index = eod_index.clone()

                # Loop through EOD indecies:
                prev_index = 0
                for j in range(eod_index.size()[0]):
                    i = eod_index[j]
                    if reset_position_ids:
                        position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                        prev_index = i + 1
        position_ids_use = position_ids
        assert position_ids_use[0][-1] == len(data_pad_tensor[0]) - 1, "check posid"
        image_info_pad = None
    attention_mask = (attention_mask < 0.5)
    return attention_mask, position_ids_use, data_pad_tensor, image_info_pad


def gptmodel_forward_yuanvl(
    model,
    input_ids,
    attention_mask,
    position_ids,
    sequence_parallel,
    multi_modal_inputs=None,
    value_model=False,
    pack_seqs=True,
    logits_processor=None,
    logits_processor_args: dict = None,
    tokenizer=None,
    processed_images=None,
    extra_info_list=None,
    **kwargs,
):
    """Default forward pass for GPT models with optional sequence packing."""
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process
    compute_probs_in_model = unwrap_model(model).compute_probs_in_model
    assert pack_seqs, "now only support pack_seqs==True, but not use packed_seq_params"
    if pack_seqs:
        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
        input_ids_rmpad = input_ids_rmpad.contiguous()

        bos_token, image_start_token, image_end_token, pad_token, sep_token, eod_token = (tokenizer(tok)['input_ids'][0] for tok in ['<BOS>','<IMAGE>', '</IMAGE>', '<pad>', '<|Assistant|>', '<eod>'])
        
        if processed_images is not None and processed_images[0] is not None:
            num_tile = [extra_info_list[0]['num_tiles']]
            all_time_num, _, _, _ = processed_images[0].shape
            assert torch.tensor(num_tile).sum() == all_time_num, f"强制检查，num_tile({num_tile})的总数应该等于processed_images({processed_images.shape})总的图片数"
            processed_images = processed_images.to(torch.bfloat16)
        else:
            num_tile = [[0]]
            if 'image_path' in extra_info_list[0]:
                assert extra_info_list[0]['image_path'] == None, f"强制检查，防止出现有图像但是未传递图像的情况"
        attention_mask_m, position_ids_m, data_pad_tensor, image_info = get_ltor_masks_and_position_ids_yuanvl_train(
            input_ids_rmpad,
            True, 
            True,
            bos_token=bos_token,
            image_start_token=image_start_token,
            image_end_token=image_end_token,
            eod_token=eod_token,
            sep_token=sep_token,
            clip_visual_size=256,
            num_tile=num_tile,
            tokenizer=tokenizer,
            pre_process=pre_process
        )
        if post_process and logits_processor is not None:
            args = {
                k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0]
                for k, v in logits_processor_args.items()
            }
            if compute_probs_in_model:
                labels = args["label"]
                loss_mask = args["label_mask"]
                mbs, seq_length_input = labels.shape
                assert mbs == 1
                output_orig = model(
                    input_ids=data_pad_tensor,
                    attention_mask=attention_mask_m,
                    position_ids=position_ids_m,
                    labels=labels,
                    loss_mask=loss_mask,
                    image_tensor=processed_images,
                    image_info=image_info,
                    logits_processor=logits_processor,
                )
                _, seq_length_output = output_orig.shape
                output_dict = {}
                if seq_length_output == seq_length_input:
                    log_probs = output_orig
                    output_dict['log_probs'] = log_probs
                else:
                    assert seq_length_output == 2 * seq_length_input, 'check hare, seq_length_output should equal (2 * seq_length_input)'
                    entropy, log_probs = torch.split(output_orig, seq_length_input, dim=-1)
                    output_dict['entropy'] = entropy
                    output_dict['log_probs'] = log_probs
            else:
                output_orig = model(
                    input_ids=data_pad_tensor,
                    attention_mask=attention_mask_m,
                    position_ids=position_ids_m,
                    image_tensor=processed_images,
                    image_info=image_info,
                    )
                output_dict = logits_processor(output_orig, **args)

            output = {
                k: postprocess_packed_seqs(
                    v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
                )
                for k, v in output_dict.items()
            }
        else:
            output_orig = model(
                input_ids=data_pad_tensor,
                attention_mask=attention_mask_m,
                position_ids=position_ids_m,
                image_tensor=processed_images,
                image_info=image_info,
                )
            output = postprocess_packed_seqs(
                output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
            )

    else:
        assert logits_processor is None, "logits_processor is not supported for non-packed sequence"
        batch_size, sequence_length = attention_mask.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(
            input_ids, attention_mask, position_ids, sequence_parallel, pre_process=pre_process
        )
        # 未验证
        bos_token, image_start_token, image_end_token, pad_token, sep_token, eod_token = (tokenizer(tok)['input_ids'][0] for tok in ['<BOS>','<IMAGE>', '</IMAGE>', '<pad>', '<|Assistant|>', '<eod>'])
        num_tile = [[0]]

        attention_mask_m, loss_mask_m, position_ids_m, data_pad_tensor, label_pad_tensor, image_info = get_ltor_masks_and_position_ids_yuanvl_train(
            new_input_ids,
            True,
            True,
            True, 
            bos_token=bos_token,
            image_start_token=image_start_token,
            image_end_token=image_end_token,
            eod_token=eod_token,
            pad_token=pad_token,
            sep_token=sep_token,
            clip_visual_size=1024,
            num_tile=num_tile,
            image_per_sample=None)

        output = model(input_ids=new_input_ids, attention_mask=attention_mask_m, position_ids=position_ids_m)
        output = recover_left_padding(
            output, new_attention_mask, attention_mask, sequence_length, post_process=post_process
        )
    if value_model and post_process:
        output = output[..., 0]
    return output
