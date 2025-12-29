#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import math
import os
import json
import torch
import pdb
import copy
from tqdm import tqdm
import safetensors
from safetensors.torch import load_file
import sentencepiece as spm
from transformers import LlamaTokenizer, AutoTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers.tokenization_utils_base import AddedToken
def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--orig-ckpt-path', type=str, required=True)
    parser.add_argument('--vit-ckpt-path', type=str, required=True)
    parser.add_argument('--tokenizer-path', type=str, required=True)
    parser.add_argument('--output-ckpt-path', type=str, required=True)
    parser.add_argument('--pipeline-parallel-size', type=int, default=1)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--expert-parallel-size', type=int, default=1)
    parser.add_argument('--num-layer', type=int, default=1)
    parser.add_argument('--hidden-size', type=int, default=1)
    parser.add_argument('--ffn-hidden-size', type=int, default=1)
    parser.add_argument('--per-layer-experts-blocks', type=str, default=None)
    parser.add_argument('--num-attention-heads', type=int, required=True)
    parser.add_argument('--kv-channels', type=int, required=True)
    parser.add_argument('--pipeline-model-parallel-blocks', type=str, default=None)
    parser.add_argument('--pp-ranks', type=str, default=None)

    args = parser.parse_args()

    return args

def gen_model_state_dict(key, layer_offset):
    if 'embedding.word_embeddings.weight' == key:
        return "language_model.model.embed_tokens.weight"
    elif 'imagemlp.linear_fc2.weight' in key:
        return 'imagemlp.down_proj.weight'
    elif 'imagemlp.linear_fc1.weight' in key:
        return key
    elif 'imagemlp_layernorm' in key:
        return 'imagemlp_layernorm.weight'
    # elif 'attention_router' in key:
    #     return key.replace('attention_router', 'query_key_value').replace('decoder', 'language_model.model')
    elif 'decoder.layers.' in key:
        key_list = key.split('.')
        key_list[0] = 'model'
        layer_id = int(key_list[2]) + layer_offset
        key_list[2] = str(layer_id)
        if 'attention_router' in key:
            key_list[5] = 'query_key_value'
        elif 'linear_qk' in key:
            key_list[4] = 'get_query_key'
        elif 'linear_proj' in key:
            key_list[4] = 'o_proj'
        elif 'linear_v' in key:
            key_list[4] = 'v_proj'
        elif 'pre_mlp_layernorm' in key:
            key_list[3] = 'post_attention_layernorm'
        new_key = 'language_model.' + '.'.join(key_list)
        new_key = new_key.replace('self_attention', 'self_attn')
        return new_key
    elif 'output_layer' in key:
        return 'language_model.lm_head.weight'
    elif 'final_layernorm' in key:
        return 'language_model.model.norm.weight'
    else:
        print(key)
        exit()
    

def main():
    args = parse_args()
    # 转换tokenizer
    # 加载预训练的tokenizer
    spm_model_path = os.path.join(args.tokenizer_path, 'tokenizer.model')
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
    # 添加自定义的特殊标记
    '''
    special_tokens_list = ['<sep>', '<pad>','<mask>','<predict>','<FIM_SUFFIX>','<FIM_PREFIX>','<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>', '<repo_name>', '<file_sep>', '<BOS>', '<IMAGE>', '</IMAGE>', '<grounding>','<obj>','</obj>','<box>','</box>','<point>','</point>','<3dbox>','</3dbox>','<depth>','</depth>']
    
    for i in range(1000):
        num = f"{i:03}"
        num_token = 's' + str(num)
        special_tokens_list.append(num_token)
    
    special_tokens_list.append('<eog>')
    special_tokens_list.append('</eog>')
    '''
    special_tokens_list = ['<sep>', '<pad>','<mask>','<predict>','<FIM_SUFFIX>','<FIM_PREFIX>','<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>', '<repo_name>', '<file_sep>', '<BOS>', '<IMAGE>', '</IMAGE>', '<grounding>','<obj>','</obj>','<box>','</box>','<point>','</point>','<3dbox>','</3dbox>','<depth>','</depth>']

    for i in range(1000):
        num = f"{i:03}"
        num_token = 's' + str(num)
        special_tokens_list.append(num_token)

    special_tokens_list.extend(['<eop>', '<eog>'])
    special_tokens_list.extend(['<|begin_of_sentence|>', '<|end_of_sentence|>', '<|User|>', '<|Assistant|>', '<think>', '</think>', '<search_result>', '</search_result>', '<search_query>', '</search_query>', '<code_query>', '</code_query>', '<code_result>', '</code_result>', '<infer>', '</infer>', '<inferresult>', '</inferresult>', "<tool_calls>", "</tool_calls>", "<tool_response>", "</tool_response>", "<final_answer>", "</final_answer>"])

    # 构建 新的tokenizer.model
    from sentencepiece import sentencepiece_model_pb2
    new_sp_model = spm.SentencePieceProcessor()
    new_sp_model.Load(spm_model_path)

    yuan_spm = sp_pb2_model.ModelProto()
    yuan_spm.ParseFromString(tokenizer.sp_model.serialized_model_proto())
    new_spm = sp_pb2_model.ModelProto()
    new_spm.ParseFromString(new_sp_model.serialized_model_proto())

    yuan_spm_tokens_set = set(p.piece for p in yuan_spm.pieces)
    # for p in new_spm.pieces:
    for piece in special_tokens_list:
        if piece not in yuan_spm_tokens_set:
            print(piece)
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            yuan_spm.pieces.append(new_p)
    score_max = 0
    for p in yuan_spm.pieces:
        score_max = max(score_max, p.score)
    # 保存临时的sp_model
    with open('./custom.model', 'wb') as f:
        f.write(yuan_spm.SerializeToString())
    # 创建新的 tokenizer
    new_tokenizer = LlamaTokenizer(vocab_file='./custom.model')
    # 更新tokenizer.model
    tokenizer.sp_model = new_tokenizer.sp_model
    # 更新special tokens list
    # for token in special_tokens_list:
    #     tokenizer.special_tokens_map[token] = token
    # tokenizer.additional_special_tokens = tokenizer.convert_ids_to_tokens(tokenizer.all_special_ids) + special_tokens_list
    # 更新其他config参数
    tokenizer.add_special_tokens({"additional_special_tokens": tokenizer.convert_ids_to_tokens(tokenizer.all_special_ids) + special_tokens_list})
    tokenizer._added_tokens_decoder = {}
    tokenizer.add_prefix_space = False
    tokenizer.clean_up_tokenization_spaces = True
    exit()
    tokenizer.chat_template = "{%- if messages[0]['role'] == 'system' -%}{%- set system_message = messages[0]['content'] -%}{%- set messages = messages[1:] -%}{%- else -%}{% set system_message = '' -%}{%- endif -%}{%- for message in messages -%}{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{%- endif -%}{%- if message['role'] == 'user' -%}{{ message['content'] }}{%- endif -%}{%- endfor -%}"
    # 保存更新后的tokenizer
    tokenizer.save_pretrained(args.output_ckpt_path)

    # 开始处理模型文件
    args.pipeline_model_parallel_blocks = [ int(s) for s in args.pipeline_model_parallel_blocks.split(',')]
    args.per_layer_experts_blocks = [ int(s) for s in args.per_layer_experts_blocks.split(',')]
    # 加载vit ckpt
    print('加载 vision model')
    vit_state_dict = load_file(os.path.join(args.vit_ckpt_path, 'model.safetensors'), device='cpu')
    total_params = 0
    tmp_params = 0
    vision_model = {}
    bin_config_json = {}
    for key in vit_state_dict.keys():
        total_params += vit_state_dict[key].numel()
        tmp_params += vit_state_dict[key].numel()
        vision_model['vision_model.' + key] = vit_state_dict[key]
    with open(os.path.join(args.orig_ckpt_path, 'latest_checkpointed_iteration.txt'), 'r') as f:
        iteration = int(f.readline().strip())

    root_path = os.path.join(args.orig_ckpt_path, 'iter_{:07d}'.format(iteration))
    output_root_path = os.path.join(args.output_ckpt_path, 'iter_{:07d}'.format(iteration))
    
    if args.pp_ranks == None:
        args.pp_ranks = list(range(args.pipeline_parallel_size))
    else:
        if ',' in args.pp_ranks:
            args.pp_ranks = [int(i) for i in args.pp_ranks.split(',')]
        else:
            args.pp_ranks = [int(args.pp_ranks)]
    max_elements = 5e9
    
    print('统计 language model 参数')
    total_device = 1
    for ep_rank in range(args.expert_parallel_size):
        for tp_rank in range(args.tensor_parallel_size):
            for pp_rank in tqdm(args.pp_ranks, total=len(args.pp_ranks)):
                output_param_ckpt_path = os.path.join(root_path, 'mp_rank_{:02d}_{:03d}'.format(tp_rank, pp_rank), 'model_optim_rng.pt')
                model_state_dict = torch.load(output_param_ckpt_path, map_location='cpu')['model']
                layer_offset = sum(args.pipeline_model_parallel_blocks[:pp_rank])
                for key in model_state_dict.keys():
                    if 'extra_state' in key:
                        continue
                    else:
                        print(key, model_state_dict[key].shape)
                        total_params += model_state_dict[key].numel()
                        tmp_params += model_state_dict[key].numel()
                        if tmp_params >= max_elements:
                            total_device += 1
                            tmp_params = 0
                            
    print(total_device)
    print(total_params)
    bin_config_json['metadata'] = {"total_size": total_params}
    bin_config_json['weight_map'] = {}
    device_map_json = {}
    tmp_params = 0
    tmp_state_dict = {}
    current_device = 1
    for key in vision_model.keys():
        bin_config_json['weight_map'][key] = 'model-{:05d}-of-{:05d}.safetensors'.format(current_device, total_device)
        device_map_json[key] = 0
        print(key, vision_model[key].shape)
        tmp_state_dict[key] = vision_model[key]
        tmp_params += tmp_state_dict[key].numel()

    for ep_rank in range(args.expert_parallel_size):
        for tp_rank in range(args.tensor_parallel_size):
            for pp_rank in tqdm(args.pp_ranks):
                output_param_ckpt_path = os.path.join(root_path, 'mp_rank_{:02d}_{:03d}'.format(tp_rank, pp_rank), 'model_optim_rng.pt')
                model_state_dict = torch.load(output_param_ckpt_path, map_location='cpu')['model']
                layer_offset = sum(args.pipeline_model_parallel_blocks[:pp_rank])
                for key in model_state_dict.keys():
                    if 'extra_state' in key:
                        continue
                    else:
                        new_key = gen_model_state_dict(key, layer_offset)
                        if 'mlp.experts' in key:
                            assert 'weight1' in key or 'weight2' in key
                            if 'weight1' in key:
                                key_list = key.split('.')
                                layer_id = layer_offset + int(key_list[2])
                                key_list[0] = 'model'
                                key_list[2] = str(layer_id)
                                key_list.insert(5, 'w1')
                                key_list.insert(6, 0)
                                key_list[-1] = 'weight'
                                experts = model_state_dict[key].view(args.per_layer_experts_blocks[layer_id], args.hidden_size, -1)
                                for i in range(args.per_layer_experts_blocks[layer_id]):
                                    key_list[6] = str(i)
                                    new_key = 'language_model.' + '.'.join(key_list)
                                    tmp_tensor = experts[i].detach().clone().t().contiguous()
                                    tmp_state_dict[new_key] = tmp_tensor
                                    bin_config_json['weight_map'][new_key] = 'model-{:05d}-of-{:05d}.safetensors'.format(current_device, total_device)
                                    device_map_json[new_key] = current_device // 4

                            elif 'weight2' in key:
                                key_list = key.split('.')
                                layer_id = layer_offset + int(key_list[2])
                                key_list[0] = 'model'
                                key_list[2] = str(layer_id)
                                key_list.insert(5, 'w2')
                                key_list.insert(6, 0)
                                key_list[-1] = 'weight'
                                experts = model_state_dict[key].view(args.per_layer_experts_blocks[layer_id], -1, args.hidden_size)
                                for i in range(args.per_layer_experts_blocks[layer_id]):
                                    key_list[6] = str(i)
                                    new_key = 'language_model.' + '.'.join(key_list)
                                    tmp_tensor = experts[i].detach().clone().t().contiguous()
                                    tmp_state_dict[new_key] = tmp_tensor
                                    bin_config_json['weight_map'][new_key] = 'model-{:05d}-of-{:05d}.safetensors'.format(current_device, total_device)
                                    device_map_json[new_key] = current_device // 4
                        elif 'imagemlp.linear_fc1.weight' == new_key:
                            tmp_params += model_state_dict[key].numel()
                            up_proj, gate_proj = torch.chunk(model_state_dict[key], 2, 0)
                            tmp_key = 'imagemlp.up_proj.weight'
                            tmp_state_dict[tmp_key] = up_proj
                            bin_config_json['weight_map'][tmp_key] = 'model-{:05d}-of-{:05d}.safetensors'.format(current_device, total_device)

                            tmp_key = 'imagemlp.gate_proj.weight'
                            tmp_state_dict[tmp_key] = gate_proj
                            bin_config_json['weight_map'][tmp_key] = 'model-{:05d}-of-{:05d}.safetensors'.format(current_device, total_device)
                            continue
                        else:
                            tmp_state_dict[new_key] = model_state_dict[key]
                            bin_config_json['weight_map'][new_key] = 'model-{:05d}-of-{:05d}.safetensors'.format(current_device, total_device)
                            device_map_json[new_key] = current_device // 4
                        tmp_params += model_state_dict[key].numel()
                        if tmp_params >= max_elements:
                            output_path = os.path.join(args.output_ckpt_path, 'model-{:05d}-of-{:05d}.safetensors'.format(current_device, total_device))
                            safetensors.torch.save_file(tmp_state_dict, output_path, metadata={"format": "pt"})
                            current_device += 1
                            tmp_state_dict = {}
                            tmp_params = 0
    output_path = os.path.join(args.output_ckpt_path, 'model-{:05d}-of-{:05d}.safetensors'.format(current_device, total_device))
                # if pp_rank == 0:
                #     for key in vision_model.keys():
                #         tmp_state_dict[key] = vision_model[key]
    safetensors.torch.save_file(tmp_state_dict, output_path, metadata={"format": "pt"})
                     

    print(f'{root_path} check success') 
    bin_config_json_path = os.path.join(args.output_ckpt_path, 'model.safetensors.index.json')
    with open(bin_config_json_path, 'w') as f:
        data = json.dumps(bin_config_json, ensure_ascii=False, indent=4)
        f.write(data)

    device_map_path = os.path.join(args.output_ckpt_path, 'device_map.json')
    with open(device_map_path, 'w') as f:
        data = json.dumps(device_map_json, ensure_ascii=False, indent=4)
        f.write(data)


if __name__ == "__main__":
    main()
