# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain YuanVL."""

import torch
from functools import partial
from contextlib import nullcontext
import inspect

from typing import List, Optional, Tuple, Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.data.yuanvl_dataset import build_train_datasets
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    get_ltor_masks_and_position_ids_yuanvl_train,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
    get_imagemlp_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core import mpu, tensor_parallel

stimer = StragglerDetector()

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump
            dump(snapshot, open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'))

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)
    
    assert not args.use_legacy_models, '源多模态模型不支持use_legacy_models'
    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts and not args.use_lf_gate:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te, normalization=args.normalization)
            elif args.heterogeneous_layers_config_path is not None:
                transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm,args.use_lf_gate, args.use_apex_rmsnorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm,
                        normalization=args.normalization)
            if args.use_yuanvl:
                imagemlp_layer_spec = get_imagemlp_layer_with_transformer_engine_spec(args.use_apex_rmsnorm, args.yuanvl_use_te_imagemlp)
            else:
                imagemlp_layer_spec = None
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=use_te)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            imagemlp_layer_spec=imagemlp_layer_spec,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
        )

    return model


def get_batch(data_iterator):
    """Generate a batch."""
  
    # TODO: this is pretty hacky, find a better way
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys_text = ['text']
    datatype_text = torch.int64

    keys_img = ['img']
    datatype_img = torch.bfloat16

    keys_num_tile = ['num_tile']
    datatype_num_tile = torch.int64

    keys_image_per_sample = ['image_per_sample']
    datatype_image_per_sample = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)

    else:
        data = None

    text_b = tensor_parallel.broadcast_data(keys_text, data, datatype_text)
    img_b = tensor_parallel.broadcast_data(keys_img, data, datatype_img)
    num_tile_b = tensor_parallel.broadcast_data(keys_num_tile, data, datatype_num_tile)
    image_per_sample_b = tensor_parallel.broadcast_data(keys_image_per_sample, data, datatype_image_per_sample)
    
    # Unpack.
    tokens_ = text_b['text'].long()
    tokens = tokens_

    num_tile = num_tile_b['num_tile']
    image_tensor = img_b['img']
    image_per_sample = image_per_sample_b['image_per_sample']

    bos_token, image_start_token, image_end_token, pad_token, sep_token, eod_token = (tokenizer(tok)['input_ids'][0] for tok in ['<BOS>','<IMAGE>', '</IMAGE>', '<pad>', '<|Assistant|>', '<eod>'])

    attention_mask, loss_mask, position_ids, data_pad_tensor, label_pad_tensor, image_info = get_ltor_masks_and_position_ids_yuanvl_train(
            tokens,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss,
            bos_token=bos_token,
            image_start_token=image_start_token,
            image_end_token=image_end_token,
            eod_token=eod_token,
            pad_token=pad_token,
            sep_token=sep_token,
            clip_visual_size=args.clip_visual_size,
            num_tile=num_tile,
            image_per_sample=image_per_sample)

    assert args.context_parallel_size == 1, 'yuan vl can not support context parallel'

    
    return data_pad_tensor, label_pad_tensor, loss_mask, attention_mask, position_ids, image_info, image_tensor
    
# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    # loss[0] is a view of loss, so it has ._base not None, which triggers assert error
    # in core/pipeline_parallel/schedule.py::deallocate_output_tensor, calling .clone()
    # on loss[0] fixes this
    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0].clone(),
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):

        tokens, labels, loss_mask, attention_mask, position_ids, image_info, image_tensor = get_batch(
            data_iterator)
    timers('batch-generator').stop()

    with stimer:
        if args.use_legacy_models:
            raise RuntimeError("检查这里，未调试use_legacy_models=True")
        else:
            output_tensor = model(tokens, position_ids, attention_mask,
                                labels=labels, loss_mask=loss_mask, image_tensor=image_tensor, image_info=image_info)

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return mpu.get_tensor_model_parallel_rank() == 0




def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    tokenizer = get_tokenizer()

    print_rank_0('> building train, validation, and test datasets '
                 'for Flamingo ...')

    train_num_samples = train_val_test_num_samples[0]

    train_ds = build_train_datasets(
            data_path=args.data_path,
            train_num_samples=train_num_samples,
            txt_seq_length=args.seq_length,
            full_idx_path = args.yuanvl_full_npy_path
            )

    print_rank_0("> finished creating Flamingo datasets ...")
    
    valid_ds = None
    test_ds = None
    return train_ds, valid_ds, test_ds

if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
