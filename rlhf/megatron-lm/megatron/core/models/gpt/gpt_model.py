# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from collections import OrderedDict
from typing import Dict, Literal, Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel, mpu
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
    RotaryEmbedding,
)
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    tie_output_layer_state_dict,
    tie_word_embeddings_state_dict,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import WrappedTensor, deprecate_inference_params

from megatron.core.models.gpt.yuanvl_clip import clip_encode_image
from functools import partial

class GPTModel(LanguageModule):
    """GPT Transformer language model.

    Args:
        config (TransformerConfig):
            Transformer config
        transformer_layer_spec (ModuleSpec):
            Specifies module to use for transformer layers
        vocab_size (int):
            Vocabulary size
        max_sequence_length (int):
            maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional):
            Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional):
            Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional):
            Defaults to False.
        parallel_output (bool, optional):
            Do not gather the outputs, keep them split across tensor
            parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional):
            When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):
            Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional):
            Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional):
            Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'.
            Defaults to 10000.
        rope_scaling (bool, optional): Toggle RoPE scaling.
        rope_scaling_factor (float): RoPE scaling factor. Default 8.
        scatter_embedding_sequence_parallel (bool, optional):
            Whether embeddings should be scattered across sequence parallel
            region or not. Defaults to True.
        seq_len_interpolation_factor (Optional[float], optional):
            scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        imagemlp_layer_spec: ModuleSpec = None,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            'learned_absolute', 'rope', 'mrope', 'none'
        ] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
    ) -> None:
        super().__init__(config=config)

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.use_loss_chunk = self.config.use_loss_chunk
        self.loss_chunk = self.config.loss_chunk

        if hasattr(self.config, 'position_embedding_type'):
            self.position_embedding_type = self.config.position_embedding_type
        else:
            self.position_embedding_type = position_embedding_type

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        # These 4 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent

        if hasattr(self.config, 'rotary_base'):
            self.rotary_base = self.config.rotary_base
        else:
            self.rotary_base = rotary_base
        self.rotary_scaling = rope_scaling
        self.mtp_block_spec = mtp_block_spec
        self.mtp_process = mtp_block_spec is not None
        self.hidden_size = config.hidden_size
        
        self.use_yuanvl = self.config.use_yuanvl

        self.compute_probs_in_model = self.config.compute_probs_in_model
        self.compute_lm_head_fp32 = self.config.compute_lm_head_fp32

        if self.pre_process or self.mtp_process:
            if self.use_yuanvl:
                self.imagemlp_recompute = self.config.imagemlp_recompute
                self.downsample_ratio = self.config.downsample_ratio
                self.clip_hidden_size = self.config.clip_hidden_size
                self.clip_model_name = self.config.clip_model_name
                self.clip_visual_size = self.config.clip_visual_size
                self.clip_download_path = self.config.clip_download_path
                
                self.imagemlp_input_hiddensize = int(self.clip_hidden_size / self.downsample_ratio ** 2)
                self.num_token_per_tile = int(self.clip_visual_size * self.downsample_ratio**2)
                
                self.pixel_unshuffle = torch.nn.PixelUnshuffle(downscale_factor=int(1 / self.downsample_ratio))
                
                self.image_encode = partial(clip_encode_image,
                                            model_name = self.clip_model_name,
                                            clip_download_path = self.clip_download_path
                                            )
                self.imagemlp = build_module(imagemlp_layer_spec, config=self.config, input_size=self.imagemlp_input_hiddensize)
                
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
            )

        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                rope_scaling=rope_scaling,
                rope_scaling_factor=rope_scaling_factor,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )

        elif self.position_embedding_type == 'mrope' and not self.config.multi_latent_attention:
            self.rotary_pos_emb = MultimodalRotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
            )
            self.mrope_section = self.config.mrope_section
            assert (
                self.mrope_section is not None
            ), "mrope require mrope_section setting, but we got None from TransformerConfig"

        # Cache for RoPE tensors which do not change between iterations.
        self.rotary_pos_emb_cache = {}

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        if self.mtp_process:
            self.mtp = MultiTokenPredictionBlock(config=self.config, spec=self.mtp_block_spec)

        # Output
        if self.post_process or self.mtp_process:

            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs
                # stored in gradient buffer to calculate the weight gradients for the embedding
                # final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

        if has_config_logger_enabled(self.config):
            log_config_to_disk(
                self.config, self.state_dict(), prefix=f'{type(self).__name__}_init_ckpt'
            )

        if mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:
            print(self)

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
        image_tensor: Optional[Tensor] = None,
        image_info: Optional[dict] = None,
        logits_processor = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.
        inference_context = deprecate_inference_params(inference_context, inference_params)

        def custom_forward(image_feature):
            image_feature = self.imagemlp(image_feature)
            return image_feature

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
            if self.use_yuanvl:
                if image_info is not None:
                    if decoder_input.shape[0] >= self.num_token_per_tile or self.training:
                        decoder_input = decoder_input.clone() 
                        imbs, num_image, ic, ih, iw = image_tensor.shape
                        image_tensor = image_tensor.view(imbs * num_image, ic, ih, iw)
                        image_feature = self.image_encode(image_tensor)
                        image_feature = image_feature[:, 1:, :]
                        
                        pn, phw, pc = image_feature.shape
                        ph = pw = int(phw**0.5)
                        image_feature = image_feature.view(pn, ph, pw, pc).permute(0, 3, 1, 2)
                        image_feature = self.pixel_unshuffle(image_feature)
                        pn, pc, ph, pw = image_feature.shape
                        image_feature = image_feature.view(pn, pc, ph * pw).permute(0, 2, 1)
                        _, cvs, chs = image_feature.shape
                        assert self.imagemlp_input_hiddensize == chs
                        image_feature = image_feature.contiguous().view(imbs, num_image * cvs, chs).permute(1, 0, 2).contiguous()
                        if self.imagemlp_recompute:
                            image_feature.requires_grad = True
                            image_feature = tensor_parallel.checkpoint(custom_forward, False, image_feature)
                        else:
                            image_feature = custom_forward(image_feature)
                            
                        image_feature = image_feature.view(num_image, cvs, imbs, -1)
                        num_all_tile = 0
                        
                    if self.training:
                        encoder_input_merge_list = []

                        num_image_per_sample = len(image_info['num_tile'])
                        assert num_image_per_sample == len(image_info['image_start_pos'])
                        for i in range(num_image_per_sample):
                            num_tile_this_image = image_info['num_tile'][i]
                            image_pos = image_info['image_start_pos'][i]
                            decoder_input[image_pos + 1 : image_pos + cvs * num_tile_this_image + 1] = image_feature[num_all_tile : num_all_tile + num_tile_this_image].view(-1, imbs, self.hidden_size)
                            num_all_tile += num_tile_this_image

                    else:
                        if decoder_input.shape[0] >= self.num_token_per_tile:
                            num_image_per_sample = len(image_info['num_tile'])
                            assert num_image_per_sample == len(image_info['image_start_pos'])
                            for i in range(num_image_per_sample):
                                num_tile_this_image = image_info['num_tile'][i]
                                image_pos = image_info['image_start_pos'][i]
                                decoder_input[image_pos + 1 : image_pos + cvs * num_tile_this_image + 1] = image_feature[num_all_tile : num_all_tile + num_tile_this_image].view(-1, imbs, self.hidden_size)
                                num_all_tile += num_tile_this_image
                        else:
                            decoder_input = decoder_input
                        
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            if not self.training and self.config.flash_decode and inference_context:
                assert (
                    inference_context.is_static_batching()
                ), "GPTModel currently only supports static inference batching."
                # Flash decoding uses precomputed cos and sin for RoPE
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb_cache.setdefault(
                    inference_context.max_sequence_length,
                    self.rotary_pos_emb.get_cos_sin(inference_context.max_sequence_length),
                )
            else:
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    inference_context, self.decoder, decoder_input, self.config, packed_seq_params
                )
                rotary_pos_emb = self.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None
                    and packed_seq_params.qkv_format == 'thd',
                )
        elif self.position_embedding_type == 'mrope' and not self.config.multi_latent_attention:
            if self.training or not self.config.flash_decode:
                rotary_pos_emb = self.rotary_pos_emb(position_ids, self.mrope_section)
            else:
                # Flash decoding uses precomputed cos and sin for RoPE
                raise NotImplementedError(
                    "Flash decoding uses precomputed cos and sin for RoPE, not implmented in "
                    "MultimodalRotaryEmbedding yet."
                )

        if (
            (self.config.enable_cuda_graph or self.config.flash_decode)
            and rotary_pos_cos is not None
            and inference_context
            and inference_context.is_static_batching()
            and not self.training
        ):
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * inference_context.current_batch_size,
                dtype=torch.int32,
                device=rotary_pos_cos.device,  # Co-locate this with the rotary tensors
            )
        else:
            sequence_len_offset = None

        # Wrap decoder_input to allow the decoder (TransformerBlock) to delete the
        # reference held by this caller function, enabling early garbage collection for
        # inference. Skip wrapping if decoder_input is logged after decoder completion.
        if (
            inference_context is not None
            and not self.training
            and not has_config_logger_enabled(self.config)
        ):
            decoder_input = WrappedTensor(decoder_input)

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **(extra_block_kwargs or {}),
        )

        # Process inference output.
        if inference_context and not inference_context.is_static_batching():
            hidden_states = inference_context.last_token_logits(
                hidden_states.squeeze(1).unsqueeze(0)
            ).unsqueeze(1)

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        if self.mtp_process:
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                loss_mask=loss_mask,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                embedding=self.embedding,
                output_layer=self.output_layer,
                output_weight=output_weight,
                runtime_gather_output=runtime_gather_output,
                compute_language_model_loss=self.compute_language_model_loss,
                **(extra_block_kwargs or {}),
            )

        if not self.post_process:
            return hidden_states

        if (
            not self.training
            and inference_context is not None
            and inference_context.is_static_batching()
            and inference_context.materialize_only_last_token_logits
        ):
            hidden_states = hidden_states[-1:, :, :]

        if self.compute_probs_in_model:
            assert self.use_loss_chunk, 'If compute_rlhf=True, use_loss_chunk must be True'
            assert (labels is not None 
                    and loss_mask is not None 
                    and logits_processor is not None ), 'If compute_rlhf=True, labels, loss_mask, logits_processor must be input'
            assert has_config_logger_enabled(self.config) == False
            seqlength, _, _ = hidden_states.shape
            num_loop = seqlength // self.loss_chunk
            remainder = seqlength % self.loss_chunk
            entropy = torch.ones_like(labels, device=hidden_states.device, dtype=torch.float32)
            log_probs = torch.ones_like(labels, device=hidden_states.device, dtype=torch.float32)

            inum_loop = None
            for inum_loop in range(num_loop):
                logits, _ = self.output_layer(hidden_states[inum_loop * self.loss_chunk : (inum_loop + 1) * self.loss_chunk, :, :], weight=output_weight, runtime_gather_output=runtime_gather_output, compute_lm_head_fp32=self.compute_lm_head_fp32)
                logits = logits.to(entropy)

                logits = logits.transpose(0, 1).contiguous()
                output_dict = logits_processor(logits,
                                               labels[:, inum_loop * self.loss_chunk : (inum_loop + 1) * self.loss_chunk],
                                               loss_mask[:, inum_loop * self.loss_chunk : (inum_loop + 1) * self.loss_chunk])
                
                if "entropy" in output_dict:
                    entropy[:, inum_loop * self.loss_chunk : (inum_loop + 1) * self.loss_chunk] = output_dict["entropy"]
                    return_entropy = True
                else:
                    return_entropy = False
                log_probs[:, inum_loop * self.loss_chunk : (inum_loop + 1) * self.loss_chunk] =  output_dict["log_probs"]
                

            if remainder > 0:
                if inum_loop is None:
                    inum_loop = -1
                logits, _ = self.output_layer(hidden_states[(inum_loop + 1) * self.loss_chunk:, :, :], weight=output_weight, runtime_gather_output=runtime_gather_output, compute_lm_head_fp32=self.compute_lm_head_fp32)
                logits = logits.to(entropy)

                logits = logits.transpose(0, 1).contiguous()
                output_dict = logits_processor(logits,
                                               labels[:, (inum_loop + 1) * self.loss_chunk:],
                                               loss_mask[:, (inum_loop + 1) * self.loss_chunk:])
                if "entropy" in output_dict:
                    entropy[:, (inum_loop + 1) * self.loss_chunk:] = output_dict["entropy"]
                    return_entropy = True
                else:
                    return_entropy = False
                log_probs[:, (inum_loop + 1) * self.loss_chunk:] =  output_dict["log_probs"]
            if return_entropy:
                output_all = torch.cat((entropy, log_probs), dim=-1)
                return output_all
            else:
                return log_probs
        
        else:
            if self.use_loss_chunk and self.training and labels is not None:
                assert has_config_logger_enabled(self.config) == False
                seqlength, _, _ = hidden_states.shape
                self.loss_chunk = 20000
                num_loop = seqlength // self.loss_chunk
                remainder = seqlength % self.loss_chunk
                loss = torch.ones_like(labels, device=hidden_states.device, dtype=torch.float32)
                inum_loop = None
                for inum_loop in range(num_loop):
                    logits, _ = self.output_layer(hidden_states[inum_loop * self.loss_chunk : (inum_loop + 1) * self.loss_chunk, :, :], weight=output_weight, runtime_gather_output=runtime_gather_output, compute_lm_head_fp32=self.compute_lm_head_fp32)

                    if labels is None:
                        # [s b h] => [b s h]
                        return logits.transpose(0, 1).contiguous()

                    loss[:, inum_loop * self.loss_chunk : (inum_loop + 1) * self.loss_chunk] = self.compute_language_model_loss(labels[:, inum_loop * self.loss_chunk : (inum_loop + 1) * self.loss_chunk], logits)
                if remainder > 0:
                    if inum_loop is None:
                        inum_loop = -1
                    logits, _ = self.output_layer(hidden_states[(inum_loop + 1) * self.loss_chunk:, :, :], weight=output_weight, runtime_gather_output=runtime_gather_output, compute_lm_head_fp32=self.compute_lm_head_fp32)

                    if labels is None:
                        # [s b h] => [b s h]
                        return logits.transpose(0, 1).contiguous()

                    loss[:, (inum_loop + 1) * self.loss_chunk:] = self.compute_language_model_loss(labels[:, (inum_loop + 1) * self.loss_chunk:], logits)
            else:
                logits, _ = self.output_layer(
                    hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output, compute_lm_head_fp32=self.compute_lm_head_fp32
                )

                if has_config_logger_enabled(self.config):
                    payload = OrderedDict(
                        {
                            'input_ids': input_ids,
                            'position_ids': position_ids,
                            'attention_mask': attention_mask,
                            'decoder_input': decoder_input,
                            'logits': logits,
                        }
                    )
                    log_config_to_disk(self.config, payload, prefix='input_and_logits')

                if labels is None:
                    # [s b h] => [b s h]
                    return logits.transpose(0, 1).contiguous()

                loss = self.compute_language_model_loss(labels, logits)

            return loss


    def shared_embedding_or_output_weight(self) -> Tensor:
        """Gets the embedding weight or output logit weights when share input embedding and
        output weights set to True or when use Multi-Token Prediction (MTP) feature.

        Returns:
            Tensor: During pre processing or MTP process it returns the input embeddings weight.
            Otherwise, during post processing it returns the final output layers weight.
        """
        if self.pre_process or self.mtp_process:
            # Multi-Token Prediction (MTP) need both embedding layer and output layer.
            # So there will be both embedding layer and output layer in the mtp process stage.
            # In this case, if share_embeddings_and_output_weights is True, the shared weights
            # will be stored in embedding layer, and output layer will not have any weight.
            assert hasattr(
                self, 'embedding'
            ), f"embedding is needed in this pipeline stage, but it is not initialized."
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility.

        Removing extra state.
        Tie word embeddings and output layer in mtp process stage.

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        if self.config.selectable_expert:
            metadata = metadata or {}
            metadata['non_homogeneous_layers'] = True
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        output_layer_extra_state_key = f'{prefix}output_layer._extra_state'

        # Old GPT checkpoints only stored the output layer weight key. So we remove the
        # _extra_state key but check that it doesn't contain any data anyway
        output_extra_state = sharded_state_dict.pop(output_layer_extra_state_key, None)
        assert not (
            output_extra_state and output_extra_state.data
        ), f'Expected output layer extra state to be empty, got: {output_extra_state}'

        # Multi-Token Prediction (MTP) need both embedding layer and output layer in
        # mtp process stage.
        # If MTP is not placed in the pre processing stage, we need to maintain a copy of
        # embedding layer in the mtp process stage and tie it to the embedding in the pre
        # processing stage.
        # Also, if MTP is not placed in the post processing stage, we need to maintain a copy
        # of output layer in the mtp process stage and tie it to the output layer in the post
        # processing stage.
        if self.mtp_process and not self.pre_process:
            emb_weight_key = f'{prefix}embedding.word_embeddings.weight'
            emb_weight = self.embedding.word_embeddings.weight
            tie_word_embeddings_state_dict(sharded_state_dict, emb_weight, emb_weight_key)
        if self.mtp_process and not self.post_process:
            # We only need to tie the output layer weight if share_embeddings_and_output_weights
            # is False. Because if share_embeddings_and_output_weights is True, the shared weight
            # will be stored in embedding layer, and output layer will not have any weight.
            if not self.share_embeddings_and_output_weights:
                output_layer_weight_key = f'{prefix}output_layer.weight'
                output_layer_weight = self.output_layer.weight
                tie_output_layer_state_dict(
                    sharded_state_dict, output_layer_weight, output_layer_weight_key
                )

        return sharded_state_dict
