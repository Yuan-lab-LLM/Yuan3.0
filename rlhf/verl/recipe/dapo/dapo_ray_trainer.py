# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

from torch.utils.data import Sampler
from typing import Optional
from torchdata.stateful_dataloader import StatefulDataLoader
from omegaconf import OmegaConf, open_dict
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
import os

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
import time
import random
import json

def datatype2list(tokenizer,data,reward_extra_infos_dict=None):
    new_data=[]
    for i,data_item in enumerate(data):
        new_dict={}

        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        # decode
        prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
        valid_response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=False)

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        reward_method = data_item.non_tensor_batch["reward_method"]
        language = data_item.non_tensor_batch["language"]  if "language" in data_item.non_tensor_batch  else None
        #image_path = data_item.non_tensor_batch['extra_info']['image_path']
        if 'image_path' in data_item.non_tensor_batch['extra_info']:
            image_path = data_item.non_tensor_batch['extra_info']['image_path']
        else:
            image_path = ''
        enable_thinking_flag = data_item.non_tensor_batch['extra_info']['enable_thinking_flag']
        # enable_thinking_flag = data_item.non_tensor_batch['extra_info']['enable_thinking_flag']

        new_dict= dict(
                      question=prompt_str,
                      valid_response_str=valid_response_str,
                      answer=ground_truth,
                      reward_method=reward_method,
                      language=language,
                      image_path=image_path,
                      enable_thinking_flag=enable_thinking_flag
                  )
        for key ,value in reward_extra_infos_dict.items():
            new_dict[key]=value[i]
        new_data.append(new_dict)
    return new_data


def increase_to_min_multiple(lst, n):
    if n <= 0:
        raise ValueError("n must be a positive integer")
    original_length = len(lst)
    if original_length == 0:
        return lst.copy()  # 返回空列表的副本

    min_multiple = (original_length + n - 1) // n  # 计算最小倍数
    target_length = min_multiple * n
    num_to_add = target_length - original_length

    if num_to_add == 0:
        return lst.copy()  # 如果已经是n的倍数，直接返回副本
    elements_to_add = [lst[i % original_length] for i in range(num_to_add)]
    return elements_to_add


def arrange_from_center(sorted_list):
    """从中间向两边交替取元素"""
    n = len(sorted_list)
    center = n // 2  # 中间位置（偶数时偏左）
    left = center - 1
    right = center + 1
    new_list = [sorted_list[center]]  # 初始从中间开始
    # 交替向左右扩展
    while left >= 0 or right < n:
        if left >= 0:
            new_list.append(sorted_list[left])
            left -= 1
        if right < n:
            new_list.append(sorted_list[right])
            right += 1
    return new_list


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    def _alternate_iterators(self, iter_short, iter_long):
        """交替从两个数据集中获取数据，直到两者都耗尽"""
        if iter_short is None:
            self.stop_short = True

        if iter_long is None:
            self.stop_long = True

        while not (self.stop_short and self.stop_long):
            # 从短数据集中获取
            if not self.stop_short:
                try:
                    batch_a = next(iter_short)
                    yield batch_a
                except StopIteration:
                    self.stop_short = True
            # 按照数据集的比例，采用取2次短数据集，1次长数据集，交替的方式
            if not self.stop_short:
                try:
                    batch_a = next(iter_short)
                    yield batch_a
                except StopIteration:
                    self.stop_short = True

            # 从长数据集中获取
            if not self.stop_long:
                try:
                    batch_b = next(iter_long)
                    yield batch_b
                except StopIteration:
                    self.stop_long = True

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        config = self.config
        clip_visual_size = getattr(config.data, 'clip_visual_size', 1024)
        downsample_ratio = getattr(config.data, 'downsample_ratio', 0.5)
        max_split_tile_num = getattr(config.data, 'max_split_tile_num', 9)

        if train_dataset is None:
            if self.config.data.train_16k_files != "":
                train_16k_dataset = create_rl_dataset(
                    self.config.data.train_16k_files,
                    self.config.data,
                    self.tokenizer,
                    self.processor,
                )
            else:
                train_16k_dataset = None

            if self.config.data.train_4k_files != "":
                train_4k_dataset = create_rl_dataset(
                    self.config.data.train_4k_files,
                    self.config.data,
                    self.tokenizer,
                    self.processor,
                )
            else:
                train_4k_dataset = None

        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
            )

        self.train_16k_dataset, self.train_4k_dataset, self.val_dataset = train_16k_dataset, train_4k_dataset, val_dataset

        train_4k_sampler = create_rl_sampler(self.config.data, self.train_4k_dataset) if self.train_4k_dataset != None else None
        train_16k_sampler = create_rl_sampler(self.config.data, self.train_16k_dataset) if self.train_16k_dataset != None else None
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn
        num_workers = self.config.data["dataloader_num_workers"]

        if train_16k_sampler:
            self.train_16k_dataloader = StatefulDataLoader(
                dataset=self.train_16k_dataset,
                batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
                num_workers=num_workers,
                drop_last=True,
                collate_fn=collate_fn,
                sampler=train_16k_sampler,
            )
        else:
            self.train_16k_dataloader = None

        if train_4k_sampler:
            self.train_4k_dataloader = StatefulDataLoader(
                dataset=self.train_4k_dataset,
                batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
                num_workers=num_workers,
                drop_last=True,
                collate_fn=collate_fn,
                sampler=train_4k_sampler,
            )
        else:
            self.train_4k_dataloader = None

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )
        data_len = len(self.train_4k_dataloader) if self.train_4k_dataloader else 0
        data_len += len(self.train_16k_dataloader) if self.train_16k_dataloader else 0
        assert data_len >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {data_len}"
            f"Size of val dataloader: {len(self.val_dataloader)}"
        )

        total_training_steps = data_len * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")


    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        if self.train_4k_dataloader and not self.stop_short:
            dataloader_4k_local_path = os.path.join(local_global_step_folder, "4k_data.pt")
            dataloader_4k_state_dict = self.train_4k_dataloader.state_dict()
            print(f"save dataloader_4k_state_dict is {dataloader_4k_state_dict}")
            torch.save(dataloader_4k_state_dict, dataloader_4k_local_path)

        if self.train_16k_dataloader and not self.stop_long:
            dataloader_16k_local_path = os.path.join(local_global_step_folder, "16k_data.pt")
            dataloader_16k_state_dict = self.train_16k_dataloader.state_dict()
            print(f"save dataloader_16k_state_dict is {dataloader_16k_state_dict}")
            torch.save(dataloader_16k_state_dict, dataloader_16k_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))


    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_4k_local_path = os.path.join(global_step_folder, "4k_data.pt")
        dataloader_16k_local_path = os.path.join(global_step_folder, "16k_data.pt")
        bFind4K = False
        bFind16K = False
        if os.path.exists(dataloader_16k_local_path):
            dataloader_state_dict = torch.load(dataloader_16k_local_path, weights_only=False)
            self.train_16k_dataloader.load_state_dict(dataloader_state_dict)
            bFind16K =  True
            #self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_16k_local_path}")

        if os.path.exists(dataloader_4k_local_path):
            dataloader_state_dict = torch.load(dataloader_4k_local_path, weights_only=False)
            self.train_4k_dataloader.load_state_dict(dataloader_state_dict)
            bFind4K = True
            #self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_4k_local_path}, will start from scratch")

        # 如果有任意一个保存了，另一个没有保存，说明没有保存的已经跑完所有数据，stop设置为True
        self.stop_long = not bFind16K and bFind4K
        self.stop_short = not bFind4K and bFind16K

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        self.stop_long = False
        self.stop_short = False

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.trainer.profile_steps
            if self.config.trainer.profile_steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

        rollout_tp_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        rollout_pp_size = self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        rollout_mp_size = rollout_tp_size * rollout_pp_size
        rollout_dp_size = (self.config.trainer.n_gpus_per_node*self.config.trainer.nnodes) // rollout_mp_size

        for epoch in range(self.config.trainer.total_epochs):
            print('run training')
            iter_4k = iter(self.train_4k_dataloader) if self.train_4k_dataloader else None
            iter_16k = iter(self.train_16k_dataloader) if self.train_16k_dataloader else None
            for batch_dict in self._alternate_iterators(iter_4k, iter_16k):
                metrics = {}
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if self.async_rollout_mode:
                    non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                    if "multi_modal_data" in new_batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("multi_modal_data")

                    assert "request_response_length" in new_batch.non_tensor_batch, f"request_response_length should in new_batch.non_tensor_batch"
                    non_tensor_batch_keys_to_pop.append("request_response_length")

                    assert "expect_len" in new_batch.non_tensor_batch, f"expect_len should in new_batch.non_tensor_batch"
                    non_tensor_batch_keys_to_pop.append("expect_len")

                    gen_batch = new_batch.pop(
                        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    )
                elif "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "processed_images"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    ) 

                # raw_prompt_ids 是加入chat template之后的token ids，没有补 <pad>
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        start_time = time.time()
                        print('run vllm', len(gen_batch), list(gen_batch.non_tensor_batch.keys()), flush=True)
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            #group_ids, _ = self._get_group_ids_by_weight_greedy(gen_batch.non_tensor_batch['request_response_length'].tolist(), rollout_dp_size)
                            #gen_batch.non_tensor_batch["server_id"] = np.array(group_ids)
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"generate a batch elapsed_time: {elapsed_time}")
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                    # adv_estimator = grpo，不会跑到下面这个分支
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    if self.async_rollout_mode:
                        new_batch.batch["prompts"] = deepcopy(new_batch.batch["input_ids"])
                        new_batch.batch["input_ids"] = torch.cat([new_batch.batch["input_ids"], gen_batch_output.batch["responses"]], dim=-1)
                        new_batch.batch["attention_mask"] = torch.cat([new_batch.batch["attention_mask"], gen_batch_output.batch["response_mask"]], dim=-1)
                        position_ids_response = (gen_batch_output.batch["response_mask"].cumsum(dim=-1) + new_batch.batch["position_ids"][:,-1:]) * gen_batch_output.batch["response_mask"]
                        new_batch.batch["position_ids"] = torch.cat([new_batch.batch["position_ids"], position_ids_response], dim=-1)
                        new_batch.batch["responses"] = gen_batch_output.batch.pop("responses")
                        for k, v in gen_batch_output.non_tensor_batch.items():
                            new_batch.non_tensor_batch[k] = v
                        for k, v in gen_batch.non_tensor_batch.items():
                            new_batch.non_tensor_batch[k] = v

                        del gen_batch_output, gen_batch 
                    else:
                        new_batch = new_batch.union(gen_batch_output)


                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        print('run reward')
                        start_time = time.time()
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )
                            # add overlong_filter_flag:
                            for i,item in enumerate(new_batch.non_tensor_batch['extra_info']):
                                item['overlong_filter_flag'] = new_batch.non_tensor_batch['overlong_filter_flag'][i]
                        
                        end_reward_time = time.time()
                        print(f"reward_fn elapsed_time: {end_reward_time - start_time}")

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"reward elapsed_time: {elapsed_time}")

                    gen_batch_acc = new_batch.non_tensor_batch["acc"]
                    enable_thinking_flag_list = [data_item.non_tensor_batch['extra_info']['enable_thinking_flag'] for data_item in new_batch]
                    reward_method_list = new_batch.non_tensor_batch["reward_method"]
                    gen_batch_acc_dct = defaultdict(list)
                    
                    for idx, reward_method in enumerate(reward_method_list):
                        enable_thinking_flag = None if not enable_thinking_flag_list else enable_thinking_flag_list[idx]

                        if reward_method in ['mllm_normal', 'mllm_choice']:
                            reward_method = 'mllm_math'
                        elif reward_method in ['mllm_knowledge', 'mllm_grounding', 'mllm_general1', 'mllm_general2']:
                            reward_method = 'mllm_chat'
                        if enable_thinking_flag is not None:
                            class_name = "longcot" if enable_thinking_flag else "shortcot"
                            reward_method = f"{class_name}_{reward_method}"
                            gen_batch_acc_dct[class_name].append(gen_batch_acc[idx])
                        
                        gen_batch_acc_dct[reward_method].append(gen_batch_acc[idx])
                    for key, value in gen_batch_acc_dct.items():
                        avg_acc = sum([0.0 if (x is None or x < -1) else (x + 1)/2 for x in value]) / len(value)
                        metrics.update({f"train/gen_batch_{key}_acc": avg_acc})

                    gen_batch_avg_acc = sum([0.0 if (x is None or x < -1) else (x + 1)/2 for x in gen_batch_acc]) / len(gen_batch_acc)
                    metrics.update({"train/gen_batch_acc": gen_batch_avg_acc})


                    # TODO: create ./steps_reward_data in script
                    step_reward_name=f"{self.config.trainer.default_local_dir}/steps_reward_data/{self.global_steps}_reward_data.json"
                    with open(step_reward_name, "w") as f:
                        data_to_write = datatype2list(self.tokenizer, new_batch,reward_extra_infos_dict)
                        f.write(json.dumps(data_to_write, ensure_ascii=False, indent=4))

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        start_time = time.time()
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                            print(f"seq_final_reward: {new_batch.non_tensor_batch['seq_final_reward']}")
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )
                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std, prompt_uid2group_acc = {}, {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)
                            prompt_uid2group_acc[prompt_uid] = sum([1.0 for x in metric_vals if x == 1.0]) / len(metric_vals) 
                        print(f"len(prompt_uid2group_acc): {len(prompt_uid2group_acc)}")
                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        new_batch_ = None
                        if len(kept_prompt_uids) % self.config.actor_rollout_ref.actor.ppo_mini_batch_size != 0 and len(kept_prompt_uids) != 0:
                            kept_prompt_uid2group_acc = {k:v for k,v in prompt_uid2group_acc.items() if k in kept_prompt_uids}
                            if self.config.algorithm.filter_groups.hard_to_easy: #"hard to easy":
                                sorted_prompt_uids = sorted(kept_prompt_uid2group_acc.keys(), key=lambda k: kept_prompt_uid2group_acc[k], reverse=False)
                            elif self.config.algorithm.filter_groups.easy_to_hard: #"easy to hard":
                                sorted_prompt_uids = sorted(kept_prompt_uid2group_acc.keys(), key=lambda k: kept_prompt_uid2group_acc[k], reverse=True)
                            elif self.config.algorithm.filter_groups.center_to_sides: #"center to sides":
                                sorted_prompt_uids = sorted(kept_prompt_uid2group_acc.keys(), key=lambda k: kept_prompt_uid2group_acc[k], reverse=False)
                                sorted_prompt_uids = arrange_from_center(sorted_prompt_uids)
                            else: # random
                                sorted_prompt_uids = kept_prompt_uids.copy()
                                random.shuffle(sorted_prompt_uids)
                            print(f"len(sorted_prompt_uids): {len(sorted_prompt_uids)}")
                            prompt_uids_to_add = increase_to_min_multiple(sorted_prompt_uids, self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
                            assert len(kept_prompt_uids+prompt_uids_to_add)%self.config.actor_rollout_ref.actor.ppo_mini_batch_size == 0, "kept_prompts still not multiple of mini batch"
                            print(f"len(prompt_uids_to_add): {len(prompt_uids_to_add)}")
                            pass_rates_ = {k:v for k,v in prompt_uid2group_acc.items() if k in prompt_uids_to_add}
                            num_prompt_in_batch += len(prompt_uids_to_add)
                            added_traj_idxs = []
                            prompt_uid2idx = defaultdict(list)
                            for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                                prompt_uid2idx[traj_from_prompt_uid].append(idx)
                            for uid in prompt_uids_to_add:
                                added_traj_idxs.extend(prompt_uid2idx[uid])

                            print(f"len(added_traj_idxs): {len(added_traj_idxs)}")
                            new_batch_ = new_batch[added_traj_idxs]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = DataProto.concat([new_batch[kept_traj_idxs], new_batch_]) if new_batch_ is not None else new_batch[kept_traj_idxs]

                        if self.config.algorithm.filter_groups.reorder_batch_enable:
                            # reorder the new_batch according to the pass_rate
                            pass_rates = [prompt_uid2group_acc[u] for u in new_batch.non_tensor_batch["uid"]]
                            if self.config.algorithm.filter_groups.reorder_batch_hard_to_easy:# hard_to_easy
                                sorted_idx = sorted(range(len(pass_rates)), key=lambda i: pass_rates[i])
                            elif self.config.algorithm.filter_groups.reorder_batch_easy_to_hard:# easy_to_hard
                                sorted_idx = sorted(range(len(pass_rates)), key=lambda i: pass_rates[i], reverse=True)
                            else:# random
                                sorted_idx = list(range(len(pass_rates)))
                                random.shuffle(sorted_idx)

                            _pass_rates = [pass_rates[i] for i in sorted_idx]
                            #print(f"sorted_pass_rates: {_pass_rates}")
                            new_batch.non_tensor_batch["pass_rates"] = np.array(pass_rates)
                            new_batch.reorder(torch.tensor(sorted_idx))
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        print(f"len(batch.batch): {len(batch.batch)}")
                        prompt_bsz = self.config.data.train_batch_size
                        print(f"prompt_bsz: {prompt_bsz}")
                        print(f"num_prompt_in_batch: {num_prompt_in_batch}")
                        if num_prompt_in_batch == 0:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"filter_groups elapsed_time: {elapsed_time}")

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                        print("打开balance")
                    else:
                        print("关闭balance")

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        start_time = time.time()
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"recompute old_log_probs elapsed_time: {elapsed_time}")

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            start_time = time.time()
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            print(f"compute reference log_probs elapsed_time: {elapsed_time}")

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        start_time = time.time()
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"compute advantages elapsed_time: {elapsed_time}")
                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            start_time = time.time()
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            print(f"update actor elapsed_time: {elapsed_time}")
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.trainer.profile_steps
                        if self.config.trainer.profile_steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1

    def _get_group_ids_by_weight_greedy(self, weights, num_groups):
        """
        Assign group IDs using the greedy algorithm.
        Each time, add the current data to the group with the smallest total weight.
        """

        n = len(weights)
        group_ids = [0] * n

        # initialize
        group_weights = [0] * num_groups

        # sort by weight
        sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)

        for idx in sorted_indices:
            # find the group
            min_weight_group = min(range(num_groups), key=lambda g: group_weights[g])

            # assign to group
            group_ids[idx] = min_weight_group
            group_weights[min_weight_group] += weights[idx]

        print(f"group_weights is {group_weights}")
        from collections import Counter

        print(f"id counts: {Counter(group_ids)}")

        return group_ids, group_weights
