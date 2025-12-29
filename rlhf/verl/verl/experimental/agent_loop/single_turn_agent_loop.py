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
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        metrics = {}
        request_id = uuid4().hex
        prompt_ids = kwargs["raw_prompt_ids"]
        if "multi_modal_data" in kwargs and "image" in kwargs["multi_modal_data"] \
                and kwargs["multi_modal_data"]["image"] is not None:
            multi_modal_data = kwargs["multi_modal_data"]
            vllm_inputs = [{"prompt_token_ids": prompt_ids, "multi_modal_data": multi_modal_data}]
        else:
            vllm_inputs = [{"prompt_token_ids": prompt_ids}]

        with simple_timer("generate_sequences", metrics):
            server_id = kwargs.get("server_id", -1)
            response_ids = await self.server_manager.generate(
                request_id=request_id, prompt_ids=vllm_inputs, sampling_params=sampling_params, server_id = server_id
            )
        response_mask = [1] * len(response_ids)

        output = AgentLoopOutput(
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=2,
            metrics=metrics,
        )
        return output
