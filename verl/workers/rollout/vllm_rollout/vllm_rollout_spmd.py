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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import pickle
import socket
import threading
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
from typing import Any, List

import numpy as np
import ray
import torch
import torch.distributed
import zmq
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.worker_base import WorkerWrapperBase

from math_verify import parse, verify
from verl import DataProto
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.math_verify_sum import getRawCorrectness
# from verl.workers.rollout.vllm_rollout.redundant_token_eviction import redundant_token_eviction
import random
import requests
from uuid import UUID
import requests
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict
import math
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from verl.workers.rollout.vllm_rollout.redundant_token_eviction_confidence import redundant_token_eviction

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)
        max_num_batched_tokens = max(max_model_len, max_num_batched_tokens)
        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.summary_mode = config.get('summary_mode', 'None')

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )
        self.model_path = model_path

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)
    
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:

        # Check if tree mode is enabled
        is_validate = prompts.meta_info.get("validate", False)

        if is_validate:
            return self.generate_original_sequences(prompts, **kwargs)
        else:
            return self.generate_summarization_sequences(prompts, **kwargs)



    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_original_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        print("Self.sampling_params: ", self.sampling_params)
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=True,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    
    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_summarization_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts while summarization the content

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        rollout_n = prompts.meta_info["rollout_n"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )
        data_source_list = non_tensor_batch['data_source']
        reward_model_list = non_tensor_batch['reward_model']
        uuid_list = non_tensor_batch['uuid']
        

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size
        
        kwargs["logprobs"] = 20

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            # sample more to have more room for errors if the format is not correct
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=True,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)


            response = []
            rollout_log_probs = []
            counter = 0
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    prompt_ids = output.prompt_token_ids
                    confidence_scores = []
                    for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                        confidence_scores.append((self.token_confidence(logprob), response_ids[i]))
                    response.append({"prompt_ids": prompt_ids, "response_ids": response_ids, "summarized": False, "uuid": uuid_list[counter], 'data_source': data_source_list[counter], 'reward_model': reward_model_list[counter], 'confidence_scores': confidence_scores})
                    counter += 1
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)
            
            uuid_set = set([res['uuid'] for res in response])
            uuid_difficulty = {}
            for id in uuid_set:
                response_ids = [res['response_ids'] for res in response if res['uuid'] == id]
                prompt_ids = [res['prompt_ids'] for res in response if res['uuid'] == id]
                data_source = [res['data_source'] for res in response if res['uuid'] == id]
                reward_model = [res['reward_model'] for res in response if res['uuid'] == id]
                score_list = []
                for i in range(len(response_ids)):
                    solution_str = self.tokenizer.decode(response_ids[i])
                    score = default_compute_score(data_source[i], solution_str, reward_model[i]['ground_truth'], self.model_path)['score']
                    if score > 0:
                        score_list.append(1)
                    else:
                        score_list.append(0)
                uuid_difficulty[id] = sum(score_list) / len(score_list)
                print(f"difficulty: {uuid_difficulty[id]}")
        

            # breakpoint()
            
            correct_format_response = []
            summary_inputs = []
            for res in response:
                prompt_ids = res["prompt_ids"]
                response_ids = res["response_ids"]
                uuid = res["uuid"]
                difficulty = uuid_difficulty[uuid]
                res['difficulty'] = difficulty
                data_source = res['data_source']
                reward_model = res['reward_model']
                confidence_scores = res['confidence_scores']
                correct_format, think_str, confidence_scores = self.check_format(self.model_path, response_ids, confidence_scores)
                # breakpoint()
                score = default_compute_score(data_source, self.tokenizer.decode(response_ids), reward_model['ground_truth'], self.model_path)['score']
                answer = res['reward_model']['ground_truth']
                if (correct_format and score > 0):
                    summary_inputs.append({'think_str': think_str, 'difficulty_category': self.get_difficulty_class(difficulty), 'reward_model': reward_model, 'confidence_scores': confidence_scores})
                    correct_format_response.append({"prompt_ids": prompt_ids, "response_ids": response_ids, "think_str": think_str, 'difficulty': difficulty, 'uuid': uuid, 'data_source': data_source, 'reward_model': reward_model})

            if self.summary_mode == 'None':
                summary_inputs = []
            print(f"Summarizing {len(summary_inputs)} responses")

            if len(summary_inputs) == 0:
                final_response = [res["response_ids"] for res in response]
                response_is_summarized = [False] * len(final_response)
                uuid_list = [res['uuid'] for res in response]
                difficulty_list = [res['difficulty'] for res in response]
            else:
                original_uuid_list = [res['uuid'] for res in response]
                if self.summary_mode == 'attention_weights' or self.summary_mode == 'compression_no_difficulty':
                    summary_outputs = self.summarize_attention_weights(summary_inputs)
                elif self.summary_mode == 'confidence_scores':
                    summary_outputs = self.summarize_confidence_scores(summary_inputs)
                elif self.summary_mode == 'self_summary':
                    summary_outputs = self.summarize_think(summary_inputs)
                elif self.summary_mode == 'early_exit_attention_weights' or self.summary_mode == 'early_exit':
                    summary_outputs = self.summarize_early_exit_attention_weights(summary_inputs)
                else:
                    raise ValueError(f"Invalid summary mode: {self.summary_mode}")
                
                for i, summary_output in enumerate(summary_outputs):
                    correct_format_response[i]["summary_output"] = summary_output
                

                vllm_inputs_summarized = []
                for res in correct_format_response:
                    prompt_ids = res["prompt_ids"]
                    response_ids = res["response_ids"]
                    think_str = res["think_str"]
                    summary_output = res["summary_output"]
                    difficulty = res["difficulty"]
                    uuid = res['uuid']
                    vllm_input, response_ids_with_summarized_thinking = self.create_vllm_summary_inputs(prompt_ids, summary_output, res)

                    vllm_inputs_summarized.append({'original_prompt_ids': prompt_ids, 'original_response_ids': response_ids, 'response_ids_with_summarized_thinking': response_ids_with_summarized_thinking, 'summarized_prompt_ids': vllm_input, 'difficulty': difficulty, 'uuid': uuid})
                
                max_tokens = self.config.response_length - max([len(vllm_input['summarized_prompt_ids']) for vllm_input in vllm_inputs_summarized])
                max_tokens = max(1, max_tokens)
                print(f"Summarized response length: {[len(vllm_input['summarized_prompt_ids']) for vllm_input in vllm_inputs_summarized]}")
                print(f"Max tokens: {max_tokens} response length: {self.config.response_length}")
                with self.update_sampling_params(
                    n=1,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=max_tokens
                ):
                    summarized_outputs = self.inference_engine.generate(
                        prompts=[{'prompt_token_ids': vllm_input['summarized_prompt_ids']} for vllm_input in vllm_inputs_summarized],  # because we have already convert it to prompt token id
                        sampling_params=self.sampling_params,
                        lora_request=lora_requests,
                        use_tqdm=True,
                    )

                    # breakpoint()

                    for i, res in enumerate(summarized_outputs):
                        response_ids = res.outputs[0].token_ids
                        prompt_ids = res.prompt_token_ids
                        assert prompt_ids == vllm_inputs_summarized[i]["summarized_prompt_ids"]
                        vllm_inputs_summarized[i]["summarized_response_ids"] = response_ids
                        vllm_inputs_summarized[i]["summarized"] = True
                    

                    unique_uuid_list = self.unique_uuids_preserve_order(original_uuid_list)

                    final_response = []
                    response_is_summarized = []
                    difficulty_list = []
                    uuid_list = []

                    for uuid in unique_uuid_list:
                        original_responses = [res for res in response if res['uuid'] == uuid and not res['summarized']]
                        summarized_responses = [res for res in vllm_inputs_summarized if res['uuid'] == uuid and res['summarized']]
                        all_responses = original_responses + summarized_responses
                        random.shuffle(all_responses)
                        selected_responses = all_responses[:rollout_n]
                        for res in selected_responses:
                            if res['summarized']:
                                final_response.append(self.get_complete_summary_rollouts(res))
                            else:
                                final_response.append(res["response_ids"])
                            response_is_summarized.append(res["summarized"])
                            difficulty_list.append(res["difficulty"])
                            uuid_list.append(res["uuid"])


            final_response = pad_2d_list_to_length(final_response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, final_response], dim=-1)

        # breakpoint()
        print(f"Final response length: {final_response.size(1)}")
        response_length = final_response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        if self.summary_mode == 'None' or self.summary_mode == 'attention_weights' or self.summary_mode == 'early_exit_attention_weights' or self.summary_mode == 'early_exit' or self.summary_mode == 'confidence_scores' or self.summary_mode == 'compression_no_difficulty':
            response_attention_mask = get_response_mask(
                response_id=final_response, 
                eos_token=eos_token_id, 
                dtype=attention_mask.dtype
            )
        elif self.summary_mode == 'self_summary':
            response_attention_mask = self.get_custom_response_mask(
                response_id=final_response, 
                eos_token=eos_token_id, 
                dtype=attention_mask.dtype,
                is_summarized_list=response_is_summarized,
            )
        # elif self.summary_mode == 'attention_weights':
        #     response_is_summarized = [not is_summarized for is_summarized in response_is_summarized]
        #     response_attention_mask = self.get_custom_response_mask(
        #         response_id=final_response, 
        #         eos_token=eos_token_id, 
        #         dtype=attention_mask.dtype,
        #         is_summarized_list=response_is_summarized,
        #     )
        else:
            raise ValueError(f"Invalid summary mode: {self.summary_mode}")
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": final_response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "difficulty": torch.tensor(difficulty_list).unsqueeze(1),
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    
    def token_confidence(self, top_logprobs: Dict[str, float]) -> float:
        """
        Compute Ci = -(1/k) * sum_{j=1..k} log P_i(j)
        where P_i(j) is probability of the j-th top token at step i.
        """
        # take top-k tokens (truncate if fewer available)
        # pro
        items = [v.logprob for v in top_logprobs.values()]
        # if token_id in top_logprobs:
        #     items.append(top_logprobs[token_id].logprob)
        # mean_logprob = sum(items) / len(items)
        # return -mean_logprob
        probs = [math.exp(lp) for lp in items]  # logprob -> prob
        Z = sum(probs)
        probs = [p / Z for p in probs]

        # Entropy = -sum(p * log(p))
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)

        # confidence of distribution
        # Max entropy depends on k
        k = len(probs)
        H_max = math.log(k)

        confidence = 1 - (entropy / H_max)
        return confidence

        # Confidence of chosen token = its probability

        # if len(probs) == 0:
        #     return 0.0
        # avg_logprob = sum(math.log(p) for p in probs) / len(probs)
        # return -avg_logprob

    def get_complete_summary_rollouts(self, summarized_response):
        response_ids_with_summarized_thinking = summarized_response['response_ids_with_summarized_thinking']
        summarized_response_ids = summarized_response['summarized_response_ids']
        
        return response_ids_with_summarized_thinking + summarized_response_ids

    def remove_answer_from_think_str(self, think_str, answer):
        import re
        # Remove the last reasoning step if it contains the answer (either as a substring or as \boxed{answer})
        if answer is not None and isinstance(think_str, str) and think_str.strip():
            # Split into reasoning steps (by double newlines)
            steps = think_str.strip().split('\n\n')
            answer_str = str(answer).strip()
            
            # Recursively remove last step if it contains the answer or \boxed{answer}
            while steps:
                last_step = steps[-1]
                boxed_match = re.search(r'\\boxed\{([A-Za-z0-9]+)\}', last_step)
                if (answer_str and answer_str in last_step) or boxed_match:
                    steps = steps[:-1]
                else:
                    break
            think_str = '\n\n'.join(steps)
        return think_str

        

    def get_difficulty_class(self, difficulty):
        if self.summary_mode == 'compression_no_difficulty':
            return "no_difficulty"
        if difficulty == None:
            return "unknown"
        if difficulty >= 0.8:
            difficulty_category = 'easy'
        elif difficulty >= 0.3:
            difficulty_category = 'medium'
        elif difficulty >= 0:
            difficulty_category = 'hard'
        else:
            difficulty_category = 'unknown'
        return difficulty_category
    
    # Modified attention mask calculation
    def get_custom_response_mask(self, response_id, eos_token, dtype, is_summarized_list):
        """
        Create attention mask where:
        - 0 for EOS tokens and everything after
        - 0 for tokens between <think> and </think> tags when response is from summarized prompt
        - 1 for all other tokens (before EOS)
        
        Based on the existing get_response_mask function but with think token masking.
        Assumes only one <think></think> pair per response.
        """
        # First get the standard EOS mask using the existing logic
        eos_mask = torch.isin(response_id, torch.tensor(eos_token, device=response_id.device)).int()
        base_mask = (eos_mask.cumsum(dim=1) - eos_mask).eq(0).to(dtype)
        
        # Get think token IDs
        think_start_token_id = self.tokenizer.encode("<think>", add_special_tokens=False)[0]
        think_end_token_id = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
        
        # Now modify for summarized responses
        batch_size = response_id.shape[0]
        final_mask = base_mask.clone()
        
        for i in range(batch_size):
            if is_summarized_list[i]:
                # Find the single <think> and </think> tokens in this sequence
                think_start_positions = (response_id[i] == think_start_token_id).nonzero(as_tuple=True)[0]
                think_end_positions = (response_id[i] == think_end_token_id).nonzero(as_tuple=True)[0]
                
                if len(think_start_positions) > 0 and len(think_end_positions) > 0:
                    start_pos = think_start_positions[0]
                    end_pos = think_end_positions[0]
                    
                    # Set attention mask to 0 for tokens from <think> to </think> (inclusive)
                    final_mask[i, start_pos:end_pos+1] = 0
        
        return final_mask


    def unique_uuids_preserve_order(self, uuid_list):
        seen = set()
        unique_list = []
        for u in uuid_list:
            # If input is string, convert to UUID object for consistency
            uuid_obj = UUID(str(u))
            if uuid_obj not in seen:
                seen.add(uuid_obj)
                unique_list.append(str(uuid_obj))  # or append uuid_obj if you want UUID objects
        return unique_list
    def check_format(self, model_path: str, response_ids: list[int], confidence_scores: list[tuple[float, str]]):
        """
        Checks if the response is in the correct format and returns the <think>...</think> span,
        along with the trimmed confidence scores corresponding to that span.
        The confidence_scores is a list of tuples: (value, [response_ids[i]])
        The second element is used as a signal to trim.
        """
        response_str = self.tokenizer.decode(response_ids)
        import re
        # Helper to trim confidence_scores using the token string signal
        def trim_confidence_scores_by_token_string(confidence_scores, start_token, end_token):
            start_idx = None
            end_idx = None
            for idx, (_, token_id) in enumerate(confidence_scores):
                token_str = self.tokenizer.convert_ids_to_tokens(token_id)
                if start_idx is None and token_str == start_token:
                    start_idx = idx
                if start_idx is not None and token_str == end_token:
                    end_idx = idx
                    break
            if start_idx is not None and end_idx is not None:
                return confidence_scores[start_idx:end_idx]
            else:
                return confidence_scores

        # Qwen models: <think>...</think>
        if model_path in ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]:
            pattern = r"<think>(.*?)</think>"
            match = re.findall(pattern, response_str, re.DOTALL)
            if match and len(match) == 1:
                # Use the second element of confidence_scores to trim between <think> and </think>
                think_start_token = "<think>"
                think_end_token = "</think>"
                trimmed_confidence_scores = trim_confidence_scores_by_token_string(
                    confidence_scores,
                    think_start_token,
                    think_end_token
                )
                return True, match[0], trimmed_confidence_scores
            else:
                return False, None, []

        # DeepSeek models: (.*?)</think>
        elif model_path in [
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        ]:
            pattern = r"(.*?)</think>"
            match = re.findall(pattern, response_str, re.DOTALL)
            if match and len(match) == 1:
                # Use the second element of confidence_scores to trim up to </think>
                think_end_token = "</think>"
                # For DeepSeek, we want everything up to and including the first </think>
                start_idx = 0
                end_idx = None
                for idx, (_, token_id) in enumerate(confidence_scores):
                    token_str = self.tokenizer.convert_ids_to_tokens(token_id)
                    if token_str == think_end_token:
                        end_idx = idx
                        break
                if end_idx is not None:
                    trimmed_confidence_scores = confidence_scores[start_idx:end_idx+1]
                else:
                    trimmed_confidence_scores = confidence_scores
                return True, match[0], trimmed_confidence_scores
            else:
                return False, None, []

        else:
            raise ValueError(f"Model path {model_path} not supported")
    
    def get_summary_prompt(self, think_input) -> str:
        think_str = think_input['think_str']
        difficulty_category = think_input['difficulty_category']
        
        print(f"Difficulty category: {difficulty_category}")
        
        summarizationPrompt = "You will get a reasoning chain that may have unnecessary details, repeated steps, or occasional backtracking. Your task is to shorten and clean it up by removing parts that do not support the main logic or conclusion. Keep only backtracking or repeated steps if they help clarify or double-check important points. Do not add new ideas or change the meaning. Try to keep the overall structure and logical flow, but provide enough explanation to make the reasoning clear and complete. Aim for a balance between brevity and clarity."

        if difficulty_category == 'easy':
            # summarizationPrompt = (
            #     "You are an expert at understanding reasoning steps. Your task is to condense a complex, multi-step reasoning process into a clear summary, preserving all key logical steps, nuanced arguments, and critical insights.\n\n"
            #     "Guidelines:\n"
            #     "Retain the reasoning structure—capture important intermediate steps and conclusions, while omitting unnecessary details.\n"
            #     "Omit detailed steps and explanations unless they are essential for understanding the main idea.\n"
            #     "Keep the summary short and simple.\n"
            #     "The summary should be brief.\n"
            # )
            summarizationPrompt = "You will get a reasoning chain that may have unnecessary details, repeated steps, or occasional backtracking. Your task is to shorten and clean it up by removing parts that do not support the main logic or conclusion. Keep only backtracking or repeated steps if they help clarify or double-check important points. Do not add new ideas or change the meaning. Try to keep the overall structure and logical flow, but provide enough explanation to make the reasoning clear and complete. Aim for a balance between brevity and clarity."
        elif difficulty_category == 'medium':
            # summarizationPrompt = (
            #     "You are an expert at understanding reasoning steps. Your task is to condense a complex, multi-step reasoning process into a clear summary, preserving all key logical steps, nuanced arguments, and critical insights.\n\n"
            #     "Guidelines:\n"
            #     "Retain the reasoning structure—capture important intermediate steps and conclusions.\n"
            #     "Compress the explanation by rephrasing and omitting minor details, but do not lose important content.\n"
            #     "Make sure the summary remains clear and faithful to the original logic.\n"
            #     "The summary should be moderately detailed.\n"
            # )
            summarizationPrompt = "You will get a reasoning chain that may have unnecessary details, repeated steps, or occasional backtracking. Your task is to shorten and clean it up by removing parts that do not support the main logic or conclusion. Keep only backtracking or repeated steps if they help clarify or double-check important points. Do not add new ideas or change the meaning. Try to keep the overall structure and logical flow, but provide enough explanation to make the reasoning clear and complete. Aim for a balance between brevity and clarity."

        elif difficulty_category == 'hard':
            # summarizationPrompt = (
            #     "You are an expert at understanding reasoning steps. Your task is to condense a complex, multi-step reasoning process into a clear summary, preserving all key logical steps, nuanced arguments, and critical insights.\n\n"
            #     "Guidelines:\n"
            #     "Carefully retain the full structure of the reasoning, including important intermediate steps, assumptions, and conclusions.\n"
            #     "Highlight subtle distinctions, exceptions, or caveats that are crucial to the argument’s validity.\n"
            #     "Summarize with maximum precision—avoid oversimplification or omission of any detail that could alter the original meaning.\n"
            #     "The summary should be detailed and comprehensive.\n"
            # )
            summarizationPrompt = "You will get a reasoning chain that may have unnecessary details, repeated steps, or occasional backtracking. Your task is to shorten and clean it up by removing parts that do not support the main logic or conclusion. Keep only backtracking or repeated steps if they help clarify or double-check important points. Do not add new ideas or change the meaning. Try to keep the overall structure and logical flow, but provide enough explanation to make the reasoning clear and complete. Aim for a balance between brevity and clarity."
        elif difficulty_category == 'unknown':        
            # summarizationPrompt = (
            #     "You are a helpful and concise reasoning summarization assistant. Your task is to summarize a long chain of reasoning into a concise summary, while preserving the logical flow and key insights.\n\n"
            #     "Guidelines:\n"
            #     "Retain the reasoning structure—capture important intermediate steps and conclusions.\n"
            #     "Be concise—rephrase and compress wherever possible without losing critical content.\n"
            #     "Avoid omitting essential details that change the meaning or validity of the reasoning.\n"
            # )
            summarizationPrompt = "You will get a reasoning chain that may have unnecessary details, repeated steps, or occasional backtracking. Your task is to shorten and clean it up by removing parts that do not support the main logic or conclusion. Keep only backtracking or repeated steps if they help clarify or double-check important points. Do not add new ideas or change the meaning. Try to keep the overall structure and logical flow, but provide enough explanation to make the reasoning clear and complete. Aim for a balance between brevity and clarity."
        else:
            raise ValueError(f"Difficulty category {difficulty_category} not supported")
        if self.model_path == "Qwen/Qwen3-4B" or self.model_path == "Qwen/Qwen3-8B":
            message = [
                {"role": "system", "content": summarizationPrompt},
                {"role": "user", "content": think_str}
            ]
            # breakpoint()
            prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            tokenized_prompt = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            return prompt, tokenized_prompt
        else:
            raise ValueError(f"Model path {self.model_path} not supported")

    def summarize_confidence_scores(self, summary_inputs) -> List[str]:
        summary_outputs = []
        for input in tqdm(summary_inputs):
            
            text = input['think_str']
            difficulty_category = input['difficulty_category']
            confidence_scores = input['confidence_scores']
            reduction_score = self.get_reduction_score(difficulty_category)

            # for i in range(len(input_ids)):
                # print(self.tokenizer.convert_ids_to_tokens([input_ids[i]]), confidence_scores[i][1])

            # breakpoint()
            confidence_scores = confidence_scores[:-1]

            input_ids = [score[1] for score in confidence_scores]
            text = self.tokenizer.decode(input_ids)
            confidence_scores = [score[0] for score in confidence_scores]

            try:
                steps_to_evict, new_reasoning_chain = redundant_token_eviction(
                        reasoning_chain=text,
                        confidence_scores=confidence_scores,
                        tokenizer=self.tokenizer,
                        input_ids=input_ids,
                        target_reduction=reduction_score
                    )   
            except Exception as e:
                print(f"Error: {e}")
                new_reasoning_chain = text
                
            summary_outputs.append(new_reasoning_chain)
        return summary_outputs

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((
            requests.exceptions.RequestException, 
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
            ValueError
        ))
    )
    def _make_api_request(self, payload: dict) -> str:
        """Make API request with retry mechanism for individual calls."""
        BASE_URL = "http://localhost:8008"
        API_ENDPOINTS = {
            "health": f"{BASE_URL}/health",
            "get_reduced_reasoning_chain": f"{BASE_URL}/get_reduced_reasoning_chain",
        }
        
        response = requests.post(API_ENDPOINTS["get_reduced_reasoning_chain"], json=payload)
        response.raise_for_status()
        
        if response.status_code == 200:
            data = response.json()
            return data['new_reasoning_chain']
        else:
            raise ValueError(f"Failed to get model output: {response.status_code}")

    def summarize_attention_weights(self, summary_inputs) -> List[str]:
        if self.model_path == "Qwen/Qwen3-4B" or self.model_path == "Qwen/Qwen3-8B":
            summary_outputs = []
            for input in tqdm(summary_inputs):
                text = input['think_str']
                difficulty_category = input['difficulty_category']
                print(f"Difficulty category: {difficulty_category}")
                reduction_score = self.get_reduction_score(difficulty_category)
                print(f"Reduction score: {reduction_score}")
                
                text = "<think>" + text + "Time is up. I should stop thinking and now write a summary containing all key steps required to solve the problem.</think>"
                
                payload = {
                    "text": text,
                    "reduction_score": reduction_score
                }
                
                try:
                    new_reasoning_chain = self._make_api_request(payload)
                    
                    if "Time is up. I should stop thinking and now write a summary containing all key steps required to solve the problem." in new_reasoning_chain:
                        new_reasoning_chain = new_reasoning_chain.replace("Time is up. I should stop thinking and now write a summary containing all key steps required to solve the problem.", "")
                    # if '<think>' in new_reasoning_chain:
                    #     new_reasoning_chain = new_reasoning_chain.replace('<think>', '')
                    # if '</think>' in new_reasoning_chain:
                    #     new_reasoning_chain = new_reasoning_chain.replace('</think>', '')

                    summary_outputs.append(new_reasoning_chain)
                except Exception as e:
                    print(f"Error processing input: {e}")
                    raise ValueError(f"Failed to get model output")
        elif self.model_path == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" or self.model_path == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":
            summary_outputs = []
            for input in tqdm(summary_inputs):
                text = input['think_str']
                difficulty_category = input['difficulty_category']
                reduction_score = self.get_reduction_score(difficulty_category)

                text = "<think>\n" + text + "Time is up. I should stop thinking and now write a summary containing all key steps required to solve the problem.</think>"
                
                payload = {
                    "text": text,
                    "reduction_score": reduction_score
                }
                
                try:
                    new_reasoning_chain = self._make_api_request(payload)
                    
                    if "Time is up. I should stop thinking and now write a summary containing all key steps required to solve the problem." in new_reasoning_chain:
                        new_reasoning_chain = new_reasoning_chain.replace("Time is up. I should stop thinking and now write a summary containing all key steps required to solve the problem.", "")
                    if '<think>\n' in new_reasoning_chain:
                        new_reasoning_chain = new_reasoning_chain.replace('<think>\n', '')
                    # if '</think>' in new_reasoning_chain:
                    #     new_reasoning_chain = new_reasoning_chain.replace('</think>', '')

                    summary_outputs.append(new_reasoning_chain)
                except Exception as e:
                    print(f"Error processing input: {e}")
                    raise ValueError(f"Failed to get model output")
                
        else:
            raise ValueError(f"Model path {self.model_path} not supported")
        return summary_outputs
    
    def summarize_early_exit_attention_weights(self, summary_inputs) -> List[str]:
        if self.model_path == "Qwen/Qwen3-4B" or self.model_path == "Qwen/Qwen3-8B":
            summary_outputs = []
            for input in tqdm(summary_inputs):
                text = input['think_str']
                difficulty_category = input['difficulty_category']
                reduction_score = self.get_reduction_score(difficulty_category)

                ground_truth = input['reward_model']['ground_truth']

                first_correct_answer_chain, remaining_answer_chain = self.get_first_correct_answer_chain(text, ground_truth)

                if self.summary_mode == 'early_exit':
                    if first_correct_answer_chain == "":
                        summary_outputs.append(remaining_answer_chain)
                    else:
                        summary_outputs.append(first_correct_answer_chain)
                elif self.summary_mode == 'early_exit_attention_weights':

                    text = "<think>" + remaining_answer_chain + "Time is up. I should stop thinking and now write a summary containing all key steps required to solve the problem.</think>"
                    # API configuration
                    BASE_URL = "http://localhost:8008"
                    API_ENDPOINTS = {
                        "health": f"{BASE_URL}/health",
                        "get_reduced_reasoning_chain": f"{BASE_URL}/get_reduced_reasoning_chain",
                    }
                    payload = {
                        "text": text,
                        "reduction_score": reduction_score
                    }
                    response = requests.post(API_ENDPOINTS["get_reduced_reasoning_chain"], json=payload)

                    if response.status_code == 200:
                        data = response.json()

                        new_reasoning_chain = data['new_reasoning_chain']
                        if "Time is up. I should stop thinking and now write a summary containing all key steps required to solve the problem." in new_reasoning_chain:
                            new_reasoning_chain = new_reasoning_chain.replace("Time is up. I should stop thinking and now write a summary containing all key steps required to solve the problem.", "")
                        if '<think>' in new_reasoning_chain:
                            new_reasoning_chain = new_reasoning_chain.replace('<think>', '')
                        if '</think>' in new_reasoning_chain:
                            new_reasoning_chain = new_reasoning_chain.replace('</think>', '')

                        new_reasoning_chain = "<think>" + first_correct_answer_chain + new_reasoning_chain + "</think>"
                        summary_outputs.append(new_reasoning_chain)
                    else:
                        raise ValueError(f"Failed to get model output: {response.status_code}")

        elif self.model_path == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" or self.model_path == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":
            summary_outputs = []
            for input in tqdm(summary_inputs):
                text = input['think_str']
                difficulty_category = input['difficulty_category']
                reduction_score = self.get_reduction_score(difficulty_category)
                
                ground_truth = input['reward_model']['ground_truth']

                first_correct_answer_chain, remaining_answer_chain = self.get_first_correct_answer_chain(text, ground_truth)

                if self.summary_mode == 'early_exit':
                    if first_correct_answer_chain == "":
                        summary_outputs.append(remaining_answer_chain)
                    else:
                        summary_outputs.append(first_correct_answer_chain)
                elif self.summary_mode == 'early_exit_attention_weights':
                    text = "<think>\n" + remaining_answer_chain + "Time is up. I should stop thinking and now write a summary containing all key steps required to solve the problem.</think>"
                    
                    BASE_URL = "http://localhost:8008"
                    API_ENDPOINTS = {
                        "health": f"{BASE_URL}/health",
                        "get_reduced_reasoning_chain": f"{BASE_URL}/get_reduced_reasoning_chain",
                    }
                    payload = {
                        "text": text,
                        "reduction_score": reduction_score
                    }
                    response = requests.post(API_ENDPOINTS["get_reduced_reasoning_chain"], json=payload)
                    
                    
                    if response.status_code == 200:
                        data = response.json()

                        new_reasoning_chain = data['new_reasoning_chain']
                        if "Time is up. I should stop thinking and now write a summary containing all key steps required to solve the problem." in new_reasoning_chain:
                            new_reasoning_chain = new_reasoning_chain.replace("Time is up. I should stop thinking and now write a summary containing all key steps required to solve the problem.", "")
                        if '<think>\n' in new_reasoning_chain:
                            new_reasoning_chain = new_reasoning_chain.replace('<think>\n', '')
                        if '</think>' in new_reasoning_chain:
                            new_reasoning_chain = new_reasoning_chain.replace('</think>', '')

                        new_reasoning_chain = "<think>\n" + first_correct_answer_chain + new_reasoning_chain + "</think>"
                        summary_outputs.append(new_reasoning_chain)
                    else:
                        raise ValueError(f"Failed to get model output: {response.status_code}")
        else:
            raise ValueError(f"Model path {self.model_path} not supported")
        return summary_outputs
            

    def get_first_correct_answer_chain(self, text: str, ground_truth: str):
        """
        Splits the reasoning chain into chunks, finds the first occurrence where the generated answer matches the ground truth,
        and returns the reasoning up to and including that chunk as the first part, and the rest as the second part.

        Args:
            text (str): The full reasoning chain, with chunks separated by double newlines.
            ground_truth (str): The ground truth answer.

        Returns:
            Tuple[str, str]: (first_correct_answer_chain, remaining_answer_chain)
        """
        split_text = text.split('\n\n')
        first_correct_answer_chain = []
        found = False
        ground_truth_answer = parse(ground_truth)
        for i, chunk in enumerate(split_text):
            generated_answer = parse(chunk)
            first_correct_answer_chain.append(chunk)
            if verify(ground_truth_answer, generated_answer):
                found = True
                break
        if found:
            # Chunks up to and including the first correct answer
            first_half = "\n\n".join(first_correct_answer_chain)
            # Remaining chunks after the first correct answer
            remaining_half = "\n\n".join(split_text[i+1:])
        else:
            # If no correct answer found, all reasoning is in the second half, first half is empty
            first_half = ""
            remaining_half = "\n\n".join(split_text)
        return first_half, remaining_half

    def get_reduction_score(self, difficulty_category: str) -> float:
        if difficulty_category == 'no_difficulty':
            return 0.6
        if difficulty_category == 'easy':
            return 0.6
        elif difficulty_category == 'medium':
            return 0.4
        elif difficulty_category == 'hard':
            return 0.2
        
    def summarize_think(self, summary_inputs) -> List[str]:
        if self.model_path == "Qwen/Qwen3-4B" or self.model_path == "Qwen/Qwen3-8B":
            prompts = []
            tokenized_prompts = []
            for input in summary_inputs:
                prompt, tokenized_prompt = self.get_summary_prompt(input)
                prompts.append(prompt)
                tokenized_prompts.append(tokenized_prompt)
            
            if self.model_path == "Qwen/Qwen3-4B" or self.model_path == "Qwen/Qwen3-8B":
                max_tokens = 40960 - max([len(tokenized_prompt) for tokenized_prompt in tokenized_prompts])
                max_tokens = min(10000, max_tokens)
            else:
                raise ValueError(f"Model path {self.model_path} not supported")
            print(f"Max tokens: {max_tokens} for summarization")
            summary_outputs = self.query_vllm_summary_batch(prompts, max_tokens)
            summary_outputs = [summary_outputs[i]["text"].strip() for i, input in enumerate(summary_inputs)]
            return summary_outputs
        else:
            raise ValueError(f"Model path {self.model_path} not supported")
            
    def convert_to_torch_format(self, serialized_data: dict) -> dict:
        """
        Convert serialized data back to PyTorch tensor format.
        
        Args:
            serialized_data: Dictionary containing serialized input_ids and attention_weights
            
        Returns:
            Dictionary with PyTorch tensors
        """
        try:
            # Convert input_ids back to tensor
            input_ids = torch.tensor(serialized_data["input_ids"], dtype=torch.long)
            
            # Convert attention weights back to tensors
            attention_weights = []
            for layer_attentions in serialized_data["attention_weights"]:
                layer_tensor = torch.tensor(layer_attentions, dtype=torch.float32)
                attention_weights.append(layer_tensor)
            
            return {
                "input_ids": input_ids,
                "attention_weights": attention_weights
            }
        except Exception as e:
            print(f"Error converting to torch format: {e}")
            raise ValueError(f"Failed to convert data to torch format: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),                          # Reduced from 5 to 3 attempts
        wait=wait_exponential(multiplier=2, min=2, max=30),  # Wait: 2s, 4s, 8s, 16s, 30s
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
    )
    def query_vllm_summary_batch(self, prompts: List[str], max_tokens: int):
        """Batch query for vllm summarization endpoint using a single max_tokens value for all"""

        # generated_texts = []
        # for prompt in prompts:
        #     text = prompt.split('<|im_start|>user')[-1].split('<|im_end|>')[0].strip()
        #     generated_texts.append({"text": text[:len(text)//2]})
        
        # return generated_texts

        response = requests.post(
            "http://localhost:8001/v1/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": self.model_path,
                "prompt": prompts,
                "max_tokens": max_tokens,
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 20,
                # "presence_penalty": 1.5,
            },
            timeout=600,  # Increased from 200 to 600 seconds (10 minutes)
        )

        response.raise_for_status()

        return response.json()["choices"]
    
    def create_vllm_summary_inputs(self, prompt_ids: List[int], summarized_think: str, res: dict) -> List[str]:
        prompt_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)

        # clean_summarized_think = self.clean_summarize_think(summarized_think, res)

        if self.model_path == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" or self.model_path == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":
            new_prompt = prompt_str + summarized_think + "</think>"
        elif self.model_path == "Qwen/Qwen3-4B" or self.model_path == "Qwen/Qwen3-8B":
            new_prompt = prompt_str + "<think>" + summarized_think + "</think>"
        else:
            raise ValueError(f"Model path {self.model_path} not supported")

        new_prompt_ids = self.tokenizer.encode(new_prompt, add_special_tokens=True)

        if self.model_path == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" or self.model_path == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":
            return new_prompt_ids, self.tokenizer.encode(summarized_think + "</think>", add_special_tokens=True)
        elif self.model_path == "Qwen/Qwen3-4B" or self.model_path == "Qwen/Qwen3-8B":
            return new_prompt_ids, self.tokenizer.encode("<think>" + summarized_think + "</think>", add_special_tokens=True)
        else:
            raise ValueError(f"Model path {self.model_path} not supported")
    
    def clean_summarize_think(self, summarized_think: str, res: dict) -> str:

        reward_model = res['reward_model']


        score = getRawCorrectness(summarized_think, reward_model['ground_truth'])

        while score >= 1:
            summarized_think = summarized_think.split('\n\n')
            summarized_think = summarized_think[:-1]
            summarized_think = '\n\n'.join(summarized_think)
            score = getRawCorrectness(summarized_think, reward_model['ground_truth'])

        return summarized_think


        

        
    def create_summarized_response(self, prompt_ids: List[int], summarized_prompt_ids: List[int], summarized_response_ids: List[int]) -> List[int]:
        # remove the overlap part of prompt_ids and summarized_prompt_ids
        # then append the remaining part to the summarized_response_ids

        prompt_ids_non_overlap = summarized_prompt_ids[len(prompt_ids):]
        response_ids = prompt_ids_non_overlap + summarized_response_ids
        return response_ids




# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.tokenizer = tokenizer

        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock("/tmp/verl_vllm_zmq.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        self.loop_thread = threading.Thread(target=self._loop_forever)
        self.loop_thread.start()

        return address

    def _get_free_port(self):
        ip = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    def _loop_forever(self):
        while True:
            message = self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = self.execute_method(method, *args, **kwargs)
            self.socket.send(pickle.dumps(result))

    def get_zeromq_address(self):
        return self.address

    def init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
