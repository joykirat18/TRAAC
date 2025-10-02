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

from collections import defaultdict
from statistics import median

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
import json
import numpy as np
import os
import math

@register("correctnessOnly")
class CorrectnessOnlyRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, save_path=None, rewardType=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.save_path = save_path
        self.rewardType = rewardType
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.model_name = self.tokenizer.name_or_path

    def compute_standardized_deviation(self, length, mean_length, std_length, epsilon=1e-6):
        return abs(length - mean_length) / (std_length + epsilon)

    def accuracy_reward_distribution(self, length, mean_length, std_length, alpha=0.1, epsilon=1e-6, difficulty_scale=1.0):
        """
        Compute accuracy reward with comprehensive debugging.
        """
        
        # Handle invalid inputs
        if np.isnan(length) or np.isnan(mean_length) or np.isnan(std_length):
            return 1.0
        
        # If std is very small or zero, return maximum reward
        if std_length < epsilon:
            return 1.0
        
        # Compute standardized deviation
        z = abs(length - mean_length) / (std_length + epsilon)
        
        # Compute reward
        reward = np.exp(-1 * z)
        
        
        return difficulty_scale * reward

        
    def get_difficulty_class(self, difficulty):
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

    def get_difficulty_scale(self, difficulty_category):
        if difficulty_category == 'easy':
            difficulty_scale = 1.0
        elif difficulty_category == 'medium':
            difficulty_scale = 0.75
        elif difficulty_category == 'hard':
            difficulty_scale = 0.5
        else:
            difficulty_scale = 0
        return difficulty_scale


    def get_matching_length_list(self, data, uuid):
        """Get the list of lengths of responses that match the valid prompt ids and are correct"""
        matching_lengths = []
        for item in data:
            if item['uuid'] == uuid and item['score']['score'] > 0:
                matching_lengths.append(item['sequence_length'])
        return matching_lengths

    def __call__(self, data: DataProto, curr_save_path=None, return_dict=False, response_length_path=None):
        """We will expand this function gradually based on the available datasets"""

        if curr_save_path is not None:
            save_path = curr_save_path
        else:
            save_path = self.save_path

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # breakpoint()

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_tensor_correctness = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_tensor_format = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_tensor_length = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        if save_path is not None:
            save_file = open(save_path, 'a')


        score_info = []

        difficulty_data_wise_length = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            if 'difficulty' in data_item.batch:
                difficulty = data_item.batch['difficulty'][0].item()
            else:
                difficulty = None
            
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]

            response_attention_mask = data_item.batch["attention_mask"][prompt_length:]
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id

            # Find the actual end of response (EOS or padding)
            valid_response_length = response_ids.shape[-1] 

            # Look for EOS token
            if isinstance(eos_token_id, list):
                eos_positions = []
                for eos_id in eos_token_id:
                    eos_pos = (response_ids == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_pos) > 0:
                        eos_positions.append(eos_pos[0].item())
                if eos_positions:
                    valid_response_length = min(eos_positions) + 1  # Include EOS token
            else:
                eos_positions = (response_ids == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    valid_response_length = eos_positions[0].item() + 1

            # Look for padding token if no EOS found
            if valid_response_length == response_ids.shape[-1]:
                pad_positions = (response_ids == pad_token_id).nonzero(as_tuple=True)[0]
                if len(pad_positions) > 0:
                    valid_response_length = pad_positions[0].item()

            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            uuid = data_item.non_tensor_batch.get("uuid", None)
            extra_info["num_turns"] = num_turns

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                model_name=self.model_name,
            )

            difficulty_category = self.get_difficulty_class(difficulty)

            if f'{data_source}_{difficulty_category}' not in difficulty_data_wise_length:
                difficulty_data_wise_length[f'{data_source}_{difficulty_category}'] = []
            if score['score'] > 0:
                difficulty_data_wise_length[f'{data_source}_{difficulty_category}'].append(valid_response_length)


            score_info.append({'score': score, 'solution_str': response_str, 'ground_truth': ground_truth, 'sequence_length': valid_response_length, 'input_id': valid_prompt_ids, 'difficulty': difficulty, 'uuid': uuid})

        if self.rewardType == 'train':
            for key, value in difficulty_data_wise_length.items():
                if os.path.exists(os.path.join(response_length_path, f"{key}.json")):
                    with open(os.path.join(response_length_path, f"{key}.json"), "r") as f:
                        difficulty_data_wise_length[key] = json.load(f)
                else:
                    difficulty_data_wise_length[key] = []
                difficulty_data_wise_length[key].append(value)
                with open(os.path.join(response_length_path, f"{key}.json"), "w") as f:
                    json.dump(difficulty_data_wise_length[key], f)
        
        for i in range(len(data)):

            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]

            response_attention_mask = data_item.batch["attention_mask"][prompt_length:]
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id

            # Find the actual end of response (EOS or padding)
            valid_response_length = response_ids.shape[-1] 

            # Look for EOS token
            if isinstance(eos_token_id, list):
                eos_positions = []
                for eos_id in eos_token_id:
                    eos_pos = (response_ids == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_pos) > 0:
                        eos_positions.append(eos_pos[0].item())
                if eos_positions:
                    valid_response_length = min(eos_positions) + 1  # Include EOS token
            else:
                eos_positions = (response_ids == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    valid_response_length = eos_positions[0].item() + 1

            # Look for padding token if no EOS found
            if valid_response_length == response_ids.shape[-1]:
                pad_positions = (response_ids == pad_token_id).nonzero(as_tuple=True)[0]
                if len(pad_positions) > 0:
                    valid_response_length = pad_positions[0].item()

            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            scores = score_info[i]

            # L_norm = None
            length = scores['sequence_length']

            correctness_reward = 0
            format_reward = 0
            length_reward = 0

            if self.rewardType == 'val':
                if scores['score']['score'] > 0:
                    correctness_reward = 1
                else:
                    correctness_reward = 0
                reward = correctness_reward + format_reward + length_reward 
                reason = f"score: {scores['score']}"
            elif self.rewardType == 'train':

                correctness_reward = scores['score']['score']
                if correctness_reward > 0:
                    correctness_reward = 1
                else:
                    correctness_reward = 0
                
                reward = correctness_reward
                format_reward = 0
                length_reward = 0
                
                reason = f"score: {scores['score']}"
                for key, value in scores['score'].items():
                    reward_extra_info[key].append(value)


            reward_tensor[i, valid_response_length - 1] = reward
            reward_tensor_correctness[i, valid_response_length - 1] = correctness_reward
            reward_tensor_format[i, valid_response_length - 1] = format_reward
            reward_tensor_length[i, valid_response_length - 1] = length_reward

            if save_path is not None:
                save_json_line = {
                    'data_source': data_source,
                    'prompt': prompt_str,
                    'response': response_str,
                    'ground_truth': ground_truth,
                    'score': scores['score'],
                    'difficulty': scores['difficulty'],
                    'reason': reason,
                    'final_reward': reward,
                    'uuid': scores['uuid'],
                }
                save_file.write(json.dumps(save_json_line, ensure_ascii=False) + '\n')

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "reward_tensor_correctness": reward_tensor_correctness,
                "reward_tensor_format": reward_tensor_format,
                "reward_tensor_length": reward_tensor_length,
            }
        else:
            return reward_tensor, reward_tensor_correctness, reward_tensor_format, reward_tensor_length
