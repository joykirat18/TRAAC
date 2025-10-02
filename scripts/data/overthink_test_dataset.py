# MATH: https://huggingface.co/datasets/NovaSky-AI/labeled_numina_difficulty_162K


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
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os
import random
import uuid



import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


def curate_overthink_question_dataset():
    import json
    with open('scripts/data/optimalThinkingBench/filtered_overthink_bench.json', 'r') as f:
        data = json.load(f)
    
    for item in data:
        for key, value in item.items():
            if isinstance(value, int):
                item[key] = str(value)
            elif isinstance(value, float):
                item[key] = str(value)
    
    ## convert to dataset
    dataset = datasets.Dataset.from_list(data)

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("prompt")

            

            answer = example.pop("answer")
            mode = example.pop('mode')
            if mode == 'mcq':
                correct_option = example.pop('correct_option')
                if correct_option == '0':
                    answer = 'A'
                elif correct_option == '1':
                    answer = 'B'
                elif correct_option == '2':
                    answer = 'C'
                elif correct_option == '3':
                    answer = 'D'
                else:
                    raise ValueError(f"Invalid correct option: {correct_option}")
                question = question + " \n" + instruction_following
            else:
                question = question + " " + instruction_following

            data = {
                "data_source": 'overthink-bench',
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(answer)},
                "extra_info": {"split": split, "index": idx},
                "uuid": str(uuid.uuid4()),
            }
            # breakpoint()
            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn("test"), with_indices=True, remove_columns=dataset.column_names) 

    return dataset

# def curate_mip_insufficient_dataset():
#     import json
#     with open('MiP-Overthinking/combined_insufficient_dataset.json', 'r') as f:
#         data = json.load(f)
    
#     ## convert to dataset
#     dataset = datasets.Dataset.from_list(data)

#     instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

#     # add a row to each data item that represents a unique id
#     def make_map_fn(split):
#         def process_fn(example, idx):
#             question = example.pop("insufficient_question")

#             question = question + " " + instruction_following

#             answer = example.pop("answer")
#             data = {
#                 "data_source": 'mip-insufficient',
#                 "prompt": [{"role": "user", "content": question}],
#                 "ability": "math",
#                 "reward_model": {"style": "rule", "ground_truth": str(answer)},
#                 "extra_info": {"split": split, "index": idx},
#                 "uuid": str(uuid.uuid4()),
#             }
#             return data

#         return process_fn

#     dataset = dataset.map(function=make_map_fn("test"), with_indices=True, remove_columns=dataset.column_names) 

#     return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./overthink_test_dataset")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    overthink_question_dataset = curate_overthink_question_dataset()
    # mip_insufficient_dataset = curate_mip_insufficient_dataset()


    dataset = datasets.concatenate_datasets([overthink_question_dataset])

    # shuffle the dataset
    dataset = dataset.shuffle(seed=42)

    print("Total length of dataset: ", len(dataset))


    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)