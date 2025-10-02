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

def curate_bbh_dataset():
    huggingface_data = 'lukaemon/bbh'
    data_source = 'bbh'
    categories = ["boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa", "logical_deduction_five_objects", "movie_recommendation", "navigate", "object_counting", "snarks"]
    combined_datasets = []
    data_source_list = []
    for category in categories:
        print(f"Loading the {huggingface_data} dataset from huggingface...", flush=True)
        dataset = datasets.load_dataset("lukaemon/bbh", category)
        dataset = dataset['test']

        instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
        data_source_list.append(data_source + "_" + category)

        # add a row to each data item that represents a unique id
        def make_map_fn(split):
            def process_fn(example, idx):
                question = example.pop("input")

                question = question + " " + instruction_following

                answer = example.pop("target")
                if '(' in answer or ')' in answer:
                    answer = answer.replace('(', '').replace(')', '')

                data = {    
                    "data_source": data_source + "_" + category,
                    "prompt": [{"role": "user", "content": question}],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": answer},
                    "extra_info": {"split": split, "index": idx},
                    "uuid": str(uuid.uuid4()),
                }
                # breakpoint()
                return data

            return process_fn

        dataset = dataset.map(function=make_map_fn("test"), with_indices=True, remove_columns=dataset.column_names)
        combined_datasets.append(dataset)

    dataset = datasets.concatenate_datasets(combined_datasets)
    print(data_source_list)

    return dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./bbh_test_dataset")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    bbh_dataset = curate_bbh_dataset()

    # bbh_dataset = bbh_dataset.shuffle(seed=42).select(range(50))
    # commonsense_dataset = commonsense_dataset.shuffle(seed=42).select(range(50))
    # olympiad_dataset = olympiad_dataset.shuffle(seed=42).select(range(50))
    # arc_easy_dataset = arc_easy_dataset.shuffle(seed=42).select(range(50))
    # arc_challenge_dataset = arc_challenge_dataset.shuffle(seed=42).select(range(50))


    dataset = datasets.concatenate_datasets([bbh_dataset])

    # shuffle the dataset
    dataset = dataset.shuffle(seed=42)

    print("Total length of dataset: ", len(dataset))


    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)