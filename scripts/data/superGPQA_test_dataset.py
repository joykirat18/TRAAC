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

def curate_superGPQA_dataset(difficulty):
    huggingface_data = 'm-a-p/SuperGPQA'
    data_source = f'superGPQA_{difficulty}'
    print(f"Loading the {huggingface_data} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(huggingface_data, trust_remote_code=True)
    dataset = dataset['train']
    ## filter by difficulty
    dataset = dataset.filter(lambda x: x['difficulty'] == difficulty)
    disciplines = set(dataset['discipline'])
    combined_datasets = []
    for discipline in disciplines:
        filtered_dataset = dataset.filter(lambda x: x['discipline'] == discipline)
        print(f"length of dataset for discipline: {discipline}, {len(filtered_dataset)}")
        if len(filtered_dataset) >= 50:
            filtered_dataset = filtered_dataset.shuffle(seed=42).select(range(50))
        combined_datasets.append(filtered_dataset)
        print(f"Length of dataset after filtering: {len(filtered_dataset)}")
    dataset = datasets.concatenate_datasets(combined_datasets)

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question") + "\n"
            options = example.pop("options")
            for i, option in enumerate(options):
                ## a,b,c,d,e,f,g,h,i,j for each options
                question = question + f"({chr(65+i)}) {option}\n"

            question = question + instruction_following

            answer_letter = example.pop("answer_letter")
            answer = example.pop("answer")

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer_letter, "answer": answer},
                "extra_info": {"split": split, "index": idx},
                "uuid": str(uuid.uuid4()),
            }
            # breakpoint()
            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn("test"), with_indices=True, remove_columns=dataset.column_names)

    return dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./superGPQA_test_dataset")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    superGPQA_easy_dataset = curate_superGPQA_dataset('easy')
    superGPQA_medium_dataset = curate_superGPQA_dataset('middle')
    superGPQA_hard_dataset = curate_superGPQA_dataset('hard')


    dataset = datasets.concatenate_datasets([superGPQA_easy_dataset, superGPQA_medium_dataset, superGPQA_hard_dataset])

    # shuffle the dataset
    dataset = dataset.shuffle(seed=42)

    print("Total length of dataset: ", len(dataset))


    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)