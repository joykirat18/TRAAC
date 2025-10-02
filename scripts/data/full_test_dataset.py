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

def curate_aime_dataset():
    huggingface_data = 'AI-MO/aimo-validation-aime'
    data_source = 'math-aime'
    print(f"Loading the {huggingface_data} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(huggingface_data, trust_remote_code=True)
    dataset = dataset['train']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")

            question = question + " " + instruction_following

            answer = example.pop("solution")
            try:
                solution = extract_solution(answer)
            except Exception as e:
                print(f"Error in extracting answer: {e}")
                return None
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
                "uuid": str(uuid.uuid4()),
            }
            # breakpoint()
            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn("test"), with_indices=True, remove_columns=dataset.column_names)

    return dataset

def curate_olympiad_dataset():
    huggingface_data = 'Hothan/OlympiadBench'
    data_source = 'olympiad'
    print(f"Loading the {huggingface_data} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(huggingface_data, "OE_TO_maths_en_COMP")
    dataset = dataset['train']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question")

            question = question + " " + instruction_following

            answer = example.pop("final_answer")

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer[0]},
                "extra_info": {"split": split, "index": idx},
                "uuid": str(uuid.uuid4()),
            }
            # breakpoint()
            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn("test"), with_indices=True, remove_columns=dataset.column_names)

    return dataset

def curate_amc_dataset():
    huggingface_data = 'AI-MO/aimo-validation-amc'
    data_source = 'math-amc'
    print(f"Loading the {huggingface_data} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(huggingface_data, trust_remote_code=True)
    dataset = dataset['train']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")

            question = question + " " + instruction_following

            answer = example.pop("answer")
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(answer)},
                "extra_info": {"split": split, "index": idx},
                "uuid": str(uuid.uuid4()),
            }
            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn("test"), with_indices=True, remove_columns=dataset.column_names) 

    return dataset


def curate_commonsense_dataset():
    huggingface_data = 'tau/commonsense_qa'
    data_source = 'commonsense'
    print(f"Loading the {huggingface_data} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(huggingface_data)

    dataset = dataset['validation']

    # TEMPLATE = (
    #     # "Answer the following multiple choice question. The last line of your response should be of the following "
    #     # "format: 'Answer: $LETTER' (without quotes) where LETTER is one of {letter}. Think step by step before "
    #     "{Question}\n\n{Choices}\n\nLet's think step by step and output the final answer within \\boxed{}."
    # )
    TEMPLATE = "{Question}\n\n{Choices}"


    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question")
            choices = example.pop("choices")
            number_of_choices = len(choices['text'])
            if 2 <= number_of_choices <= 26:
                letter = ''.join([chr(ord('A') + i) for i in range(number_of_choices)])
            else:
                print("number of choices ", number_of_choices, "not in supported range (2-26)")
                return None


            choices = [f"{choices['label'][i]}) {choices['text'][i]}" for i in range(number_of_choices)]
            choices = "\n".join(choices)

            question = TEMPLATE.format(Question=question, Choices=choices) + "\n\nLet's think step by step and output the final answer within \\boxed{}."

            answerKey = example.pop("answerKey")
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answerKey},
                "extra_info": {"split": split, "index": idx},
                "uuid": str(uuid.uuid4()),
            }
            # only keep entries in 

            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn("test"), with_indices=True, remove_columns=dataset.column_names)

    return dataset

def curate_gpqa_dataset():
    huggingface_data = 'Idavidrein/gpqa'
    data_source = 'gpqa'
    print(f"Loading the {huggingface_data} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(huggingface_data, "gpqa_diamond")

    dataset = dataset['train']

    # GPQA_QUERY_TEMPLATE = (
    #     # "Answer the following multiple choice question. The last line of your response should be of the following "
    #     # "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before "
    #     "{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}\n\nLet's think step by step and output the final answer within \\boxed{}."
    # )
    GPQA_QUERY_TEMPLATE = "{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"


    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):

            choices = [example["Incorrect Answer 1"], example["Incorrect Answer 2"], example["Incorrect Answer 3"]]
            random.shuffle(choices)
            gold_index = random.randint(0, 3)
            choices.insert(gold_index, example["Correct Answer"])
            query_prompt = GPQA_QUERY_TEMPLATE.format(
                A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=example["Question"]
            ) + "\n\nLet's think step by step and output the final answer within \\boxed{}."

            gold_choice = "ABCD"[gold_index]

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": query_prompt}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": gold_choice},
                "extra_info": {"split": split, "index": idx},
                "uuid": str(uuid.uuid4()),
            }

            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn("test"), with_indices=True, remove_columns=dataset.column_names)

    return dataset

def curate_arc_dataset(data_source):
    huggingface_data = 'allenai/ai2_arc'
    print(f"Loading the {huggingface_data} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(huggingface_data, data_source)
    # combine dataset
    dataset = dataset['test']

    TEMPLATE = (
        "Answer the following multiple choice question. The last line of your response should be of the following "
        "format: 'Answer: $LETTER' (without quotes) where LETTER is one of {letter}. Think step by step before "
        "answering.\n\n{Question}\n\n{Choices}"
    )


    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question")
            choices = example.pop("choices")
            number_of_choices = len(choices['text'])
            if 2 <= number_of_choices <= 26:
                letter = ''.join([chr(ord('A') + i) for i in range(number_of_choices)])
            else:
                print("number of choices ", number_of_choices, "not in supported range (2-26)")
                return None


            choices = [f"{choices['label'][i]}) {choices['text'][i]}" for i in range(number_of_choices)]
            choices = "\n".join(choices)

            question = TEMPLATE.format(letter=letter, Question=question, Choices=choices)

            answerKey = example.pop("answerKey")
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answerKey},
                "extra_info": {"split": split, "index": idx},
                "uuid": str(uuid.uuid4()),
            }

            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn("test"), with_indices=True, remove_columns=dataset.column_names)

    return dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./full_test_dataset")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    aime_dataset = curate_aime_dataset()
    amc_dataset = curate_amc_dataset()
    # commonsense_dataset = curate_commonsense_dataset()
    gpqa_dataset = curate_gpqa_dataset()
    # olympiad_dataset = curate_olympiad_dataset()
    # arc_easy_dataset = curate_arc_dataset('ARC-Easy')
    # arc_challenge_dataset = curate_arc_dataset('ARC-Challenge')

    # aime_dataset = aime_dataset.shuffle(seed=42).select(range(50))
    # amc_dataset = amc_dataset.shuffle(seed=42).select(range(50))
    # commonsense_dataset = commonsense_dataset.shuffle(seed=42).select(range(50))
    # gpqa_dataset = gpqa_dataset.shuffle(seed=42).select(range(50))
    # olympiad_dataset = olympiad_dataset.shuffle(seed=42).select(range(50))
    # arc_easy_dataset = arc_easy_dataset.shuffle(seed=42).select(range(50))
    # arc_challenge_dataset = arc_challenge_dataset.shuffle(seed=42).select(range(50))


    dataset = datasets.concatenate_datasets([amc_dataset, aime_dataset, gpqa_dataset])

    # shuffle the dataset
    dataset = dataset.shuffle(seed=42)

    print("Total length of dataset: ", len(dataset))


    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)