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

from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

def getCorrectness(model_name: str, model_output: str, ground_truth: str) -> bool:
    if model_name == "Qwen/Qwen3-4B" or model_name == "Qwen/Qwen3-8B":
        if '</think>' in model_output:
            model_output = model_output.split('</think>')[-1]
    else:
        raise ValueError(f"Model name {model_name} not supported")

    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass

    return ret_score

def get_soft_format_score(model_name: str, model_output:str) -> float:
    if model_name == "Qwen/Qwen3-4B" or model_name == "Qwen/Qwen3-8B":
        if model_output.count('<think>') == 1 and model_output.count('</think>') == 1:
            return 0.25
        else:
            return 0.0
    else:
        raise ValueError(f"Model name {model_name} not supported")

def get_hard_format_score(model_name: str, model_output:str) -> float:
    if model_name == "Qwen/Qwen3-4B" or model_name == "Qwen/Qwen3-8B":
        current_pos = 0
        think_pos = model_output.find('<think>', current_pos)
        if think_pos == -1:
            return 0.0
        current_pos = think_pos + len('<think>')
        think_end_pos = model_output.find('</think>', current_pos)
        if think_end_pos == -1:
            return 0.0
        if think_pos > think_end_pos:
            return 0.0
        return 0.25
    else:
        raise ValueError(f"Model name {model_name} not supported")
        

def compute_score(model_output: str, ground_truth: str, model_name: str, timeout_score: float = 0) -> bool:

    correctness = getCorrectness(model_name, model_output, ground_truth)

    length = len(model_output)
    

    return correctness, length

    
