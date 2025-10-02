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


def reward_func(data_source, solution_str, ground_truth, model_name, extra_info=None):
    if data_source == 'math-numina' or data_source == 'math-aime' or data_source == 'math-amc' or data_source == 'olympiad' or data_source == 'commonsense' or data_source == 'ARC-Easy' or data_source == 'ARC-Challenge' or data_source == 'gpqa' or data_source == "DAPO-Math-17k":
        from scripts.eval import math_evaluation
        res, length = math_evaluation.compute_score(solution_str, ground_truth, model_name)
        return res, length
    else:
        raise NotImplementedError
