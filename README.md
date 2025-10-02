# [Think Right: Learning to Mitigate Under-Over Thinking via Adaptive, Attentive Compression]()

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Joykirat Singh](https://joykirat18.github.io/) | [Justin Chih-Yao Chen](https://dinobby.github.io/) | [Archiki Prasad](https://archiki.github.io/) | [Elias Stengel-Eskin](https://esteng.github.io/) | [Akshay Nambi](https://www.microsoft.com/en-us/research/people/akshayn/) | [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

<!-- ## Abstract -->
<!-- Recent thinking models are capable of solving complex reasoning tasks by scaling test-time compute across various domains, but this scaling must be allocated in line with task difficulty. On one hand, short reasoning (underthinking) leads to errors on harder problems that require extended reasoning steps; but, excessively long reasoning (overthinking) can be token-inefficient, generating unnecessary steps even after reaching a correct intermediate solution. We refer to this as under-adaptivity, where the model fails to modulate its response length appropriately given problems of varying difficulty. To address under-adaptivity and strike a balance between under- and overthinking, we propose TRAAC (Think Right with Adaptive, Attentive Compression), an online post-training RL method that leverages the model’s self-attention over a long reasoning trajectory to identify important steps and prune redundant ones. TRAAC also estimates difficulty and incorporates it into training rewards, thereby learning to allocate reasoning budget commensurate with example difficulty. Our approach improves accuracy, reduces reasoning steps, and enables adaptive thinking compared to base models and other RL baselines. Across a variety of tasks (AIME, AMC, GPQA-D, BBEH), TRAAC (Qwen3-4B) achieves an average absolute accuracy gain of 8.4% with a relative reduction in reasoning length of 36.8% compared to the base model, and a 7.9% accuracy gain paired with a 29.4% length drop compared to the best RL baseline. TRAAC also shows strong generalization: although our models are trained on math datasets, they show accuracy and efficiency gains on out-of-distribution non-math datasets like GPQA-D, BBEH, and OptimalThinkingBench. Our analysis further verifies that TRAAC provides fine-grained adjustments to thinking budget based on difficulty and that a combination of task-difficulty calibration and attention-based compression yields gains across diverse tasks. -->


## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Download Models](#download-models)
- [Training Data & Evaluation](#-training-data--evaluation)
- [Run Evaluations](#run-evaluations)
- [Train Models](#train-models)
- [Citation](#citation)

## Overview
This repository contains the implementation of TRAAC (Think Right with Adaptive, Attentive Compression), an online post-training RL method that leverages the model’s self-attention over a long reasoning trajectory to identify important steps and prune redundant ones.

![Overview of TRAAC](/Asset/image.png)
Given a problem, the model first generates $N$ rollouts, and the pass rate of these rollouts is used to estimate the problem's difficulty (easy, medium, or hard). Next, the generated reasoning is fed back into the model, which is asked to compute the attention score of each reasoning token from `</think>`. During this attention-based compression step, we remove steps with lower scores. The degree of removal is determined by the estimated difficulty: easier problems undergo more aggressive compression. Finally, we compute the correctness and length rewards using the compressed reasoning trajectory, and these rewards are used to update the policy.

## Installation

Please make sure that you have torch compiled with `CUDA` enabled. Repository developed with `python 3.12.3`, please ensure `python` envokes `python 3.12.3`. The codebase has been build on top of [verl](https://github.com/volcengine/verl).

Create virtual environment and Install verl and other packages from `requirements.txt`:
```bash
python -m venv traac_venv
source traac_venv/bin/activate
pip install -e .[vllm]
pip install -r requirements.txt
```
## Download Models
Download our trained Adaptive reasoning models directly from huggingface:

| Model | Download Link |
|-------|---------------|
| **(TRAAC)DeepSeek-R1-Distill-Qwen-7B** | [![Hugging Face](https://img.shields.io/badge/🤗-DeepSeek--R1--Distill--Qwen--7B--TRAAC-yellow.svg)](https://huggingface.co/joykirat/DeepSeek-R1-Distill-Qwen-7B-TRAAC) |
| **(TRAAC) Qwen3-4B** | [![Hugging Face](https://img.shields.io/badge/🤗-Qwen3--4B--TRAAC-yellow.svg)](https://huggingface.co/joykirat/Qwen3-4B-TRAAC) |


## Training Data & Evaluation

All training and evaluation scripts are located in the [`scripts/data`](scripts/data) folder.  
Make sure you `cd` into this folder before running the commands:

```bash
cd scripts/data
```
| Purpose                              | Script                 | 
| ------------------------------------ | ---------------------- | 
| **Training Data Generation**         | `dapo-17k.py`          | 
| **Evaluation – AIME, AMC, GPQA**     | `full_test_dataset.py` | 
| **Evaluation – Overthinking Benchmark** | `overthink_test_dataset.py`    |
| **Evaluation – Underthinking Benchmark** | `underthink_test_dataset.py`    |  
| **Evaluation – BBEH Benchmark** | `bbeh_test_dataset.py`    |  


These will create appropriate parquet files in the [`scripts/data`](scripts/data) folder.

## Run Evaluations
To run your custom evaluation on any of the above datasets use the bas script provided in [`scripts/eval`](scripts/eval/) folder.
Change the `data.val_files` field inside verl configuraiton to the required dataset:
```bash
./eval_deepseek-qwen-with-summary-linear-reward-attention.sh
./eval_qwen3-4b-with-summary-linear-reward-attention.sh
```

## Train Models

Training was conducted on **3 GPUs**:  
- **1 GPU** was dedicated to hosting the policy model for calculating attention scores (attention-based compression).  
- **2 GPUs** were used to train the main model.  


### Step 1: Host Attention-Based Compression (Qwen3-4B)
The host model script is located at [`scripts/train`](scripts/train):  
```bash
CUDA_VISIBLE_DEVICES=0 python model_host_qwen.py
```
This will host the Qwen3-4B model at [http://localhost:8008](http://localhost:8008).
### Step 2: Training TRAAC (Qwen3-4B)
```bash
source run_qwen3-4b-with-summary-linear-reward-attention.sh
```

The file [`vllm_rollout_spmd.py`](/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py) contains the implementation for adaptive, attentive summarization, which is used during training.

## Citation
If you find this work useful, please consider citing us:
```bibtex

```
