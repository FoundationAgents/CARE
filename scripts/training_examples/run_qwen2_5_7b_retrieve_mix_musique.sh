#!/usr/bin/env bash
set -x

# -------------------------------
# Attention backend configuration
# -------------------------------
# Set the attention backend for vLLM.
# Options: FLASH_ATTN | XFORMERS | TORCH_SDPA
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0

# -------------------------------
# Model path
# -------------------------------
# Current example uses a locally fine-tuned Qwen2.5-7B model.
# Replace the path with either:
#   1. Your local model checkpoint directory
#   2. Or a Hugging Face Hub repo id, e.g. Qwen/Qwen2.5-7B-Instruct
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct

# -------------------------------
# System prompt
# -------------------------------
# Defines the reasoning style of the model.
# You can edit SYSTEM_PROMPT to change how reasoning/answers are generated.
SYSTEM_PROMPT='You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. WITHIN the thinking process, make reference to the relevant texts in the prompt that provide critical information to move the reasoning process forward. The referenced texts MUST BE enclosed within <retrieval> </retrieval> tags, and MUST BE placed within the reasoning process only. The final answer MUST BE put at the end of the response after "Answer:".'

# -------------------------------
# Training command
# -------------------------------
# - config: Hydra/YAML config file (examples/grpo_example.yaml provided)
# - data.*: dataset inputs (DROP + MuSiQue in this example)
# - worker.actor.model.model_path: points to the model defined above
# - trainer.experiment_name: name for this training run (change for new runs)
# - trainer.max_steps: total number of training steps
# - worker.reward.compute_score: which reward function to use ("retrieve" here)
# - trainer.n_gpus_per_node: number of GPUs per node to use
python3 -m verl.trainer.main \
    config=examples/grpo_example.yaml \
    data.train_files=sheryc/DROP_processed \
    data.extra_files=sheryc/MuSiQue_processed \
    data.val_files=sheryc/DROP_processed@validation \
    data.system_prompt="$SYSTEM_PROMPT" \
    data.prompt_key=prompt \
    data.answer_key=answer \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_7b_ret_cur_drop_musique \
    trainer.max_steps=10000 \
    worker.reward.compute_score=retrieve \
    trainer.n_gpus_per_node=8
