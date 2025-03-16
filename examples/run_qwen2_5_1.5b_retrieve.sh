set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
MODEL_PATH=/home/mcb/users/zsong15/qwen_model

# Use single quotes to protect the entire string
SYSTEM_PROMPT='You have a question that requires multi-step reasoning and information retrieval. Follow these steps - 1) FIRST, think about the reasoning process as an internal monologue. This MUST BE enclosed within <think> </think> tags. 2) THEN, identify and extract relevant information needed to answer the question. This MUST BE enclosed within <retrieval> </retrieval> tags. 3) FINALLY, provide your final answer. This MUST BE enclosed within <answer> </answer> tags.'

python3 -m verl.trainer.main \
config=examples/grpo_example.yaml \
data.train_files=hiyouga/geometry3k@train \
data.val_files=hiyouga/geometry3k@test \
data.system_prompt=''"$SYSTEM_PROMPT"'' \
worker.actor.model.model_path=${MODEL_PATH} \
worker.rollout.enable_chunked_prefill=false \
trainer.experiment_name=qwen2_5_retrieval_geo \
worker.reward.compute_score=retrieve \
trainer.n_gpus_per_node=4