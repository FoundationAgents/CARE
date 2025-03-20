set -x
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0
MODEL_PATH=/mnt/home/LLaMA-Factory/saves/Qwen2.5-7B-Instruct-Ret/full/sft

# Use single quotes to protect the entire string
SYSTEM_PROMPT='You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. WITHIN the thinking process, make reference to the relevant texts in the prompt that provide critical information to move the reasoning process forward. The referenced texts MUST BE enclosed within <retrieval> </retrieval> tags, and MUST BE placed within the reasoning process only. The final answer MUST BE put at the end of the response after "Answer:".'

python3 -m verl.trainer.main \
config=examples/grpo_example.yaml \
data.train_files=sheryc/DROP_processed@train \
data.val_files=sheryc/DROP_processed@validation \
data.system_prompt=''"$SYSTEM_PROMPT"'' \
data.prompt_key=prompt \
data.answer_key=answer \
worker.actor.model.model_path=${MODEL_PATH} \
worker.rollout.enable_chunked_prefill=false \
trainer.experiment_name=qwen2_5_7b_retrieval \
worker.reward.compute_score=retrieve \
trainer.n_gpus_per_node=8