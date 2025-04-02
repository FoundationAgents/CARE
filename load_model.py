from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    local_dir="./Qwen2.5-1.5B-Instruct",
    local_dir_use_symlinks=False
)
