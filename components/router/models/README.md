# Router Models

Each model folder contains:
- `prompt.md` with the hop-count prompt template (few-shot)
- `config.json` with model and decoding settings

Shared in this directory:
- `prompt_zero_shot.md` — zero-shot prompt (no few-shot examples). Use `--prompt_file prompt_zero_shot.md` to run without few-shot learning.

Folder naming uses lowercase with underscores (e.g., `qwen3_1_7b`).
