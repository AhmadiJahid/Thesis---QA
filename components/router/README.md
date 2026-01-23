# Router Component

This component classifies MetaQA questions into hop counts (1/2/3). Model-specific
assets live under `components/router/models/<model_name>/`.

Each model folder should include:
- `prompt.md` for the hop-count prompt template
- `config.json` for model and decoding settings

Runs are stored under `runs/router/<run_id>/` (output-only, gitignored).
