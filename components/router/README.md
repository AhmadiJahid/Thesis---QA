# Router Component

This component classifies MetaQA questions into hop counts (1/2/3). Model-specific
assets live under `components/router/models/<model_name>/`.

Each model folder should include:
- `prompt.md` for the hop-count prompt template (few-shot)
- `config.json` for model and decoding settings

Shared in `components/router/models/`:
- `prompt_zero_shot.md` — zero-shot prompt (no few-shot examples). Use `--prompt_file prompt_zero_shot.md` when running a model.

**CLI options (per-model router.py):**
- `--prompt_file FILE` — Override prompt (e.g. `prompt_zero_shot.md` for zero-shot).
- `--num_runs N` — Run inference N times with seeds seed, seed+1, … and save per-run metrics plus `metrics_aggregated.json` (mean ± std). Default: 1.
- `--output_root DIR` — Override output root (defaults below).

**Where results are saved (under `runs/`, gitignored):**
- **Zero-shot:** `runs/average_zero_shot/<run_id>/` when using `--prompt_file prompt_zero_shot.md`.
- **Few-shot:** `runs/average_few_shot/<run_id>/` when using the default prompt.
- Pass `--output_root runs/router` to keep the old behaviour.

Generate graphs and HTML report for presentation:
```bash
python scripts/analyze_runs.py --component average_zero_shot   # -> reports/average_zero_shot/
python scripts/analyze_runs.py --component average_few_shot   # -> reports/average_few_shot/
```
