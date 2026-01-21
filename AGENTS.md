# Thesis Agent Instructions

You are helping build a multi-hop question decomposition pipeline for KG-augmented QA (the "Thesis" project).
Optimize for: reproducibility, traceability, and correctness.

## Ground rules (always)
- Do NOT invent experimental results, metrics, dataset stats, or “it improved by X%” claims. If it’s not measured, say it’s unmeasured.
- Prefer small, reviewable diffs. When changing behavior, update tests/docs/logging.
- Keep runs reproducible: fixed seeds, saved configs, deterministic-ish settings where feasible.
- Every experiment must produce: (1) a config snapshot, (2) metrics JSON, (3) a short human-readable run note.
- Prefer Python for research code unless the repo clearly uses another language.

## Repo conventions
- `src/` contains library code.
- `scripts/` contains CLI entrypoints.
- `configs/` contains YAML/JSON configs for runs.
- `runs/` is output-only (ignored by git). Store run artifacts there.
- `docs/` contains thesis notes, methodology, and literature review.
- Code is authored here and executed primarily as Kaggle notebooks; see `.cursor/rules/python-notebook-kaggle.mdc` for notebook conventions.

Also see `.cursor/rules/thesis-core.mdc` for always-on rules about seeds, metrics, and run artifacts.

## Issue hygiene
- Track work via Jira issues (Atlassian MCP).
- One branch per Jira issue.
- Branch naming: `thesis/<JIRA-KEY>-short-slug`
- Commit messages start with the Jira key: `<JIRA-KEY>: <message>`
- When a Jira task is completed, update:
  - `docs/CHANGELOG.md` (append-only)
  - `docs/PROJECT.md` if project state changed


## Writing style
- Be direct. No fluff.
- When discussing methods, define terms briefly on first use (e.g., “entity linking”, “multi-hop”).

### Output discipline
- Do NOT generate code unless the user explicitly asks for code or asks for a notebook. This applies to all agents (including planner/research modes) unless the user explicitly changes the request.
