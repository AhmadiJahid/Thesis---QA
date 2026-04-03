# Changelog (Thesis)

Format:
- Date — JIRA-KEY — Title
  - Summary of what changed (1–5 bullets)
  - GitHub: PR/commit link (if applicable)

---

## Unreleased

- MusiQue train JSONL: stratified 4-way split (`MusiQue/scripts/split_musique_train_stratified.py`), field cleaning (`MusiQue/scripts/clean_musique_train_chunks.py`), chunk stats plots (`MusiQue/scripts/plot_musique_chunk_stats.py`); shared id helpers in `MusiQue/scripts/musique_ids.py`; added `scikit-learn` and `matplotlib` to `requirements.txt`.
- MusiQue question-only pipeline: `MusiQue/scripts/extract_musique_clean_questions.py` (clean JSONL → per-stratum `chunks_only_question/*_questions_*_hop*.jsonl` + `runs/musique_question_extract`); `MusiQue/scripts/ner_mask_musique_question_chunks.py` (defaults: `dslim/bert-large-NER` + `tner/deberta-v3-large-conll2003`, typed + uniform `[MASK]`, optional regex `[NUM]`/`[DATE]` gaps, per-model under `chunks_only_question_masked/<slug>/` + `runs/musique_question_ner_mask`); `stratum_to_questions_slug` added in `MusiQue/scripts/musique_ids.py`.
- `ner_mask_musique_question_chunks.py` default output is one combined JSONL per input under `<slug>/` with `id`, `index`, `question`, `question_masked_typed`, `question_masked_uniform`; use `--split-typed-uniform-dirs` for separate `typed/` and `uniform/` trees; batched NER via `datasets.Dataset` + `--batch-size`, single entity pass per row for typed and uniform masks; `datasets` added to `requirements.txt`; tokenizer load falls back to `use_fast=False` when the Hub fast tokenizer file is missing (e.g. `tner/deberta-v3-large-conll2003`).

---

## Entries
<!-- New entries appended below -->

- 2025-12-15 — TQ-5 — Implement Results Visualization and Analysis Agent
  - Created `scripts/analyze_runs.py` script that analyzes router component run results
  - Implemented visualization generation: overall accuracy, per-hop accuracy, confusion matrices, error patterns
  - Script correctly skips folders with "_archived" suffix in their names
  - All plots include run ID and model name for easy identification
  - Confusion matrices use red colormap (Reds) for better visibility
  - Generated HTML report with embedded plots and summary statistics table
  - Created virtual environment and installed dependencies
  - GitHub: n/a
