# Thesis Project

**Goal:** Multi-hop question decomposition pipeline for KG-augmented QA (MetaQA).

**Execution model:** Code is authored in this repo (Cursor), executed on Kaggle (no local GPU assumed).

**Dataset:** MetaQA (stored in `DATA/`).  
**Data policy (current phase):** no splitting, no cleaning, no preprocessing.

---

## Current Pipeline

### Stages
- Stage A (Router): TBD
- Stage B (Decomposer): TBD
- Retrieval / Execution: TBD
- Evaluation: TBD

### Current status (1–5 lines)
- [PLACEHOLDER: What works today? What’s the latest runnable thing?]

---

## Repo Structure

- `src/` — core library code
- `scripts/` — entrypoints / CLIs (optional; Kaggle-first)
- `notebooks/` — Kaggle-ready notebooks (primary execution artifacts)
- `configs/` — experiment configs (when we begin running experiments)
- `docs/` — thesis notes, methodology, literature notes (later)
- `DATA/` — MetaQA files (as provided)
- `runs/` — output artifacts (must be gitignored)

---

## Operating Conventions

### Reproducibility (when experiments start)
Each run should produce:
- `metrics.json`
- `notes.md`
- config snapshot (yaml/json) if configs are used

### Tracking
We use Jira as the source of truth for planned work.
Completed Jira tasks must be reflected in:
- `docs/CHANGELOG.md` (append-only)
- `docs/PROJECT.md` (only if project state changed)

---

## Next Up (from Jira)
- [PLACEHOLDER: Agent fills from Jira during sync]
