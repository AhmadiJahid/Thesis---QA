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
- Initial Router experiments (e.g., run `20251215_040826` using Qwen/Qwen2.5-1.5B-Instruct) achieved 77.78% overall hop-count accuracy on a 90-question MetaQA sample; 1-hop is strong, 2-hop remains weakest.
- Decomposer and Jury components are not yet implemented; notebooks are planned but not runnable.
- Results visualization script `scripts/analyze_runs.py` exists for Router runs but is not yet wired into a full pipeline.

---

## Repo Structure

- `components/` — component-specific assets (router/decomposer/jury)
- `components/<component>/models/` — per-model prompt + config for that component
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
- **Jira Project:** TQ (Thesis - QA)
- **Cloud ID:** f97c7789-c9f8-4559-97e4-18d9d8c3845f
- **Project URL:** https://jahid-ahmadi-student.atlassian.net/browse/TQ

Completed Jira tasks must be reflected in:
- `docs/CHANGELOG.md` (append-only)
- `docs/PROJECT.md` (only if project state changed)

---

## Next Up (from Jira)

**Ranked tasks (created 2025-12-15):**

1. **[TQ-1] Implement Router Component - Hop Count Classification**
   - Classify questions into 1-hop, 2-hop, or 3-hop
   - Model: Qwen2.5-0.5B-Instruct
   - Status: To Do

2. **[TQ-2] Implement Decomposer Component - Question Decomposition to Sub-questions**
   - Break multi-hop questions into ordered sub-questions (JSON format)
   - Model: Qwen2.5-7B-Instruct
   - Status: To Do

3. **[TQ-3] Implement Jury Component - Decomposition Validation**
   - Validate sub-questions (order, composition, sense)
   - Model: Qwen2.5-7B-Instruct (same as Decomposer)
   - Status: To Do

4. **[TQ-4] Integrate Pipeline Components - End-to-End Pipeline**
   - Combine Router → Decomposer → Jury into complete pipeline
   - Status: To Do

5. **[TQ-5] Implement Results Visualization and Analysis Agent**
   - Analyze run results, generate visualizations, create presentable reports
   - Script: `scripts/analyze_runs.py`
   - Status: To Do

See Jira project: https://jahid-ahmadi-student.atlassian.net/browse/TQ
