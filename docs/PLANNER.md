# Thesis Planner

Purpose: A dedicated space to think, argue, and decide *before* implementing.
This document is allowed to be messy, but decisions must be promoted to `docs/DECISIONS.md`
and concrete work must become Jira tasks.

---

## How to use this (aligned with our workflow)

### When starting a new initiative (new feature/experiment)
1) Write the idea in **Current Topics**.
2) Use **Decision Template** to settle key choices.
3) Convert outcomes into Jira tasks.
4) Run `@cmd-sync-jira` to rank work and update `docs/PROJECT.md`.

### When closing work
- If something became a real decision: copy it into `docs/DECISIONS.md`.
- If work is completed: update Jira and append to `docs/CHANGELOG.md`.

---

## Rules for the Planner

- Planner is for exploration, tradeoffs, and strategy.
- No fabricated results or claims.
- Keep entries short. Prefer bullets and clear headings.
- If something is decided, record it in `docs/DECISIONS.md` (not here).
- If something becomes actionable, create Jira tasks (not just discussion).

---

## Current Topics (active)

### Topic: Multi-hop Question Decomposition Pipeline
**Goal / question:**  
Build a 3-stage pipeline to decompose multi-hop questions (1-hop, 2-hop, 3-hop) from MetaQA dataset into sub-questions for KG-augmented QA.

**Context:**  
- Dataset: MetaQA questions stored in `Data/refined_1hop.txt`, `refined_2hop.txt`, `refined_3hop.txt`
- Each file contains sample questions (31 samples each currently)
- 1-hop: Direct questions (e.g., "What does Grégoire Colin appear in?")
- 2-hop: One intermediate step (e.g., "which person directed the movies starred by John Krasinski")
- 3-hop: Two intermediate steps (e.g., "the films that share directors with the film Catch Me If You Can were in which languages")

**Constraints:**  
**What we know:**  
**What we don’t know:**  
**Options:**
- Option A: Router outputs hop count (1/2/3), Decomposer outputs ordered list of sub-questions, Jury validates each sub-question makes sense
- Option B: Router outputs binary (single-hop vs multi-hop), Decomposer handles all multi-hop, Jury validates entire decomposition chain
- Option C: Router outputs hop count + confidence, Decomposer outputs sub-questions with reasoning, Jury provides detailed feedback

**Risks / failure modes:**  
**Decision needed by:**  
**Proposed next step:** (usually “create Jira tasks”)

---

## Backlog of Ideas (not active yet)
Short one-liners only:
- 
- 
- 

---

## Decision Template (copy/paste when needed)

### Decision: [Short title]
**Context:**  
**Decision:**  
**Why:**  
**Tradeoffs:**  
**Consequences / follow-up:**  

✅ After agreeing: move this to `docs/DECISIONS.md` and create Jira tasks.

---

## Experiment Planning Template (for Experiment-0, Experiment-1, ...)

### Experiment: [Name]
**Hypothesis:**  
**Dataset:** (MetaQA; no splitting/cleaning in current phase)  
**Task definition:** (what the model must output)  
**Output format:** (e.g., strict JSON schema)  
**Models to compare:**  
**Metrics (early):**
- JSON validity rate
- hop-count correctness (2-hop vs 3-hop)
- manual sanity check size (e.g., 20–50)

**Stop condition:** (when to stop iterating)  
**Artifacts:** (metrics.json, notes.md, config snapshot if used)

✅ After agreeing: create Jira tasks.

---

## Notes / Scratchpad

### 2025-12-15: Kaggle Notebook Execution Approach
**Decision:** All pipeline components (Router, Decomposer, Jury) will be implemented as Jupyter Notebooks for Kaggle execution.

**Rationale:**
- No local GPU available → Kaggle provides free GPU access
- Notebook format aligns with `python-notebook-kaggle.mdc` conventions
- Notebooks are primary execution artifacts per `docs/PROJECT.md`
- Structure already defined: Title → Environment → Config → Data → Model → Run → Eval → Artifacts

**Notebook Structure (per python-notebook-kaggle.mdc):**
1. Title + goal (markdown)
2. Environment + versions (python, torch, transformers)
3. Config cell (dataset paths, model id, hyperparams, seed)
4. Data loading
5. Model setup
6. Run / inference / training loop
7. Evaluation + metrics
8. Save artifacts (`/kaggle/working/runs/<run_id>/`)

**Files to create:**
- `notebooks/router_component.ipynb` (TQ-1)
- `notebooks/decomposer_component.ipynb` (TQ-2)
- `notebooks/jury_component.ipynb` (TQ-3)
- `notebooks/pipeline_integration.ipynb` (TQ-4)

**Kaggle filesystem:**
- Inputs: `/kaggle/input/...` (read-only)
- Outputs: `/kaggle/working/` (writable)
- Artifacts: `/kaggle/working/runs/<run_id>/`

**Reproducibility:**
- Seed in config
- Log package versions
- Save: `metrics.json`, `notes.md`, config snapshot
