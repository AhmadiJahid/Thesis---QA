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

### Topic: Code Analyzer Rule for Generated Notebooks
**Goal / question:**  
Create a Cursor rule file (`.mdc`) that defines how to analyze/review generated notebook code for quality, logic, and correctness.

**Context:**  
- We generate code/notebooks for Router, Decomposer, Jury components
- Code is generated in Cursor but executed on Kaggle
- Need a rule file (like `python-notebook-kaggle.mdc`) that guides code review
- When notebooks are generated, the analyzer rule should be used to review them

**What we know:**  
- Should be a `.mdc` file in `.cursor/rules/` directory
- Should provide guidelines for reviewing generated `.ipynb` files
- Focus: Code quality, logic, and correctness
- Should output structured feedback/reviews

**Decision:** ✅ **DECIDED**
- Create `.cursor/rules/code-analyzer.mdc` file
- Analyzer checks: code quality, logic errors, correctness
- Provides structured review feedback
- Used automatically when reviewing generated notebooks

**Proposed next step:**  
Create the `code-analyzer.mdc` rule file

---

### Topic: Router Component Performance Analysis (Run 20251215_040826)
**Goal / question:**  
Evaluate Router component performance after fixing data leakage and prompt overengineering. Determine if accuracy is acceptable to proceed to Decomposer (TQ-2).

**Context:**  
- Run ID: 20251215_040826
- Model: Qwen/Qwen2.5-1.5B-Instruct
- Fixed issues: Removed test data from prompt examples, simplified prompt structure
- Dataset: 90 questions (30 per hop count) from MetaQA

**Results:**
- Overall Accuracy: 77.78% (70/90)
- 1-hop Accuracy: 100% (30/30) ✅
- 2-hop Accuracy: 63.33% (19/30) ⚠️
- 3-hop Accuracy: 70% (21/30) ✅

**Error Patterns (from detailed_results.json):**
- 2-hop errors (11/30 wrong):
  - Predicted as 1-hop: 4 cases (e.g., "who appeared in the same movie with Angie Everhart", "what genres do the films starred by Al St. John fall under")
  - Predicted as 3-hop: 7 cases (e.g., "who are movie co-directors of Delbert Mann", "the director of The Brown Bunny also directed which movies")
- 3-hop errors (9/30 wrong):
  - Mostly predicted as 2-hop (pattern: model underestimates complexity)
  - Some edge cases with "same" or "also" patterns

**What we know:**  
- 1-hop classification is perfect (100%)
- 3-hop improved significantly from 13.33% → 70% after fixing data leakage
- 2-hop is the weakest (63.33%), with confusion between 1-hop and 3-hop
- Model tends to overestimate 2-hop as 3-hop when "also" or "same" appears
- Model underestimates some 2-hop as 1-hop when question structure is simple

**What we don't know:**  
- Is 77.78% overall accuracy acceptable for downstream Decomposer?
- Will 2-hop misclassification significantly impact Decomposer performance?
- Should we iterate more on Router or accept current performance?

**Options:**
- Option A: Accept current performance (77.78%) and proceed to Decomposer (TQ-2)
  - Pros: 1-hop perfect, 3-hop good, overall reasonable
  - Cons: 2-hop weak, may propagate errors to Decomposer
- Option B: Iterate on Router prompt to improve 2-hop accuracy
  - Pros: Better accuracy before moving forward
  - Cons: More time, diminishing returns
- Option C: Proceed to Decomposer but monitor Router errors in pipeline
  - Pros: Move forward while tracking impact
  - Cons: May need to revisit Router later

**Risks / failure modes:**  
- If Router misclassifies, Decomposer receives wrong hop count → wrong decomposition
- 2-hop questions misclassified as 3-hop may cause Decomposer to over-decompose
- 2-hop questions misclassified as 1-hop may cause Decomposer to under-decompose

**Decision needed by:**  
User to decide if 77.78% is acceptable or if we should iterate more

**Proposed next step:**  
- If acceptable: Proceed to Decomposer component (TQ-2)
- If not: Analyze 2-hop error patterns more deeply, refine prompt with better examples

---

### Topic: Results Visualization and Analysis Agent
**Goal / question:**  
Create an agent/component that analyzes run results, generates figures/plots, and creates presentable reports for professor presentation.

**Context:**  
- Multiple runs exist in `runs/router/` directory (e.g., 20251215_031710, 20251215_035343, 20251215_040826)
- Each run contains: `metrics.json`, `detailed_results.json`, `config.json`, `notes.md`
- Need to compare runs, show trends, visualize performance
- Some runs may be archived (folders named "archive" should be ignored)
- Output should be presentable for academic presentation

**What we know:**  
- Run structure: `runs/<component>/<run_id>/` with standardized artifacts
- Metrics include: overall accuracy, per-hop accuracy, model info, seed
- Detailed results include: question, expected_hop, predicted_hop, correct
- Need to skip "archive" folders

**Decision:** ✅ **DECIDED**
- **Output format:** Local Python script (Option B) - easier to iterate, can generate HTML/PDF
- **Execution:** Can also work as Kaggle notebook if needed (flexible)
- **Scope:** All runs except folders named exactly "archived"
- **Archive detection:** Exact match - folders named "archived" (case-sensitive)

**Recommended Visualizations (for professor presentation):**
1. **Overall Accuracy Comparison** (bar chart)
   - Compare all runs side-by-side
   - Shows improvement over iterations
   - Essential for showing progress

2. **Per-Hop Accuracy Trends** (line/bar chart)
   - 1-hop, 2-hop, 3-hop accuracy across runs
   - Shows which hop counts are challenging
   - Critical for understanding model behavior

3. **Confusion Matrices** (heatmap, one per run)
   - Shows misclassification patterns
   - Helps identify systematic errors
   - Academic standard for classification tasks

4. **Error Pattern Summary** (table or bar chart)
   - Most common error types (e.g., "2-hop → 3-hop", "2-hop → 1-hop")
   - Shows what the model struggles with
   - Useful for identifying improvement areas

5. **Model/Config Comparison** (if different models used)
   - Compare performance across model variants
   - Shows impact of model choice

**Visualizations to skip (too detailed for presentation):**
- Individual question-level errors (too granular)
- Timeline view (if runs are close in time, not meaningful)
- Raw detailed results (can be in appendix)

**Output Structure:**
- HTML report with embedded plots (easy to view, can convert to PDF)
- Save plots as PNG files for potential use in slides
- Include summary statistics table
- Keep it concise (5-7 visualizations max for presentation)

**Risks / failure modes:**  
- Archive detection might miss variations (e.g., "archived", "old")
- Too many visualizations might overwhelm
- Missing run data might cause errors
- Different run structures might break parsing

**Decision:** ✅ **DECIDED**
- **Output format:** Local Python script (Option B) - easier to iterate, can generate HTML/PDF
- **Execution:** Can also work as Kaggle notebook if needed (flexible)
- **Scope:** All runs except folders named exactly "archived"
- **Archive detection:** Exact match - folders named "archived" (case-sensitive)

**Recommended Visualizations (for professor presentation):**
1. **Overall Accuracy Comparison** (bar chart)
   - Compare all runs side-by-side
   - Shows improvement over iterations
   - Essential for showing progress

2. **Per-Hop Accuracy Trends** (line/bar chart)
   - 1-hop, 2-hop, 3-hop accuracy across runs
   - Shows which hop counts are challenging
   - Critical for understanding model behavior

3. **Confusion Matrices** (heatmap, one per run)
   - Shows misclassification patterns
   - Helps identify systematic errors
   - Academic standard for classification tasks

4. **Error Pattern Summary** (table or bar chart)
   - Most common error types (e.g., "2-hop → 3-hop", "2-hop → 1-hop")
   - Shows what the model struggles with
   - Useful for identifying improvement areas

5. **Model/Config Comparison** (if different models used)
   - Compare performance across model variants
   - Shows impact of model choice

**Visualizations to skip (too detailed for presentation):**
- Individual question-level errors (too granular)
- Timeline view (if runs are close in time, not meaningful)
- Raw detailed results (can be in appendix)

**Output Structure:**
- HTML report with embedded plots (easy to view, can convert to PDF)
- Save plots as PNG files for potential use in slides
- Include summary statistics table
- Keep it concise (5-7 visualizations max for presentation)

**Decision:** ✅ **DECIDED** - Moved to DECISIONS.md on 2025-12-15
**Proposed next step:**  
Script created. Create Jira task TQ-5 for tracking.

---

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
