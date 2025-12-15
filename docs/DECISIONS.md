# Decisions (Thesis)

Keep this short. Only record decisions that prevent re-litigating choices.

Template:
## YYYY-MM-DD — Decision title
**Context:**  
**Decision:**  
**Why:**  
**Tradeoffs:**  
**Consequences / Follow-up:**

---

## 2025-12-15 — Multi-hop Question Decomposition Pipeline Architecture
**Context:**  
Building a 3-stage pipeline to decompose multi-hop questions (1-hop, 2-hop, 3-hop) from MetaQA dataset. Need to decide on architecture for Router, Decomposer, and Jury components.

**Decision:**  
- **Router:** Outputs hop count (1/2/3) using small model (0.5B-1.5B, e.g., Qwen2.5-0.5B-Instruct)
- **Decomposer:** Outputs JSON list of sub-questions (no reasoning) using medium model (7B-8B, e.g., Qwen2.5-7B-Instruct)
- **Jury:** Validates (1) sub-questions are in correct order, (2) they compose to original question, (3) sub-questions make sense. Output: pass/fail. Uses same model as decomposer.
- **Evaluation:** Manual review for now

**Why:**  
- Hop count from router provides specific guidance for decomposer
- JSON format enables structured processing and validation
- Jury using same model reduces infrastructure complexity
- Manual evaluation allows iterative refinement before automated metrics

**Tradeoffs:**  
- No reasoning in decomposer output simplifies pipeline but may reduce interpretability
- Jury pass/fail is binary but sufficient for initial validation
- Manual evaluation is time-consuming but necessary without ground truth

**Consequences / Follow-up:**  
- Need to implement JSON schema validation for decomposer output
- Jury validation logic needs clear criteria for pass/fail
- Create Jira tasks for Router, Decomposer, and Jury implementation
- Plan for future automated evaluation metrics

**Model Selection (from researcher-mode analysis):**
- **Router:** `Qwen/Qwen2.5-0.5B-Instruct` (494M, Apache 2.0) - Primary choice
- **Decomposer & Jury:** `Qwen/Qwen2.5-7B-Instruct` (7.6B, Apache 2.0) - Primary choice
- See `docs/MODEL_SELECTION.md` for full details and alternatives

---

## 2025-12-15 — Results Visualization and Analysis Agent
**Context:**  
Multiple runs exist in `runs/router/` directory with standardized artifacts (metrics.json, detailed_results.json, etc.). Need to analyze results, generate visualizations, and create presentable reports for professor presentation.

**Decision:**  
- Create local Python script (`scripts/analyze_runs.py`) that:
  - Scans `runs/<component>/` directories
  - Skips folders named exactly "archived" (case-sensitive)
  - Generates 5 key visualizations: overall accuracy comparison, per-hop accuracy trends, confusion matrices, error pattern summary, model comparison
  - Outputs HTML report with embedded plots + PNG files for slides
  - Can also work as Kaggle notebook if needed (flexible)

**Why:**  
- Local script allows easy iteration and doesn't require Kaggle setup
- HTML format is easy to view, share, and convert to PDF
- PNG files can be directly used in presentation slides
- Academic-style visualizations suitable for professor presentation

**Tradeoffs:**  
- Requires local Python environment with matplotlib/seaborn/pandas
- HTML may need conversion to PDF for formal submission
- Script needs to be maintained as run structure evolves

**Consequences / Follow-up:**  
- Script created: `scripts/analyze_runs.py`
- Documentation added: `scripts/README.md`
- Jira task TQ-5 to be created for tracking
- Future: May extend to support multiple components (decomposer, jury) when available
