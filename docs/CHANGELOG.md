# Changelog (Thesis)

Format:
- Date — JIRA-KEY — Title
  - Summary of what changed (1–5 bullets)
  - GitHub: PR/commit link (if applicable)

---

## Unreleased

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
