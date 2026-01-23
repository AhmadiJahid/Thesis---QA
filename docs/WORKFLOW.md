# Thesis Workflow

This repo is operated with a Jira-first workflow and lightweight Git checkpoints (no PRs).
Primary goals: reproducibility, traceability, correctness.

---

## Core loop

### 1) Sync (planning)
Use: `@cmd-sync-jira`

Outcome:
- Agent fetches Jira tasks (To Do / In Progress / Blocked).
- Agent proposes the top 1–3 next tasks (ranked).
- Each proposed task includes a crisp Definition of Done (DoD).
- `docs/PROJECT.md` is updated in the “Next Up (from Jira)” section.

Rules:
- No implementation code during sync.
- No task is considered “done” unless Jira is updated and docs are updated.

---

### 2) Execute (implementation)
Work on the top-ranked Jira task on a dedicated branch.

Branch naming:
- `thesis/<JIRA-KEY>-<short-slug>`

Execution discipline:
- Prefer small, reviewable changes.
- Don’t invent results. If something isn’t measured, it’s unmeasured.
- Keep repo structure consistent (src/, notebooks/, configs/, docs/).
- Never commit run artifacts (`runs/`, `.ipynb_checkpoints/`, etc.).

---

### 3) Checkpoint (commit + push)
Use: `@cmd-git-checkpoint`

When to checkpoint:
- after adding rules/docs
- after stabilizing repo structure
- when a task’s acceptance criteria is met
- whenever you reach a coherent “safe point”

Commit message format:
- `<JIRA-KEY>: <short message>`

Push:
- push the branch to GitHub regularly to avoid local-only progress.

---

### 4) Close (Jira + docs)
Use: `@cmd-close-task`

Definition of “done” requires all of the following:
- Jira issue has a completion comment summarizing what was done
- Jira issue is transitioned to Done (if permitted)
- `docs/CHANGELOG.md` has a new append-only entry
- `docs/PROJECT.md` is updated ONLY if the project state changed

Optional:
- If a real decision was made (choice that prevents re-arguing later), record it in `docs/DECISIONS.md`.

---

## Kaggle execution policy (high-level)

- Code is authored in this repo (Cursor) and executed on Kaggle (no local GPU assumed).
- Runnable artifacts should usually be notebooks in `notebooks/`.
- Kaggle outputs go to `/kaggle/working/runs/<run_id>/` and must not be committed to git.

(Implementation details live in `docs/RUNBOOK_KAGGLE.md` if/when created.)

---

## What the agent should update automatically

When a Jira task is completed:
- Append to `docs/CHANGELOG.md`
- Update `docs/PROJECT.md` if and only if the task changes:
  - pipeline stages
  - repo structure
  - run method
  - operating conventions

The agent must not rewrite past changelog entries.

---

## Quick commands summary

- `@cmd-sync-jira` → pull tasks, rank next, define DoD, update PROJECT.md
- `@cmd-git-checkpoint` → prepare clean commit + push steps (no PRs)
- `@cmd-close-task` → close Jira + update docs (changelog + project brief)
