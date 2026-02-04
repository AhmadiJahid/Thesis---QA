# Decomposition pipeline: errors and how to infer them

The question decomposition pipeline turns a **decomposition** (sub-questions from the model) into **KG operations** and runs them on the knowledge graph. Two stages can fail: **compilation** and **execution**. The analysis files (`compile_fail.json`, `exec_fail.json`) and the report charts are built from these.

---

## Pipeline overview

1. **Input**: Each item has `question`, `hop_count`, and `decomposition` (e.g. `"1. What films did X star in?\n2. Who directed [#1]?"`).
2. **Compilation**: The script parses each sub-question (step) and maps it to a **template**. From the template it produces a list of **ops** (e.g. `MOVIES_WITH_VALUE(relation='starred_actors', entity='X')`, `PROJECT_VALUES_FROM_MOVIES(relation='directed_by', ref_step=1)`).
3. **Execution**: Those ops are run on the KG (and reverse index). Step 1 might return movie IDs; step 2 uses those IDs to look up directors, etc. The final result is a set of entity IDs, which are turned into answer strings.

If **compilation** fails → the item goes to **compile_fail** (template/relation error).  
If compilation succeeds but **execution** fails → the item goes to **exec_fail** (KG lookup or plan error).  
If both succeed → the item goes to **success.json** and can be compared to gold answers.

---

## Compilation (compile_fail)

**What “compilation” means here**: turning the **text** of each decomposition step into a **structured op** the executor understands. The compiler only supports a fixed set of **sentence templates** (e.g. “What movies did PERSON star in?”, “Who directed MOVIE?”, “What languages are [#1] in?”). It also infers the **KG relation** from keywords (actor → `starred_actors`, director → `directed_by`, etc.).

### Compile-fail reasons

| Reason | How it happens | How to infer / fix |
|--------|----------------|---------------------|
| **missing_decomposition** | `decomposition` is missing, empty, or not a string. | Check model output: ensure the decomposer actually produced a decomposition for this question. |
| **unsupported_template** | The step text did not match any of the compiler’s regex templates (e.g. “In which movies does X appear?” vs “What films did X star in?”). The compiler raises `NotImplementedError`. | The **wording** of the sub-question doesn’t match the supported patterns. Either add a new template in `evaluate_decompositions.py` (e.g. “In which movies does …”) or change the prompt so the model produces steps that match existing templates (e.g. “What films did X appear in?”). |
| **cannot_infer_relation** | The step text didn’t contain any keyword the compiler uses to infer a relation (actor, director, writer, genre, language, year, etc.). The compiler raises `ValueError` from `infer_relation()`. | The step asks about something the compiler doesn’t map to a KG relation (e.g. “Who produced …?” if there’s no producer rule). Add a relation rule in `REL_RULES` / `infer_relation`, or change the decomposition to use wording that matches existing rules. |
| **compile_error_other** | Any other exception during compilation (e.g. bad placeholder syntax, bug in a template). | Inspect the exception message in `compile_fail.json` (`error_reason`). Fix the compiler or the decomposition format. |

**Summary**: Compile fails = “we couldn’t turn this sub-question text into a known op.” So it’s about **template coverage** and **relation inference**, not about the KG content yet.

---

## Execution (exec_fail)

**What “execution” means**: Running the **compiled ops** on the KG. This step looks up entities by name, follows relations, and resolves placeholders like `[#1]` using results from previous steps.

### Exec-fail reasons

| Reason | How it happens | How to infer / fix |
|--------|----------------|---------------------|
| **entity_not_in_kb** | An op needs an **entity by name** (e.g. a person or movie). The name is not in `kg.entity_to_id`, so the executor raises `KeyError`. The full message is like `entity_not_in_kb: 'the'` or `entity_not_in_kb: 'Some Actor'`. | The **name** extracted from the decomposition (e.g. “the”, or a misspelling, or a name not in the KB) doesn’t exist in the KG. Fix by improving the decomposer so it outputs entity names that exist in the KB, or by entity linking / normalization. |
| **bad_reference_or_plan** | A step uses a **placeholder** `[#k]` but step `k` doesn’t exist or didn’t produce a result; or an op is missing `entity` / `ref_step`. The executor raises `KGExecutionError`. | The **plan** is invalid: e.g. “Step 1 refers to missing step #1” (circular or wrong numbering), or “Step 2 refers to missing step #1” (step 1 failed or wasn’t the right type). Fix by improving the model’s decomposition (correct step order and references) or by extending the executor for more plan shapes. |
| **exec_error_other** | Any other exception during execution. | Check the full `error_reason` in `exec_fail.json`; usually a bug or an unexpected op/state. |

**Summary**: Exec fails = “the op was valid, but we couldn’t run it on the KG.” So it’s about **entity names** and **step references**, and sometimes bugs in the executor.

---

## How the report infers and shows them

- **decomposition_pipeline.png**: Counts from the three outcomes — **Success** (compiled and executed), **Compile fail**, **Exec fail** — so you see how many questions end in each bucket.
- **compile_fail_reasons.png**: Count per **compile** reason (`missing_decomposition`, `unsupported_template`, `cannot_infer_relation`, `compile_error_other`). So you see which kind of “template/relation” problem is most frequent.
- **exec_fail_reasons.png**: Count per **exec** reason (`entity_not_in_kb`, `bad_reference_or_plan`, `exec_error_other`). The script normalizes reasons so e.g. `entity_not_in_kb: 'the'` is grouped under `entity_not_in_kb`.

So: **compilation** = turning decomposition text into ops (templates + relation inference); **execution** = running those ops on the KG (entity lookup + step references). The errors tell you whether to fix templates/relations (compile) or entity names/plan (exec).
