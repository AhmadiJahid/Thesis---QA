# Model Selection for Multi-hop Question Decomposition Pipeline

## Summary

Based on researcher-mode analysis (Dec 2025), here are the selected models for each pipeline component:

---

## Router Component (Hop Count Classification)

**Requirement:** Small, fast model (0.5B-1.5B) for classifying questions into 1-hop, 2-hop, or 3-hop.

### Primary Choice
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Parameters:** 494M
- **License:** Apache 2.0
- **Why it fits:** Fast baseline, excellent for classification tasks, strong instruction-following
- **Kaggle feasibility:** ✅ Likely fits Kaggle GPU easily
- **Link:** [https://hf.co/Qwen/Qwen2.5-0.5B-Instruct](https://hf.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Updated:** Sep 2024

### Alternatives
1. **`Qwen/Qwen2.5-1.5B-Instruct`** (1.5B, Apache 2.0)
   - Better accuracy if 0.5B insufficient
   - Still fits easily on Kaggle GPU
   - [Link](https://hf.co/Qwen/Qwen2.5-1.5B-Instruct)

2. **`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`** (1.8B, MIT)
   - Reasoning-focused distillation model
   - Good for hop detection with reasoning capabilities
   - [Link](https://hf.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

---

## Decomposer Component (Question Decomposition)

**Requirement:** Medium model (7B-8B) for generating JSON-structured sub-questions from multi-hop questions.

### Primary Choice
- **Model:** `Qwen/Qwen2.5-7B-Instruct`
- **Parameters:** 7.6B
- **License:** Apache 2.0
- **Why it fits:** Strong instruction-following, good for structured JSON generation, proven track record
- **Kaggle feasibility:** ✅ Likely fits Kaggle GPU with 4-bit quantization
- **Link:** [https://hf.co/Qwen/Qwen2.5-7B-Instruct](https://hf.co/Qwen/Qwen2.5-7B-Instruct)
- **Updated:** Jan 2025

### Alternative
- **`EssentialAI/rnj-1-instruct`** (8.3B, Apache 2.0)
  - Very recent (Dec 2025), optimized for instruction tasks
  - May have better structured output capabilities
  - [Link](https://hf.co/EssentialAI/rnj-1-instruct)

---

## Jury Component (Decomposition Validation)

**Requirement:** Same model as Decomposer (7B-8B) to validate decomposition quality.

**Model:** Same as Decomposer (see above)
- Validates: (1) sub-questions are in correct order, (2) they compose to original question, (3) sub-questions make sense
- Output: pass/fail

---

## Implementation Notes

- All models support instruction-following and structured generation
- Router can use smaller model for speed
- Decomposer and Jury share the same model to reduce infrastructure complexity
- Quantization (4-bit/8-bit) may be needed for 7B-8B models on Kaggle GPU
- JSON schema validation needed for Decomposer output

---

## References

- Decision recorded in `docs/DECISIONS.md` (2025-12-15)
- Planning discussion in `docs/PLANNER.md`
