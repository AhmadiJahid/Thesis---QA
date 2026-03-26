import argparse
import json
import random
import re
import sys
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add repo src for pool_embeddings
_repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_repo_root / "src"))

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_questions(file_path: Path):
    """Load questions from a text file (one per line)."""
    if not file_path.exists():
        print(f"Warning: {file_path} not found.")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions

def load_few_shot_decompositions(workspace_root: Path) -> dict:
    """Load few-shot examples from Pool/few_shot_decompositions.json."""
    path = workspace_root / "Pool" / "few_shot_decompositions.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_few_shot_examples(examples: list, hop_count: int | None = None) -> str:
    """Format (question, decomposition) pairs for the prompt. Omit hop line when unguided (hop_count is None)."""
    blocks = []
    for ex in examples:
        if hop_count is not None:
            block = f"Hop count: {hop_count}\nQuestion: {ex['question']}\nDecomposition:\n{ex['decomposition']}"
        else:
            block = f"Question: {ex['question']}\nDecomposition:\n{ex['decomposition']}"
        blocks.append(block)
    return "\n\n".join(blocks)


def _all_decomposer_items(few_shot_data: dict) -> list:
    """Combined pool (1hop, 2hop, 3hop) for fallback sampling."""
    out = []
    for key in ("1hop", "2hop", "3hop"):
        out.extend(few_shot_data.get(key, []))
    return out


def sample_few_shot_combined(few_shot_data: dict, n: int = 3) -> list:
    """Sample n random examples from combined pool. Fallback when similarity unavailable."""
    pool = _all_decomposer_items(few_shot_data)
    if len(pool) <= n:
        return pool
    return random.sample(pool, n)


def get_similar_decomposer(
    question: str,
    mask_fn,
    decomposer_items: list,
    decomposer_embeddings,
    embed_model,
    embed_model_id: str,
    n: int = 3,
) -> tuple[list, list[tuple[dict, float]]]:
    """Get top-n most similar examples by masked similarity. Returns (items, items_with_scores)."""
    from pool_embeddings import top_k_similar_decomposer
    masked_q = mask_fn(question)
    similar = top_k_similar_decomposer(
        masked_q, decomposer_items, decomposer_embeddings,
        embed_model, embed_model_id, k=n,
    )
    return ([it for it, _ in similar], similar)


def build_prompt(template: str, question: str, hop_count: int | None = None, few_shot_examples: str = "") -> str:
    """When hop_count is None (unguided), template must not contain {hop_count}."""
    if hop_count is not None:
        return template.format(question=question, hop_count=hop_count, few_shot_examples=few_shot_examples)
    return template.format(question=question, few_shot_examples=few_shot_examples)

def decompose_question(question: str, hop_count, model, tokenizer, device, prompt_template, config, few_shot_examples: str = "", prompt: str = None):
    if prompt is None:
        prompt = build_prompt(prompt_template, question, hop_count, few_shot_examples)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.get("max_new_tokens", 128),
            temperature=config.get("temperature", 0.0),
            top_p=config.get("top_p", 1.0),
            do_sample=config.get("do_sample", False),
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    
    # Simple cleaning: remove everything after another "Question:" or similar if model hallucinated
    clean_response = response.split("Question:")[0].strip()
    return clean_response

def main():
    parser = argparse.ArgumentParser(description="Run Decomposer Component")
    parser.add_argument("--model_id", type=str, help="Hugging Face model ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_dir", type=str, default="Data", help="Directory containing MetaQA files")
    parser.add_argument("--output_root", type=str, default="runs/decomposer", help="Root directory for runs")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of questions to sample per hop")
    parser.add_argument("--guided", action="store_true", help="Use hop count in prompt")
    parser.add_argument(
        "--embed-model",
        choices=["minilm", "e5-small"],
        default="e5-small",
        help="Embedding model for similarity-based few-shot (default: e5-small)",
    )
    args = parser.parse_args()

    EMBED_MODELS = {
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "e5-small": "intfloat/e5-small-v2",
    }
    embed_model_id = EMBED_MODELS[args.embed_model]

    # 1. Load Local Config
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.json"
    with open(config_path, "r") as f:
        base_config = json.load(f)

    # Merge CLI args into config
    config = {
        "seed": args.seed,
        "model_id": args.model_id or base_config.get("model_id"),
        "model_name": base_config.get("model_name"),
        "data_dir": args.data_dir,
        "output_root": args.output_root,
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "guided": args.guided,
        "embed_model": args.embed_model,
        "max_new_tokens": 128,
        "temperature": base_config.get("decoding", {}).get("temperature", 0.0),
        "top_p": base_config.get("decoding", {}).get("top_p", 1.0),
        "do_sample": base_config.get("decoding", {}).get("do_sample", False),
    }

    set_seed(config["seed"])
    print(f"Starting Decomposer Run: {config['run_id']} (Guided: {config['guided']})")

    # 2. Load Prompt Template (guided vs unguided)
    workspace_root = Path(__file__).resolve().parents[4]
    prompt_file = "prompt_unguided.md" if not config["guided"] else base_config.get("prompt_file", "prompt.md")
    prompt_path = script_dir / prompt_file
    if not prompt_path.exists():
        prompt_path = script_dir / base_config.get("prompt_file", "prompt.md")
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    # 3. Load Data & Few-Shot Pool
    data_path = workspace_root / config["data_dir"]
    few_shot_data = load_few_shot_decompositions(workspace_root)

    # 3b. Load decomposer pool embeddings (masked, combined) and mask_fn
    pool_path = workspace_root / "Pool" / "few_shot_decompositions.json"
    decomposer_items = []
    decomposer_embeddings = None
    embed_model = None
    mask_fn = None
    if pool_path.exists():
        from pool_embeddings import get_decomposer_pool_embeddings
        from sentence_transformers import SentenceTransformer
        from entity_masking import build_masker
        mask_cfg = json.loads((workspace_root / "configs" / "masking.json").read_text())
        mask_fn = build_masker(
            workspace_root / mask_cfg["kb_path"],
            corpus_paths=[workspace_root / p for p in mask_cfg["corpus_paths"]],
            movie_placeholder=mask_cfg.get("movie_placeholder", "[MOVIE]"),
            person_placeholder=mask_cfg.get("person_placeholder", "[PERSON]"),
            repo_root=workspace_root,
        )
        print(f"Loading decomposer pool embeddings ({args.embed_model}, masked)...")
        decomposer_items, decomposer_embeddings, embed_model = get_decomposer_pool_embeddings(
            pool_path, model_id=embed_model_id
        )
    
    hops = [1, 2, 3]
    all_questions = []
    all_expected_hops = []
    
    for h in hops:
        qs = load_questions(data_path / f"refined_{h}hop.txt")
        if args.sample_size:
            qs = random.sample(qs, min(len(qs), args.sample_size))
        all_questions.extend(qs)
        all_expected_hops.extend([h] * len(qs))

    if not all_questions:
        print(f"Error: No questions loaded from {data_path}. Check data paths.")
        return

    print(f"Loaded {len(all_questions)} total questions.")

    # 4. Model Setup
    print(f"Loading model: {config['model_id']} on {config['device']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], trust_remote_code=False)
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        torch_dtype=torch.float16 if config["device"] == "cuda" else torch.float32,
        device_map="auto" if config["device"] == "cuda" else None,
        trust_remote_code=False,
    )
    if config["device"] != "cuda":
        model = model.to(config["device"])
    model.eval()

    # 5. Inference
    output_dir = workspace_root / config["output_root"] / config["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = output_dir / "prompts_log"
    prompts_dir.mkdir(exist_ok=True)

    results = []
    print("Running inference...")
    for i, (q, hop) in enumerate(zip(all_questions, all_expected_hops)):
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(all_questions)}...")

        hop_input = hop if config["guided"] else None
        sampled_with_scores = []
        if decomposer_items and decomposer_embeddings is not None and embed_model and mask_fn:
            sampled, sampled_with_scores = get_similar_decomposer(
                q, mask_fn, decomposer_items, decomposer_embeddings,
                embed_model, embed_model_id, n=3,
            )
        else:
            sampled = sample_few_shot_combined(few_shot_data, n=3)
        few_shot_str = format_few_shot_examples(sampled, hop_input)
        prompt = build_prompt(prompt_template, q, hop_input, few_shot_str)
        decomposition = decompose_question(
            q, hop_input, model, tokenizer, config["device"],
            prompt_template, config, few_shot_examples=few_shot_str,
            prompt=prompt,
        )
        if (i + 1) % 5 == 0:
            log_path = prompts_dir / f"prompt_idx{i+1:04d}_hop{hop}.txt"
            masked_q = mask_fn(q) if mask_fn else "N/A"
            header = (
                "--- Log Header ---\n"
                f"Question (original): {q}\n"
                f"Question (masked): {masked_q}\n"
                "Top 3 similar examples:\n"
            )
            for j, (it, score) in enumerate(sampled_with_scores, 1):
                header += f"  {j}. sim={score:.4f} | masked={it['masked']} | question={it['question']}\n"
            if not sampled_with_scores:
                header += "  (random sampling, no similarity scores)\n"
            log_content = header + "\n--- Prompt + Response ---\n" + prompt + "\n" + decomposition
            log_path.write_text(log_content, encoding="utf-8")
            print(f"  Logged prompt+response to {log_path.name}")

        results.append({
            "question": q,
            "hop_count": hop,
            "decomposition": decomposition
        })

    # 6. Save Artifacts
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    notes = f"""Decomposer Run - {config['run_id']}
Model: {config['model_id']}
Guided: {config['guided']}
Total Questions: {len(results)}
"""
    with open(output_dir / "notes.md", "w") as f:
        f.write(notes)

    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
