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


def format_few_shot_examples(examples: list, hop_count: int) -> str:
    """Format (question, decomposition) pairs for the prompt."""
    blocks = []
    for ex in examples:
        blocks.append(
            f"Hop count: {hop_count}\n"
            f"Question: {ex['question']}\n"
            f"Decomposition:\n{ex['decomposition']}"
        )
    return "\n\n".join(blocks)


def sample_few_shot(few_shot_data: dict, hop_count: int, n: int = 2) -> list:
    """Sample n random examples for the given hop. Fallback when similarity unavailable."""
    key = f"{hop_count}hop"
    pool = few_shot_data.get(key, [])
    if len(pool) <= n:
        return pool
    rng = random.Random()
    return rng.sample(pool, n)


def get_similar_few_shot(
    question: str,
    hop_count: int,
    pool_embeddings: dict,
    few_shot_data: dict,
    embed_model,
    embed_model_id: str,
    n: int = 2,
) -> list:
    """Get top-n most similar examples from pool. Fallback to random if unavailable."""
    from pool_embeddings import top_k_similar
    key = f"{hop_count}hop"
    similar = top_k_similar(
        question, key, pool_embeddings, k=n,
        model_id=embed_model_id, model=embed_model,
    )
    items = [it for it, _ in similar]
    if len(items) >= n:
        return items
    # Fallback: fill remaining with random from pool
    pool = few_shot_data.get(key, [])
    existing_qs = {it["question"] for it in items}
    remaining = [x for x in pool if x["question"] not in existing_qs]
    need = n - len(items)
    if need > 0 and remaining:
        extra = random.sample(remaining, min(need, len(remaining)))
        items.extend(extra)
    return items[:n]


def build_prompt(template: str, question: str, hop_count: int = None, few_shot_examples: str = "") -> str:
    h = hop_count if hop_count is not None else "Unknown"
    return template.format(question=question, hop_count=h, few_shot_examples=few_shot_examples)

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

    # 2. Load Prompt Template
    prompt_path = script_dir / base_config.get("prompt_file", "prompt.md")
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    # 3. Load Data & Few-Shot Pool
    workspace_root = Path(__file__).resolve().parents[4]
    data_path = workspace_root / config["data_dir"]
    few_shot_data = load_few_shot_decompositions(workspace_root)

    # 3b. Load pool embeddings for similarity-based few-shot
    pool_path = workspace_root / "Pool" / "few_shot_decompositions.json"
    if pool_path.exists():
        from pool_embeddings import get_pool_embeddings
        from sentence_transformers import SentenceTransformer
        print(f"Loading embeddings ({args.embed_model}) for similarity few-shot...")
        pool_embeddings = get_pool_embeddings(pool_path, model_id=embed_model_id)
        embed_model = SentenceTransformer(embed_model_id)
    else:
        pool_embeddings = {}
        embed_model = None
    
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

        if hop == 1:
            decomposition = q
        else:
            hop_input = hop
            if pool_embeddings and embed_model:
                sampled = get_similar_few_shot(
                    q, hop, pool_embeddings, few_shot_data,
                    embed_model, embed_model_id, n=2,
                )
            else:
                sampled = sample_few_shot(few_shot_data, hop, n=2)
            few_shot_str = format_few_shot_examples(sampled, hop)
            prompt = build_prompt(prompt_template, q, hop_input, few_shot_str)
            decomposition = decompose_question(
                q, hop_input, model, tokenizer, config["device"],
                prompt_template, config, few_shot_examples=few_shot_str,
                prompt=prompt,
            )
            if (i + 1) % 5 == 0:
                log_path = prompts_dir / f"prompt_idx{i+1:04d}_hop{hop}.txt"
                log_content = prompt + "\n" + decomposition
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
