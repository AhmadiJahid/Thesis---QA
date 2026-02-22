import argparse
import json
import random
import re
import statistics
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def build_prompt(template: str, question: str) -> str:
    # Handle both {question} and {{question}} formats
    if "{{question}}" in template:
        return template.replace("{{question}}", question)
    return template.format(question=question)

def classify_hop_count(question: str, model, tokenizer, device, prompt_template, config) -> int:
    prompt = build_prompt(prompt_template, question)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.get("max_new_tokens", 64),
            temperature=config.get("temperature", 0.0),
            top_p=config.get("top_p", 1.0),
            do_sample=config.get("do_sample", False),
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    try:
        # Clean response (remove reasoning/thought tags if present)
        clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        
        # Take the first line or look for the first digit
        clean_response = clean_response.strip()
        
        # Look for "Output: X" or "A: X" pattern
        match = re.search(r"(?:Output|A):\s*([123])", clean_response, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Fallback to looking for the first digit in the response
        match = re.search(r"[123]", clean_response)
        if match:
            return int(match.group())

        # Fallback keyword matching
        lower_resp = clean_response.lower()
        if "1-hop" in lower_resp or "one hop" in lower_resp: return 1
        if "2-hop" in lower_resp or "two hop" in lower_resp: return 2
        if "3-hop" in lower_resp or "three hop" in lower_resp: return 3

        # Last resort: any digit in the whole response
        numbers = re.findall(r"[123]", clean_response)
        if numbers:
            return int(numbers[0])

        print(f"Warning: No valid hop count in response for: '{question[:50]}...'. Defaulting to 2.")
        return 2
    except Exception as e:
        print(f"Error parsing response for '{question[:50]}...': {e}. Defaulting to 2.")
        return 2

def compute_metrics(predictions, all_expected, run_seed, config, run_idx=None):
    """Compute metrics for one run. Returns dict suitable for metrics.json."""
    correct = sum(1 for p, e in zip(predictions, all_expected) if p == e)
    accuracy = correct / len(predictions) if predictions else 0.0
    per_hop_acc = {}
    for h in [1, 2, 3]:
        total = all_expected.count(h)
        match = sum(1 for p, e in zip(predictions, all_expected) if e == h and p == e)
        per_hop_acc[f"hop_{h}_accuracy"] = match / total if total > 0 else 0.0
        per_hop_acc[f"hop_{h}_total"] = total
    metrics = {
        "overall_accuracy": accuracy,
        "total_questions": len(predictions),
        "correct_predictions": correct,
        **per_hop_acc,
        "seed": run_seed,
        "model": config["model_id"],
        "run_id": config["run_id"],
    }
    if run_idx is not None:
        metrics["run_index"] = run_idx
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Run Router Component (Hop Count Classification)")
    parser.add_argument("--model_id", type=str, help="Hugging Face model ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_dir", type=str, default="Data", help="Directory containing MetaQA files")
    parser.add_argument("--output_root", type=str, default="runs/router", help="Root directory for runs")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of questions to sample per hop")
    parser.add_argument("--prompt_file", type=str, default=None, help="Prompt file (e.g. prompt_zero_shot.md); overrides config for zero-shot")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs to average (seeds seed, seed+1, ...)")
    args = parser.parse_args()

    # 1. Load Local Config
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.json"
    with open(config_path, "r") as f:
        base_config = json.load(f)

    prompt_file = args.prompt_file or base_config.get("prompt_file", "prompt.md")

    # Merge CLI args into config
    config = {
        "seed": args.seed,
        "model_id": args.model_id or base_config.get("model_id") or "microsoft/Phi-3.5-mini-instruct",
        "model_name": base_config.get("model_name", "phi_3_5_mini_instruct"),
        "data_dir": args.data_dir,
        "output_root": args.output_root,
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_new_tokens": 64,
        "temperature": base_config.get("decoding", {}).get("temperature", 0.0),
        "top_p": base_config.get("decoding", {}).get("top_p", 1.0),
        "do_sample": base_config.get("decoding", {}).get("do_sample", False),
        "prompt_file": prompt_file,
        "num_runs": args.num_runs,
    }
    if config["output_root"] == "runs/router":
        config["output_root"] = "runs/average_zero_shot" if "zero_shot" in prompt_file else "runs/average_few_shot"

    print(f"Starting Run: {config['run_id']} (num_runs={config['num_runs']})")
    print(f"Config: {json.dumps(config, indent=2)}")

    # 2. Load Prompt Template (model dir or shared models dir)
    prompt_path = script_dir / prompt_file
    if not prompt_path.exists():
        prompt_path = script_dir.parent / prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file} in {script_dir} or {script_dir.parent}")
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    # 3. Load Data
    # Path is: components/router/models/phi_3_5_mini_instruct/router.py
    workspace_root = Path(__file__).resolve().parents[4]
    data_path = workspace_root / config["data_dir"]
    
    questions_1hop = load_questions(data_path / "refined_1hop.txt")
    questions_2hop = load_questions(data_path / "refined_2hop.txt")
    questions_3hop = load_questions(data_path / "refined_3hop.txt")

    if args.sample_size:
        questions_1hop = random.sample(questions_1hop, min(len(questions_1hop), args.sample_size))
        questions_2hop = random.sample(questions_2hop, min(len(questions_2hop), args.sample_size))
        questions_3hop = random.sample(questions_3hop, min(len(questions_3hop), args.sample_size))

    all_questions = questions_1hop + questions_2hop + questions_3hop
    all_expected = [1]*len(questions_1hop) + [2]*len(questions_2hop) + [3]*len(questions_3hop)

    if not all_questions:
        print(f"Error: No questions loaded from {data_path}. Check data paths.")
        return

    print(f"Loaded {len(all_questions)} total questions (1h: {len(questions_1hop)}, 2h: {len(questions_2hop)}, 3h: {len(questions_3hop)})")

    # 4. Model Setup
    print(f"Loading model: {config['model_id']} on {config['device']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], trust_remote_code=False, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        trust_remote_code=False,
        torch_dtype=torch.float16 if config["device"] == "cuda" else torch.float32,
        device_map="auto" if config["device"] == "cuda" else None,
    )
    if config["device"] != "cuda":
        model = model.to(config["device"])

    model.eval()

    num_runs = config["num_runs"]
    all_runs_metrics = []
    all_runs_predictions = []

    for run_idx in range(num_runs):
        run_seed = config["seed"] + run_idx
        set_seed(run_seed)
        print(f"\n--- Run {run_idx + 1}/{num_runs} (seed={run_seed}) ---")
        predictions = []
        for i, q in enumerate(all_questions):
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(all_questions)}...")
            pred = classify_hop_count(q, model, tokenizer, config["device"], prompt_template, config)
            predictions.append(pred)
        all_runs_predictions.append(predictions)
        run_metrics = compute_metrics(predictions, all_expected, run_seed, config, run_idx=run_idx)
        all_runs_metrics.append(run_metrics)
        print(f"Run {run_idx + 1} accuracy: {run_metrics['overall_accuracy']:.4f}")

    # 6. Aggregate (when num_runs > 1)
    if num_runs == 1:
        metrics = all_runs_metrics[0]
        predictions = all_runs_predictions[0]
        print("\n" + "="*30)
        print(f"Accuracy: {metrics['overall_accuracy']:.4f}")
        for h in [1, 2, 3]:
            print(f"{h}-hop: {metrics[f'hop_{h}_accuracy']:.4f}")
        print("="*30)
    else:
        agg = {
            "num_runs": num_runs,
            "overall_accuracy_mean": statistics.mean(m["overall_accuracy"] for m in all_runs_metrics),
            "overall_accuracy_std": statistics.stdev(m["overall_accuracy"] for m in all_runs_metrics) if num_runs > 1 else 0.0,
            "model": config["model_id"],
            "run_id": config["run_id"],
        }
        for h in [1, 2, 3]:
            accs = [m[f"hop_{h}_accuracy"] for m in all_runs_metrics]
            agg[f"hop_{h}_accuracy_mean"] = statistics.mean(accs)
            agg[f"hop_{h}_accuracy_std"] = statistics.stdev(accs) if num_runs > 1 else 0.0
        metrics = agg
        print("\n" + "="*30)
        print(f"Accuracy (mean ± std): {metrics['overall_accuracy_mean']:.4f} ± {metrics['overall_accuracy_std']:.4f}")
        for h in [1, 2, 3]:
            print(f"{h}-hop: {metrics[f'hop_{h}_accuracy_mean']:.4f} ± {metrics[f'hop_{h}_accuracy_std']:.4f}")
        print("="*30)

    # 7. Save Artifacts
    output_dir = workspace_root / config["output_root"] / config["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    if num_runs == 1:
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        detailed_results = [
            {"question": q, "expected": e, "predicted": p, "correct": p == e}
            for q, e, p in zip(all_questions, all_expected, predictions)
        ]
        with open(output_dir / "detailed_results.json", "w") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        notes = f"""Router Component Run - {config['run_id']}
Model: {config['model_id']}
Overall Accuracy: {metrics['overall_accuracy']:.4f}
"""
    else:
        for run_idx, run_metrics in enumerate(all_runs_metrics):
            with open(output_dir / f"metrics_run_{run_idx}.json", "w") as f:
                json.dump(run_metrics, f, indent=2)
        with open(output_dir / "metrics_aggregated.json", "w") as f:
            json.dump(metrics, f, indent=2)
        detailed_results = [
            {"question": q, "expected": e, "predicted": p, "correct": p == e}
            for q, e, p in zip(all_questions, all_expected, all_runs_predictions[0])
        ]
        with open(output_dir / "detailed_results_run_0.json", "w") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        notes = f"""Router Component Run - {config['run_id']} ({num_runs} runs averaged)
Model: {config['model_id']}
Overall Accuracy (mean ± std): {metrics['overall_accuracy_mean']:.4f} ± {metrics['overall_accuracy_std']:.4f}
"""
    with open(output_dir / "notes.md", "w") as f:
        f.write(notes)

    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
