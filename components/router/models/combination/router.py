import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_questions(file_path: Path) -> List[str]:
    if not file_path.exists():
        print(f"Warning: {file_path} not found.")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_prompt(template: str, question: str) -> str:
    # Support both {question} and {{question}}
    if "{{question}}" in template:
        return template.replace("{{question}}", question)
    return template.format(question=question)


def stable_softmax(logits: torch.Tensor) -> torch.Tensor:
    # logits: [K]
    m = torch.max(logits)
    exps = torch.exp(logits - m)
    return exps / torch.sum(exps)


# -----------------------------
# Label scoring (log-likelihood)
# -----------------------------
@torch.no_grad()
def score_labels_logprob(
    model,
    tokenizer,
    prompt: str,
    labels: List[str],
    device: str,
) -> torch.Tensor:
    """
    Returns a tensor of shape [len(labels)] with log P(label | prompt)
    computed via next-token (or multi-token) conditional log-likelihood.

    Works even if label tokenizes into multiple tokens (rare for "1","2","3",
    but we handle it correctly).
    """
    # Encode prompt once
    enc_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc_prompt["input_ids"].to(device)
    attn_mask = enc_prompt.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    # We'll compute token-by-token logprob by feeding prompt + previous label tokens.
    # This is exact conditional likelihood under the model.
    scores = []

    for lab in labels:
        # Tokenize label WITHOUT adding BOS/EOS
        lab_ids = tokenizer(lab, add_special_tokens=False)["input_ids"]
        if len(lab_ids) == 0:
            scores.append(torch.tensor(-1e9, device=device))
            continue

        # Start with prompt context
        cur_ids = input_ids
        cur_mask = attn_mask
        total_logprob = torch.tensor(0.0, device=device)

        for t_id in lab_ids:
            out = model(input_ids=cur_ids, attention_mask=cur_mask)
            # Next-token distribution is last position
            next_logits = out.logits[:, -1, :]  # [1, vocab]
            log_probs = torch.log_softmax(next_logits, dim=-1)  # [1, vocab]
            total_logprob += log_probs[0, t_id]

            # Append the chosen token to context
            t = torch.tensor([[t_id]], device=device)
            cur_ids = torch.cat([cur_ids, t], dim=1)
            if cur_mask is not None:
                cur_mask = torch.cat([cur_mask, torch.ones_like(t)], dim=1)

        scores.append(total_logprob)

    return torch.stack(scores, dim=0)  # [K]


def predict_with_confidence(
    model,
    tokenizer,
    prompt: str,
    labels: List[str],
    device: str,
) -> Tuple[int, float, Dict[str, float]]:
    """
    Returns:
      pred_label_int (1/2/3),
      confidence (max prob),
      probs dict mapping label->prob
    """
    logps = score_labels_logprob(model, tokenizer, prompt, labels, device)  # [K]
    probs = stable_softmax(logps).detach().cpu().tolist()

    best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    pred = int(labels[best_idx])
    conf = float(probs[best_idx])
    prob_map = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return pred, conf, prob_map


# -----------------------------
# Ensemble: Argmax over model confidence
# -----------------------------
@dataclass
class ModelSpec:
    name: str
    model_id: str


def load_model_and_tokenizer(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)

    # Ensure pad token exists (needed for some models; not strictly required here but safe)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if device == "cuda":
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=False,
        )
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=False,
        ).to(device)

    mdl.eval()
    return mdl, tok


def main():
    parser = argparse.ArgumentParser("Argmax ensemble router")
    parser.add_argument("--seed", type=int, default=42)

    # Data / prompt
    parser.add_argument("--data_dir", type=str, default="Data")
    parser.add_argument("--output_root", type=str, default="runs/router_ensemble")
    parser.add_argument("--prompt_file", type=str, default="prompt.md")

    # Models
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--phi_model", type=str, default="microsoft/Phi-4-mini-instruct")
    parser.add_argument("--mistral_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")

    # Optional sampling
    parser.add_argument("--sample_size", type=int, default=None)

    args = parser.parse_args()

    set_seed(args.seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    workspace_root = Path(__file__).resolve().parents[4]
    script_dir = Path(__file__).resolve().parent
    data_path = (workspace_root / args.data_dir).resolve()

    # Load prompt
    prompt_path = (script_dir / args.prompt_file).resolve()
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    prompt_template = prompt_path.read_text(encoding="utf-8")

    # Load data
    q1 = load_questions(data_path / "refined_1hop.txt")
    q2 = load_questions(data_path / "refined_2hop.txt")
    q3 = load_questions(data_path / "refined_3hop.txt")

    if args.sample_size:
        q1 = random.sample(q1, min(len(q1), args.sample_size))
        q2 = random.sample(q2, min(len(q2), args.sample_size))
        q3 = random.sample(q3, min(len(q3), args.sample_size))

    all_questions = q1 + q2 + q3
    all_expected = [1] * len(q1) + [2] * len(q2) + [3] * len(q3)
    if not all_questions:
        raise RuntimeError(f"No questions found under {data_path}")

    print(f"Run: {run_id}")
    print(f"Device: {device}")
    print(f"Questions: {len(all_questions)} (1h={len(q1)},2h={len(q2)},3h={len(q3)})")

    model_specs = [
        ModelSpec("qwen_3b", args.qwen_model),
        ModelSpec("phi4_mini", args.phi_model),
        ModelSpec("mistral_7b", args.mistral_model),
    ]

    # Load all models (if VRAM is tight, see note below to load sequentially)
    models = {}
    for spec in model_specs:
        print(f"Loading: {spec.name} => {spec.model_id}")
        mdl, tok = load_model_and_tokenizer(spec.model_id, device)
        models[spec.name] = (mdl, tok, spec.model_id)

    labels = ["1", "2", "3"]

    # Inference
    detailed = []
    correct = 0

    for i, (q, expected) in enumerate(zip(all_questions, all_expected), start=1):
        prompt = build_prompt(prompt_template, q)

        # Get per-model predictions + confidence
        per_model = {}
        best_model = None
        best_conf = -1.0
        best_pred = None

        for spec in model_specs:
            mdl, tok, mid = models[spec.name]
            pred, conf, prob_map = predict_with_confidence(mdl, tok, prompt, labels, device)
            per_model[spec.name] = {
                "model_id": mid,
                "pred": pred,
                "confidence": conf,
                "probs": prob_map,
            }
            if conf > best_conf:
                best_conf = conf
                best_model = spec.name
                best_pred = pred

        is_correct = (best_pred == expected)
        correct += int(is_correct)

        detailed.append({
            "question": q,
            "expected": expected,
            "ensemble_pred": best_pred,
            "ensemble_model": best_model,
            "ensemble_confidence": best_conf,
            "correct": is_correct,
            "per_model": per_model,
        })

        if i % 25 == 0:
            print(f"Processed {i}/{len(all_questions)}")

    accuracy = correct / len(all_questions)
    print(f"\nEnsemble accuracy: {accuracy:.4f} ({correct}/{len(all_questions)})")

    # Save artifacts
    out_dir = (workspace_root / args.output_root / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "run_id": run_id,
        "seed": args.seed,
        "device": device,
        "n": len(all_questions),
        "accuracy": accuracy,
        "models": {spec.name: spec.model_id for spec in model_specs},
        "ensemble_rule": "argmax_over_model_confidence",
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "detailed_results.json").write_text(json.dumps(detailed, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
