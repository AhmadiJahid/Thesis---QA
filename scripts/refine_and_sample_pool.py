#!/usr/bin/env python3
"""
Refine Pool training data and sample questions for evaluation.

1. Reads qa_train_{1,2,3}_hop.txt from Pool/
2. Cleans brackets (e.g. [entity] -> entity) and removes answers (tab-separated)
3. Writes refined files: qa_train_{1,2,3}hop_refined.txt
4. Randomly samples 30 questions from each refined file
5. Writes pool files: 1hop_pool.txt, 2hop_pool.txt, 3hop_pool.txt

Usage:
    python scripts/refine_and_sample_pool.py [--pool-dir DIR] [--seed N] [--sample-size N]
"""

import argparse
import json
import re
import random
from pathlib import Path


def clean_brackets(text: str) -> str:
    """Remove square brackets but keep the content inside."""
    return re.sub(r"\[([^\]]*)\]", r"\1", text)


def refine_line(line: str) -> str | None:
    """
    Extract question, clean brackets, drop answers.
    Returns None for empty/invalid lines.
    """
    line = line.strip()
    if not line:
        return None
    parts = line.split("\t", 1)
    question = parts[0]
    cleaned = clean_brackets(question)
    return cleaned.strip() if cleaned else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine Pool data and sample questions.")
    parser.add_argument(
        "--pool-dir",
        type=Path,
        default=Path("Pool"),
        help="Directory containing qa_train_*_hop.txt",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="Number of questions to sample per hop",
    )
    args = parser.parse_args()

    pool_dir = args.pool_dir
    if not pool_dir.is_absolute():
        pool_dir = Path.cwd() / pool_dir

    config = {
        "pool_dir": str(pool_dir),
        "seed": args.seed,
        "sample_size": args.sample_size,
    }

    hop_configs = [
        ("qa_train_1_hop.txt", "qa_train_1hop_refined.txt", "1hop_pool.txt"),
        ("qa_train_2_hop.txt", "qa_train_2hop_refined.txt", "2hop_pool.txt"),
        ("qa_train_3_hop.txt", "qa_train_3hop_refined.txt", "3hop_pool.txt"),
    ]

    metrics = {}

    for src_name, refined_name, pool_name in hop_configs:
        src_path = pool_dir / src_name
        refined_path = pool_dir / refined_name
        pool_path = pool_dir / pool_name

        if not src_path.exists():
            raise FileNotFoundError(f"Input file not found: {src_path}")

        # Read, refine, write
        questions = []
        with open(src_path) as f:
            for line in f:
                q = refine_line(line)
                if q:
                    questions.append(q)

        with open(refined_path, "w") as f:
            f.write("\n".join(questions) + "\n")

        # Sample
        random.seed(args.seed)
        n = min(args.sample_size, len(questions))
        sampled = random.sample(questions, n)

        with open(pool_path, "w") as f:
            f.write("\n".join(sampled) + "\n")

        hop_label = src_name.replace("qa_train_", "").replace(".txt", "")
        metrics[hop_label] = {
            "input_lines": len(questions),
            "refined_path": str(refined_path),
            "pool_path": str(pool_path),
            "sampled_count": n,
        }

    # Log config + metrics (thesis-core: reproducibility)
    config_path = pool_dir / "refine_and_sample_config.json"
    with open(config_path, "w") as f:
        json.dump({"config": config, "metrics": metrics}, f, indent=2)

    print(f"Seed: {args.seed}")
    print(f"Refined: {', '.join(m['refined_path'].split('/')[-1] for m in metrics.values())}")
    print(f"Pools: {', '.join(m['pool_path'].split('/')[-1] for m in metrics.values())}")
    for k, v in metrics.items():
        print(f"  {k}: {v['input_lines']} questions -> sampled {v['sampled_count']}")


if __name__ == "__main__":
    main()
