#!/usr/bin/env python3
"""
Test similarity-based few-shot selection.

Shows a few questions and their top 6 most similar pool examples.
Run from repo root. First run will download the embedding model and build cache.

Usage:
    python scripts/test_similarity_few_shot.py [--model MODEL_ID]
    python scripts/test_similarity_few_shot.py --compare   # Run both MiniLM and E5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from pool_embeddings import get_pool_embeddings, top_k_similar

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "e5-small": "intfloat/e5-small-v2",
}


def run_test(pool_embeddings, test_cases, pool_questions, model_id: str) -> None:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id)
    for hop_key, query in test_cases:
        print(f"\n>>> ORIGINAL QUESTION ({hop_key}):")
        print(f"    {query}")
        print(f"\n    Top 6 most similar from pool:")
        exclude = query if query in pool_questions else None
        similar = top_k_similar(
            query, hop_key, pool_embeddings, k=6,
            exclude_question=exclude,
            model_id=model_id,
            model=model,
        )
        for i, (item, sim) in enumerate(similar, 1):
            q = item["question"]
            print(f"    {i}. [{sim:.3f}] {q[:70]}{'…' if len(q) > 70 else ''}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test similarity few-shot selection")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="minilm",
        help="Embedding model: minilm or e5-small (default: minilm)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both MiniLM and E5 for side-by-side comparison",
    )
    args = parser.parse_args()

    pool_path = repo_root / "Pool" / "few_shot_decompositions.json"
    if not pool_path.exists():
        print(f"Error: {pool_path} not found")
        return

    models_to_run = list(MODELS.keys()) if args.compare else [args.model]

    test_cases = [
        ("2hop", "which movies have the same actor of Jack the Bear"),
        ("2hop", "what are the genres of the movies acted by Velibor Topic"),
        ("3hop", "who starred in the films whose screenwriters also wrote Young Goethe in Love"),
        ("3hop", "what are the languages spoken in the movies directed by the A Man Apart director"),
    ]
    refined_2 = repo_root / "Data" / "refined_2hop.txt"
    pool_questions = set()

    for model_key in models_to_run:
        model_id = MODELS[model_key]
        print(f"\n{'=' * 80}")
        print(f"  MODEL: {model_key} ({model_id})")
        print("=" * 80)
        print(f"Loading pool and building embeddings...")
        pool_embeddings = get_pool_embeddings(pool_path, model_id=model_id)
        if not pool_questions:
            for hop_key in ("2hop", "3hop"):
                if hop_key in pool_embeddings:
                    pool_questions |= {it["question"] for it in pool_embeddings[hop_key][0]}
            if refined_2.exists() and (line := refined_2.read_text(encoding="utf-8").strip().split("\n")[0]):
                test_cases.append(("2hop", line))
        run_test(pool_embeddings, test_cases, pool_questions, model_id)


if __name__ == "__main__":
    main()
