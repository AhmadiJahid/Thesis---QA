#!/usr/bin/env python3
"""
Test similarity-based few-shot selection.

Shows a few questions and their top 6 most similar pool examples.
Run from repo root. First run will download the embedding model and build cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from pool_embeddings import get_pool_embeddings, top_k_similar


def main() -> None:
    pool_path = repo_root / "Pool" / "few_shot_decompositions.json"
    if not pool_path.exists():
        print(f"Error: {pool_path} not found")
        return

    print("Loading pool and building embeddings (first run may download model + create cache)...")
    pool_embeddings = get_pool_embeddings(pool_path)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Test questions: 2 from pool (2hop), 2 from pool (3hop), 1 from refined data
    test_cases = [
        ("2hop", "who is listed as director of Ashley Olsen starred movies"),
        ("2hop", "what are the genres of the movies acted by Velibor Topic"),
        ("3hop", "who starred in the films whose screenwriters also wrote Young Goethe in Love"),
        ("3hop", "what are the languages spoken in the movies directed by the A Man Apart director"),
    ]

    # Add one from refined if available (not in pool, so no exclusion)
    refined_2 = repo_root / "Data" / "refined_2hop.txt"
    pool_questions = set()
    for hop_key in ("2hop", "3hop"):
        if hop_key in pool_embeddings:
            pool_questions |= {it["question"] for it in pool_embeddings[hop_key][0]}
    if refined_2.exists() and (line := refined_2.read_text(encoding="utf-8").strip().split("\n")[0]):
        test_cases.append(("2hop", line))

    print("\n" + "=" * 80)
    for hop_key, query in test_cases:
        print(f"\n>>> ORIGINAL QUESTION ({hop_key}):")
        print(f"    {query}")
        print(f"\n    Top 6 most similar from pool:")
        # Exclude query from results when it's a pool question (avoid self-match)
        exclude = query if query in pool_questions else None
        similar = top_k_similar(
            query, hop_key, pool_embeddings, k=6,
            exclude_question=exclude,
            model=model,
        )
        for i, (item, sim) in enumerate(similar, 1):
            q = item["question"]
            print(f"    {i}. [{sim:.3f}] {q[:70]}{'…' if len(q) > 70 else ''}")
        print()


if __name__ == "__main__":
    main()
