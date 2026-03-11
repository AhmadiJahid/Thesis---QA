#!/usr/bin/env python3
"""
Test similarity for router few-shot pool (hop count unknown).

Embeds few_shot_router.json, then for 10 test questions (hop unknown),
shows top 6 most similar examples and their hop_count.
Run from repo root.
"""
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from pool_embeddings import get_router_pool_embeddings, top_k_similar_router

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "e5-small": "intfloat/e5-small-v2",
}


def main() -> None:
    pool_path = repo_root / "Pool" / "few_shot_router.json"
    if not pool_path.exists():
        print(f"Error: {pool_path} not found")
        return

    model_key = "e5-small"
    model_id = MODELS[model_key]
    print(f"Loading router pool and building embeddings ({model_key})...")
    items, embeddings, model = get_router_pool_embeddings(pool_path, model_id=model_id)
    print(f"Pool: {len(items)} items (1hop+2hop+3hop combined)")

    # 10 test questions - (question, og_hop) for ground-truth comparison
    test_questions = [
        ("who directed The Godfather", 1),
        ("what are the genres of the movies acted by Tom Hanks", 2),
        ("who starred in the films whose screenwriters also wrote Pulp Fiction", 3),
        ("what was the release year of Titanic", 1),
        ("which person directed the films acted by Meryl Streep", 2),
        ("when did the films directed by the Inception director release", 2),
        ("what languages are the movies that share actors with Forrest Gump in", 2),
        ("describe The Shawshank Redemption", 1),
        ("who are the directors of the movies written by Christopher Nolan", 2),
        ("what genres do the movies that share directors with Avatar fall under", 2),
    ]

    print("\n" + "=" * 80)
    print("  TEST: 10 questions, hop count UNKNOWN – Top 6 similar from pool")
    print("=" * 80)

    for i, (query, og_hop) in enumerate(test_questions, 1):
        similar = top_k_similar_router(query, items, embeddings, model, model_id=model_id, k=6)
        hop_counts = [it["hop_count"] for it, _ in similar]
        majority_hop = max(set(hop_counts), key=hop_counts.count)
        match = "✓" if majority_hop == og_hop else "✗"
        print(f"\n[{i}] Q: {query[:65]}{'…' if len(query) > 65 else ''}")
        print(f"    og_hop: {og_hop}  |  Top 6 similar → hop_counts: {hop_counts}  (majority: {majority_hop})  {match}")
        for j, (it, sim) in enumerate(similar, 1):
            q = it["question"][:55] + "…" if len(it["question"]) > 55 else it["question"]
            print(f"    {j}. [hop={it['hop_count']}] [{sim:.3f}] {q}")
    print()


if __name__ == "__main__":
    main()
