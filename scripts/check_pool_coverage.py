#!/usr/bin/env python3
"""
Check similarity between test questions and their relevant hop-specific pool.

Loads questions from Data/refined_{1,2,3}hop.txt, compares each to the
hop-specific pool in few_shot_router.json. Assesses whether the pool size is
sufficient—low similarity may indicate the pool lacks coverage for that type.
Run from repo root.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from pool_embeddings import get_router_pool_embeddings, load_router_pool
from pool_embeddings import _needs_e5_prefix

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "e5-small": "intfloat/e5-small-v2",
}


def load_test_questions(data_dir: Path) -> list[tuple[str, str]]:
    """Load (question, hop_key) from refined_{1,2,3}hop.txt."""
    out: list[tuple[str, str]] = []
    for hop in (1, 2, 3):
        path = data_dir / f"refined_{hop}hop.txt"
        if not path.exists():
            continue
        hop_key = f"{hop}hop"
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            q = line.strip()
            if q:
                out.append((q, hop_key))
    return out


def top_k_in_pool(
    query: str,
    items: list[dict],
    embeddings: np.ndarray,
    model,
    model_id: str,
    k: int,
) -> list[float]:
    """Return top-k cosine similarities of query vs pool (hop-specific)."""
    use_prefix = _needs_e5_prefix(model_id)
    to_encode = [f"query: {query}"] if use_prefix else [query]
    q_emb = model.encode(to_encode, normalize_embeddings=True)[0]
    scores = np.dot(embeddings, q_emb)
    top_indices = np.argsort(-scores)[:k]
    return [float(scores[i]) for i in top_indices]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check question–pool similarity to assess pool coverage"
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=repo_root / "Pool" / "few_shot_router.json",
        help="Path to few_shot_router.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "Data",
        help="Directory with refined_{1,2,3}hop.txt",
    )
    parser.add_argument(
        "--embed-model",
        choices=list(MODELS),
        default="e5-small",
        help="Embedding model",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample N questions per hop for quick runs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write metrics.json to this path",
    )
    args = parser.parse_args()

    pool_path = args.pool if args.pool.is_absolute() else repo_root / args.pool
    data_dir = args.data_dir if args.data_dir.is_absolute() else repo_root / args.data_dir

    if not pool_path.exists():
        print(f"Error: Pool not found: {pool_path}")
        sys.exit(1)

    model_id = MODELS[args.embed_model]
    print(f"Loading pool and embeddings ({args.embed_model})...")
    pool = load_router_pool(pool_path)
    all_items, all_embeddings, model = get_router_pool_embeddings(
        pool_path, model_id=model_id
    )

    # Split by hop (order: 1hop, 2hop, 3hop)
    offsets = [0]
    for h in ("1hop", "2hop", "3hop"):
        offsets.append(offsets[-1] + len(pool.get(h, [])))
    hop_data: dict[str, tuple[list[dict], np.ndarray]] = {}
    for i, h in enumerate(("1hop", "2hop", "3hop")):
        s, e = offsets[i], offsets[i + 1]
        hop_data[h] = (all_items[s:e], all_embeddings[s:e])

    for h, (items, _) in hop_data.items():
        print(f"  {h}: {len(items)} items")

    questions = load_test_questions(data_dir)
    if not questions:
        print(f"Error: No questions in {data_dir} (refined_*hop.txt)")
        sys.exit(1)

    if args.sample_size:
        rng = random.Random(args.seed)
        by_hop: dict[str, list[str]] = {}
        for q, h in questions:
            by_hop.setdefault(h, []).append(q)
        sampled = []
        for h in ("1hop", "2hop", "3hop"):
            if h in by_hop:
                pool_qs = by_hop[h]
                n = min(args.sample_size, len(pool_qs))
                chosen = rng.sample(pool_qs, n)
                sampled.extend((q, h) for q in chosen)
        rng.shuffle(sampled)
        questions = sampled
        print(f"Sampled {len(questions)} questions (seed={args.seed})")
    else:
        print(f"Loaded {len(questions)} questions from refined_*hop.txt")

    # Compute similarities per question
    results: dict[str, list[dict]] = {"1hop": [], "2hop": [], "3hop": []}

    for query, hop_key in questions:
        items_h, emb_h = hop_data[hop_key]
        if not items_h:
            continue
        k_max = min(6, len(items_h))
        sims = top_k_in_pool(query, items_h, emb_h, model, model_id, k_max)
        rec = {
            "question": query[:80] + ("…" if len(query) > 80 else ""),
            "sim_top1": round(sims[0], 4) if sims else None,
            "sim_top3": round(np.mean(sims[:3]).item(), 4) if len(sims) >= 3 else round(np.mean(sims).item(), 4) if sims else None,
            "sim_top6": round(np.mean(sims[:6]).item(), 4) if len(sims) >= 6 else round(np.mean(sims).item(), 4) if sims else None,
        }
        results[hop_key].append(rec)

    # Summary stats
    print("\n" + "=" * 70)
    print("  POOL COVERAGE: similarity of questions to their hop-specific pool")
    print("=" * 70)

    all_top1: list[float] = []
    for hop_key in ("1hop", "2hop", "3hop"):
        recs = results[hop_key]
        if not recs:
            continue
        top1 = [r["sim_top1"] for r in recs if r["sim_top1"] is not None]
        top3 = [r["sim_top3"] for r in recs if r["sim_top3"] is not None]
        top6 = [r["sim_top6"] for r in recs if r["sim_top6"] is not None]
        all_top1.extend(top1)

        n = len(recs)
        print(f"\n{hop_key} (n={n}):")
        print(f"  sim_top1  min={min(top1):.3f}  mean={np.mean(top1):.3f}  max={max(top1):.3f}")
        if top3:
            print(f"  sim_top3 mean={np.mean(top3):.3f}")
        if top6:
            print(f"  sim_top6 mean={np.mean(top6):.3f}")

        # Count below threshold
        for thresh in (0.5, 0.6, 0.7):
            below = sum(1 for s in top1 if s < thresh)
            pct = 100 * below / n if n else 0
            print(f"  below {thresh}: {below}/{n} ({pct:.0f}%)")

    if all_top1:
        print("\nOverall:")
        print(f"  sim_top1  min={min(all_top1):.3f}  mean={np.mean(all_top1):.3f}  max={max(all_top1):.3f}")

    # Output metrics
    metrics = {
        "seed": args.seed,
        "embed_model": args.embed_model,
        "pool_path": str(pool_path),
        "pool_sizes": {h: len(hop_data[h][0]) for h in ("1hop", "2hop", "3hop")},
        "n_questions": {h: len(results[h]) for h in ("1hop", "2hop", "3hop")},
        "per_hop": {},
    }
    for hop_key in ("1hop", "2hop", "3hop"):
        recs = results[hop_key]
        if not recs:
            continue
        top1 = [r["sim_top1"] for r in recs if r["sim_top1"] is not None]
        metrics["per_hop"][hop_key] = {
            "sim_top1_min": float(min(top1)),
            "sim_top1_mean": float(np.mean(top1)),
            "sim_top1_max": float(max(top1)),
            "below_0.5": sum(1 for s in top1 if s < 0.5),
            "below_0.6": sum(1 for s in top1 if s < 0.6),
            "below_0.7": sum(1 for s in top1 if s < 0.7),
        }
    metrics["overall_sim_top1_mean"] = float(np.mean(all_top1)) if all_top1 else None

    if args.output:
        out_path = args.output if args.output.is_absolute() else repo_root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
