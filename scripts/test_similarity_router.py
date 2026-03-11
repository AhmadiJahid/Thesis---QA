#!/usr/bin/env python3
"""
Test similarity for router few-shot pool (hop count unknown).

Three-way comparison using refined questions:
  A: Question NOT masked -> Pool NOT masked
  B: Question NOT masked -> Pool MASKED
  C: Question MASKED -> Pool MASKED

Uses top-3 majority vote for hop prediction.
Requires configs/masking.json for mask_fn (modes B & C).
Outputs to runs/ (metrics JSON + human-readable report).
Run from repo root.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from pool_embeddings import get_router_pool_embeddings, top_k_similar_router
from entity_masking import build_masker

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "e5-small": "intfloat/e5-small-v2",
}

# TEST_QUESTIONS_RAW = [
#     ("who directed The Godfather", 1),
#     ("what are the genres of the movies acted by Tom Hanks", 2),
#     ("who starred in the films whose screenwriters also wrote Pulp Fiction", 3),
#     ("what was the release year of Titanic", 1),
#     ("which person directed the films acted by Meryl Streep", 2),
#     ("when did the films directed by the Inception director release", 3),
#     ("what languages are the movies that share actors with Forrest Gump in", 3),
#     ("describe The Shawshank Redemption", 1),
#     ("who are the directors of the movies written by Christopher Nolan", 2),
#     ("what genres do the movies that share directors with Avatar fall under", 3),
# ]
# TEST_QUESTIONS_MASKED = [
#     ("who directed [MOVIE]", 1),
#     ("what are the genres of the movies acted by [PERSON]", 2),
#     ("who starred in the films whose screenwriters also wrote [MOVIE]", 3),
#     ("what was the release year of [MOVIE]", 1),
#     ("which person directed the films acted by [PERSON]", 2),
#     ("when did the films directed by the [MOVIE] director release", 3),
#     ("what languages are the movies that share actors with [MOVIE] in", 3),
#     ("describe [MOVIE]", 1),
#     ("who are the directors of the movies written by [PERSON]", 2),
#     ("what genres do the movies that share directors with [MOVIE] fall under", 3),
# ]


def load_refined_questions(data_dir: Path) -> list[tuple[str, int]]:
    """Load refined questions from refined_1hop.txt, refined_2hop.txt, refined_3hop.txt."""
    out: list[tuple[str, int]] = []
    for hop in (1, 2, 3):
        path = data_dir / f"refined_{hop}hop.txt"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            q = line.strip()
            if q:
                out.append((q, hop))
    return out


def run_tests(items, embeddings, model, model_id, test_questions, majority_over_top_k: int = 3):
    """Run all test questions, return list of (majority_hop, match, top_sim) per question."""
    results = []
    for query, og_hop in test_questions:
        similar = top_k_similar_router(query, items, embeddings, model, model_id=model_id, k=6)
        top_for_vote = similar[:majority_over_top_k]
        hop_counts = [it["hop_count"] for it, _ in top_for_vote]
        majority_hop = max(set(hop_counts), key=hop_counts.count) if hop_counts else 0
        top_sim = similar[0][1] if similar else 0.0
        match = majority_hop == og_hop
        results.append((majority_hop, match, top_sim, similar))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-way similarity test: raw/masked question vs raw/masked pool"
    )
    parser.add_argument(
        "--no-corpus-filter",
        action="store_true",
        help="Use full KB for masking (no corpus filtering)",
    )
    args = parser.parse_args()

    pool_raw = repo_root / "Pool" / "few_shot_router.json"
    pool_masked = repo_root / "Pool" / "few_shot_router_masked.json"
    config_path = repo_root / "configs" / "masking.json"
    data_dir = repo_root / "Data"

    if not pool_raw.exists():
        print("Error: Pool file not found:", pool_raw)
        return
    if not pool_masked.exists():
        print("Error: Masked pool not found:", pool_masked)
        return
    if not config_path.exists():
        print("Error: configs/masking.json required for mask_fn")
        return

    refined = load_refined_questions(data_dir)
    if not refined:
        print("Error: No refined questions found in Data/refined_*hop.txt")
        return

    model_key = "e5-small"
    model_id = MODELS[model_key]
    top_k = 3

    cfg = json.loads(config_path.read_text())
    kb_path = repo_root / cfg["kb_path"]
    corpus_paths = None if args.no_corpus_filter else [repo_root / p for p in cfg.get("corpus_paths", [])]
    mask_fn = build_masker(
        kb_path,
        corpus_paths=corpus_paths or None,
        movie_placeholder=cfg.get("movie_placeholder", "[MOVIE]"),
        person_placeholder=cfg.get("person_placeholder", "[PERSON]"),
        repo_root=repo_root,
    )

    runs_dir = repo_root / "runs" / "similarity"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = runs_dir / f"similarity_router_3way_{ts}"

    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log(f"Refined questions: {len(refined)} total")
    log(f"Majority vote: top-{top_k}")
    log()

    # ---- Mode A: raw question -> raw pool ----
    log("=" * 80)
    log("  A: Question NOT masked -> Pool NOT masked")
    log("=" * 80)
    items_a, emb_a, model = get_router_pool_embeddings(pool_raw, model_id=model_id)
    queries_a = [(q, h) for q, h in refined]
    res_a = run_tests(items_a, emb_a, model, model_id, queries_a, majority_over_top_k=top_k)
    correct_a = sum(1 for _, m, _, _ in res_a if m)
    at_least_one_a = sum(1 for (_, og), r in zip(queries_a, res_a) if any(it["hop_count"] == og for it, _ in r[3][:top_k]))
    log(f"Accuracy: {correct_a}/{len(refined)} ({100*correct_a/len(refined):.1f}%)")
    log(f"At least one top-{top_k} has correct hop: {at_least_one_a}/{len(refined)} ({100*at_least_one_a/len(refined):.1f}%)")
    log()

    # ---- Mode B: raw question -> masked pool ----
    log("=" * 80)
    log("  B: Question NOT masked -> Pool MASKED")
    log("=" * 80)
    items_b, emb_b, model = get_router_pool_embeddings(
        pool_masked, model_id=model_id
    )
    queries_b = [(q, h) for q, h in refined]
    res_b = run_tests(items_b, emb_b, model, model_id, queries_b, majority_over_top_k=top_k)
    correct_b = sum(1 for _, m, _, _ in res_b if m)
    at_least_one_b = sum(1 for (_, og), r in zip(queries_b, res_b) if any(it["hop_count"] == og for it, _ in r[3][:top_k]))
    log(f"Accuracy: {correct_b}/{len(refined)} ({100*correct_b/len(refined):.1f}%)")
    log(f"At least one top-{top_k} has correct hop: {at_least_one_b}/{len(refined)} ({100*at_least_one_b/len(refined):.1f}%)")
    log()

    # ---- Mode C: masked question -> masked pool ----
    log("=" * 80)
    log("  C: Question MASKED -> Pool MASKED")
    log("=" * 80)
    items_c, emb_c, model = get_router_pool_embeddings(
        pool_masked, model_id=model_id
    )
    queries_c = [(mask_fn(q), h) for q, h in refined]
    res_c = run_tests(items_c, emb_c, model, model_id, queries_c, majority_over_top_k=top_k)
    correct_c = sum(1 for _, m, _, _ in res_c if m)
    at_least_one_c = sum(1 for (_, og), r in zip(queries_c, res_c) if any(it["hop_count"] == og for it, _ in r[3][:top_k]))
    log(f"Accuracy: {correct_c}/{len(refined)} ({100*correct_c/len(refined):.1f}%)")
    log(f"At least one top-{top_k} has correct hop: {at_least_one_c}/{len(refined)} ({100*at_least_one_c/len(refined):.1f}%)")
    log()

    # ---- Per-hop breakdown ----
    def per_hop_stats(res: list, queries: list[tuple[str, int]]) -> dict:
        by_hop: dict[int, list[bool]] = {}
        for (_, og_hop), (_, match, _, _) in zip(queries, res):
            by_hop.setdefault(og_hop, []).append(match)
        out: dict[int, dict] = {}
        for h in (1, 2, 3):
            lst = by_hop.get(h, [])
            n = len(lst)
            out[h] = {
                "correct": sum(lst),
                "total": n,
                "acc_pct": 100 * sum(lst) / n if n else 0,
            }
        return out

    stats_a = per_hop_stats(res_a, queries_a)
    stats_b = per_hop_stats(res_b, queries_b)
    stats_c = per_hop_stats(res_c, queries_c)

    # ---- Comparison table ----
    log("=" * 80)
    log("  COMPARISON & ANALYSIS")
    log("=" * 80)
    log()
    log("Overall accuracy (majority vote):")
    log(f"  A (raw q, raw pool):   {correct_a}/{len(refined)} ({100*correct_a/len(refined):.1f}%)")
    log(f"  B (raw q, masked pool): {correct_b}/{len(refined)} ({100*correct_b/len(refined):.1f}%)")
    log(f"  C (masked q, masked pool): {correct_c}/{len(refined)} ({100*correct_c/len(refined):.1f}%)")
    log()
    log("At least one top-k has correct hop:")
    log(f"  A: {at_least_one_a}/{len(refined)} ({100*at_least_one_a/len(refined):.1f}%)  "
        f"B: {at_least_one_b}/{len(refined)} ({100*at_least_one_b/len(refined):.1f}%)  "
        f"C: {at_least_one_c}/{len(refined)} ({100*at_least_one_c/len(refined):.1f}%)")
    log()
    log("Per-hop accuracy:")
    log(f"  {'hop':<5} {'A':<12} {'B':<12} {'C':<12}")
    log(f"  {'-'*5} {'-'*12} {'-'*12} {'-'*12}")
    for h in (1, 2, 3):
        sa, sb, sc = stats_a[h], stats_b[h], stats_c[h]
        log(f"  {h:<5} {sa['correct']}/{sa['total']} ({sa['acc_pct']:.0f}%)  "
            f"{sb['correct']}/{sb['total']} ({sb['acc_pct']:.0f}%)  "
            f"{sc['correct']}/{sc['total']} ({sc['acc_pct']:.0f}%)")
    log()

    best = max([
        ("A (raw q, raw pool)", correct_a),
        ("B (raw q, masked pool)", correct_b),
        ("C (masked q, masked pool)", correct_c),
    ], key=lambda x: x[1])
    log(f"Best mode: {best[0]} ({best[1]}/{len(refined)})")
    log()

    avg_sim = lambda res: sum(r[2] for r in res) / len(res) if res else 0
    log("Avg top-1 similarity:")
    log(f"  A: {avg_sim(res_a):.3f}  B: {avg_sim(res_b):.3f}  C: {avg_sim(res_c):.3f}")
    log()

    # ---- Per-question sample (first 15) ----
    log("Per-question (first 15):")
    log(f"  {'#':<4} {'og':<4} {'A':<6} {'B':<6} {'C':<6}  Question")
    log(f"  {'-'*4} {'-'*4} {'-'*6} {'-'*6} {'-'*6}  {'-'*40}")
    for i in range(min(15, len(refined))):
        q, og = refined[i]
        maj_a, m_a = res_a[i][0], res_a[i][1]
        maj_b, m_b = res_b[i][0], res_b[i][1]
        maj_c, m_c = res_c[i][0], res_c[i][1]
        sa, sb, sc = ("✓" if m_a else "✗", "✓" if m_b else "✗", "✓" if m_c else "✗")
        qs = (q[:38] + "…") if len(q) > 38 else q
        log(f"  {i+1:<4} {og:<4} {maj_a}({sa})  {maj_b}({sb})  {maj_c}({sc})  {qs}")
    log()

    # ---- JSON metrics ----
    metrics = {
        "n_questions": len(refined),
        "top_k": top_k,
        "model": model_key,
        "accuracy": {"A": correct_a / len(refined), "B": correct_b / len(refined), "C": correct_c / len(refined)},
        "correct": {"A": correct_a, "B": correct_b, "C": correct_c},
        "at_least_one_match": {"A": at_least_one_a, "B": at_least_one_b, "C": at_least_one_c},
        "at_least_one_match_pct": {"A": at_least_one_a / len(refined), "B": at_least_one_b / len(refined), "C": at_least_one_c / len(refined)},
        "per_hop": {"A": stats_a, "B": stats_b, "C": stats_c},
        "avg_top1_sim": {"A": avg_sim(res_a), "B": avg_sim(res_b), "C": avg_sim(res_c)},
    }
    metrics_path = out_base.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"Metrics: {metrics_path}")

    report_path = out_base.with_suffix(".txt")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"Report: {report_path}")

    # ---- Per-question similar neighbors (to file only) ----
    sim_path = runs_dir / f"similarity_router_3way_{ts}_similarities.txt"
    with sim_path.open("w", encoding="utf-8") as f:
        f.write("Per-question: query, ground-truth hop, and top-3 similar pool items for A/B/C.\n")
        f.write("=" * 80 + "\n\n")
        for i, (q, og_hop) in enumerate(refined):
            masked_q = mask_fn(q)
            f.write(f"[{i+1}] {q}\n")
            f.write(f"    masked: {masked_q}\n")
            f.write(f"    ground_truth: {og_hop}\n")
            for label, res in [
                ("A (raw q, raw pool)", res_a),
                ("B (raw q, masked pool)", res_b),
                ("C (masked q, masked pool)", res_c),
            ]:
                _, _, _, similar = res[i]
                maj = res[i][0]
                f.write(f"    {label} -> majority: {maj}\n")
                for j, (it, sim) in enumerate(similar[:3], 1):
                    f.write(f"      {j}. [hop={it['hop_count']}] sim={sim:.3f}  {it['question']}\n")
            f.write("\n")
    log(f"Similarities: {sim_path}")


if __name__ == "__main__":
    main()
