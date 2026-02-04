#!/usr/bin/env python3
"""
Compare kg_results in analysis/success.json with gold answers (Data/answers_*hop.txt).
Uses Jaccard similarity (set overlap) and reports per-hop and overall accuracy stats.

Usage:
  python scripts/compare_answer_accuracy.py RUN_DIR [--data-dir DIR] [--seed N]
  e.g. python scripts/compare_answer_accuracy.py runs/decomposer/20260123_072902

Output:
  - Prints summary to stdout (per-hop: answered, % correct, mean Jaccard, etc.)
  - Writes metrics to RUN_DIR/analysis/answer_metrics.json
  - Writes run note to RUN_DIR/analysis/answer_accuracy_notes.md
  - With --report: writes visualizations to reports/<run_id>/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def load_gold_by_hop(data_dir: Path) -> Tuple[Dict[int, Dict[str, Set[str]]], Dict[int, int]]:
    """Load refined_*hop.txt and answers_*hop.txt; return (gold[hop][question], total_per_hop)."""
    gold: Dict[int, Dict[str, Set[str]]] = {1: {}, 2: {}, 3: {}}
    total_per_hop: Dict[int, int] = {1: 0, 2: 0, 3: 0}
    for hop in (1, 2, 3):
        q_path = data_dir / f"refined_{hop}hop.txt"
        a_path = data_dir / f"answers_{hop}hop.txt"
        if not q_path.exists() or not a_path.exists():
            continue
        questions = [line.strip() for line in q_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        answers = [line.strip() for line in a_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(questions) != len(answers):
            raise ValueError(f"{q_path.name} has {len(questions)} lines but {a_path.name} has {len(answers)}")
        total_per_hop[hop] = len(questions)
        for q, a in zip(questions, answers):
            gold[hop][q] = {x.strip() for x in a.split("|") if x.strip()}
    return gold, total_per_hop


def jaccard(pred: Set[str], gold: Set[str]) -> float:
    """Jaccard similarity = |intersection| / |union|. Empty vs empty => 1.0."""
    if not pred and not gold:
        return 1.0
    union = pred | gold
    if not union:
        return 1.0
    return len(pred & gold) / len(union)


def run_analysis(
    run_dir: Path,
    data_dir: Path,
    seed: int,
) -> Tuple[Dict, List[Dict]]:
    """
    Load success.json and gold; compute per-item Jaccard and aggregate stats.
    Returns (summary_dict, per_item_list).
    """
    analysis_dir = run_dir / "analysis"
    success_path = analysis_dir / "success.json"
    if not success_path.exists():
        raise FileNotFoundError(f"Not found: {success_path}")

    gold_by_hop, total_per_hop = load_gold_by_hop(data_dir)
    with open(success_path, encoding="utf-8") as f:
        success_items = json.load(f)

    # Per-hop aggregates: count, with_gold, jaccard_sum, exact_match_count
    hop_stats: Dict[int, Dict] = {
        1: {"n": 0, "with_gold": 0, "jaccard_sum": 0.0, "exact": 0, "jaccards": []},
        2: {"n": 0, "with_gold": 0, "jaccard_sum": 0.0, "exact": 0, "jaccards": []},
        3: {"n": 0, "with_gold": 0, "jaccard_sum": 0.0, "exact": 0, "jaccards": []},
    }
    per_item: List[Dict] = []

    for it in success_items:
        question = (it.get("question") or "").strip()
        hop_count = it.get("hop_count")
        kg_results = it.get("kg_results") or []
        pred_set = {x.strip() for x in kg_results if x and str(x).strip()}

        if hop_count not in hop_stats:
            hop_stats[hop_count] = {"n": 0, "with_gold": 0, "jaccard_sum": 0.0, "exact": 0, "jaccards": []}
        hop_stats[hop_count]["n"] += 1

        gold_set = gold_by_hop.get(hop_count, {}).get(question)
        if gold_set is None:
            j = None  # no gold to compare
            per_item.append({
                "question": question,
                "hop_count": hop_count,
                "jaccard": None,
                "exact_match": None,
                "gold_found": False,
            })
            continue

        hop_stats[hop_count]["with_gold"] += 1
        j = jaccard(pred_set, gold_set)
        exact = j >= 1.0
        hop_stats[hop_count]["jaccard_sum"] += j
        hop_stats[hop_count]["jaccards"].append(j)
        if exact:
            hop_stats[hop_count]["exact"] += 1

        per_item.append({
            "question": question,
            "hop_count": hop_count,
            "jaccard": round(j, 4),
            "exact_match": exact,
            "gold_found": True,
        })

    # Build summary
    total_answered = len(success_items)
    total_with_gold = sum(s["with_gold"] for s in hop_stats.values())
    total_exact = sum(s["exact"] for s in hop_stats.values())
    total_jaccard_sum = sum(s["jaccard_sum"] for s in hop_stats.values())

    summary = {
        "seed": seed,
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "total_in_success": total_answered,
        "total_with_gold": total_with_gold,
        "total_exact_match": total_exact,
        "overall_pct_exact": round(100.0 * total_exact / total_with_gold, 2) if total_with_gold else None,
        "overall_mean_jaccard": round(total_jaccard_sum / total_with_gold, 4) if total_with_gold else None,
        "per_hop": {},
    }

    for hop in (1, 2, 3):
        s = hop_stats[hop]
        n = s["n"]
        wg = s["with_gold"]
        total_gold = total_per_hop.get(hop, 0)
        coverage_pct = round(100.0 * n / total_gold, 2) if total_gold else None
        summary["per_hop"][str(hop)] = {
            "total_gold_questions": total_gold,
            "answered_count": n,
            "coverage_pct": coverage_pct,
            "with_gold_count": wg,
            "exact_match_count": s["exact"],
            "pct_exact": round(100.0 * s["exact"] / wg, 2) if wg else None,
            "mean_jaccard": round(s["jaccard_sum"] / wg, 4) if wg else None,
        }
        # Optional: median (for notes)
        jaccards = s["jaccards"]
        if jaccards:
            jaccards_sorted = sorted(jaccards)
            mid = len(jaccards_sorted) // 2
            median_j = (jaccards_sorted[mid] + jaccards_sorted[mid - 1]) / 2 if len(jaccards_sorted) % 2 == 0 else jaccards_sorted[mid]
            summary["per_hop"][str(hop)]["median_jaccard"] = round(median_j, 4)
        else:
            summary["per_hop"][str(hop)]["median_jaccard"] = None

    return summary, per_item


def load_decomposition_pipeline_stats(analysis_dir: Path) -> Dict:
    """
    Load success/compile_fail/exec_fail counts and error reason breakdowns from run analysis.
    Returns dict with total, success_count, compile_fail_count, exec_fail_count,
    compile_fail_reasons, exec_fail_reasons (counts by reason key).
    """
    out = {
        "total": 0,
        "success_count": 0,
        "compile_fail_count": 0,
        "exec_fail_count": 0,
        "compile_fail_reasons": {},
        "exec_fail_reasons": {},
    }
    success_path = analysis_dir / "success.json"
    compile_path = analysis_dir / "compile_fail.json"
    exec_path = analysis_dir / "exec_fail.json"
    if success_path.exists():
        with open(success_path, encoding="utf-8") as f:
            out["success_count"] = len(json.load(f))
    if compile_path.exists():
        with open(compile_path, encoding="utf-8") as f:
            items = json.load(f)
        out["compile_fail_count"] = len(items)
        for it in items:
            r = (it.get("error_reason") or "unknown").strip()
            out["compile_fail_reasons"][r] = out["compile_fail_reasons"].get(r, 0) + 1
    if exec_path.exists():
        with open(exec_path, encoding="utf-8") as f:
            items = json.load(f)
        out["exec_fail_count"] = len(items)
        for it in items:
            r = (it.get("error_reason") or "unknown").strip()
            # Normalize to short reason: "entity_not_in_kb: 'x'" -> "entity_not_in_kb"
            if ":" in r:
                r = r.split(":", 1)[0].strip()
            out["exec_fail_reasons"][r] = out["exec_fail_reasons"].get(r, 0) + 1
    out["total"] = out["success_count"] + out["compile_fail_count"] + out["exec_fail_count"]
    return out


def generate_visualizations(
    summary: Dict,
    per_item: List[Dict],
    report_dir: Path,
    pipeline_stats: Optional[Dict] = None,
) -> None:
    """Generate plots and save to report_dir. No-op if matplotlib not available."""
    if not _HAS_MATPLOTLIB:
        return
    report_dir.mkdir(parents=True, exist_ok=True)
    per_hop = summary.get("per_hop") or {}

    # 1) Per-hop bar chart: coverage %, exact %, mean Jaccard
    hops = ["1-hop", "2-hop", "3-hop"]
    coverage = [per_hop.get(str(h), {}).get("coverage_pct") or 0 for h in (1, 2, 3)]
    pct_exact = [per_hop.get(str(h), {}).get("pct_exact") or 0 for h in (1, 2, 3)]
    mean_jaccard = [(per_hop.get(str(h), {}).get("mean_jaccard") or 0) * 100 for h in (1, 2, 3)]

    x = np.arange(len(hops))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width, coverage, width, label="Coverage %", color="steelblue")
    bars2 = ax.bar(x, pct_exact, width, label="Exact match %", color="seagreen")
    bars3 = ax.bar(x + width, mean_jaccard, width, label="Mean Jaccard (×100)", color="coral")
    ax.bar_label(bars1, fmt="%.1f")
    ax.bar_label(bars2, fmt="%.1f")
    ax.bar_label(bars3, fmt="%.1f")
    ax.set_ylabel("Percentage")
    ax.set_title("Answer accuracy by hop")
    ax.set_xticks(x)
    ax.set_xticklabels(hops)
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(report_dir / "per_hop_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2) Jaccard distribution per hop (box plot)
    fig, ax = plt.subplots(figsize=(7, 4))
    jaccards_by_hop = {1: [], 2: [], 3: []}
    for it in per_item:
        if it.get("jaccard") is not None:
            jaccards_by_hop[it["hop_count"]].append(it["jaccard"])
    # Use at least one value per hop so boxplot renders (avoid empty list)
    data = [jaccards_by_hop[h] if jaccards_by_hop[h] else [0.0] for h in (1, 2, 3)]
    bp = ax.boxplot(data, tick_labels=hops, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax.set_ylabel("Jaccard similarity")
    ax.set_title("Jaccard distribution by hop")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(report_dir / "jaccard_by_hop.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3) Overall summary: single pie or bar for exact vs non-exact
    total_exact = summary.get("total_exact_match") or 0
    total_with_gold = summary.get("total_with_gold") or 1
    non_exact = total_with_gold - total_exact
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Exact match", "Non-exact"], [total_exact, non_exact], color=["seagreen", "coral"])
    ax.bar_label(bars, fmt="%d")
    ax.set_ylabel("Count")
    ax.set_title(f"Overall answer match (n={total_with_gold})")
    fig.tight_layout()
    fig.savefig(report_dir / "overall_match.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4) Decomposition pipeline: success vs compile fail vs exec fail
    if pipeline_stats and pipeline_stats.get("total", 0) > 0:
        total = pipeline_stats["total"]
        success_count = pipeline_stats.get("success_count") or 0
        compile_fail_count = pipeline_stats.get("compile_fail_count") or 0
        exec_fail_count = pipeline_stats.get("exec_fail_count") or 0
        labels = ["Success", "Compile fail", "Exec fail"]
        counts = [success_count, compile_fail_count, exec_fail_count]
        colors = ["seagreen", "coral", "indianred"]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, counts, color=colors)
        ax.bar_label(bars, fmt="%d")
        ax.set_ylabel("Count")
        ax.set_title(f"Question decomposition pipeline (n={total})")
        fig.tight_layout()
        fig.savefig(report_dir / "decomposition_pipeline.png", dpi=150, bbox_inches="tight")
        plt.close()

        # 5) Compile fail reasons (template / relation errors)
        comp_reasons = pipeline_stats.get("compile_fail_reasons") or {}
        if comp_reasons:
            reason_labels = list(comp_reasons.keys())
            reason_counts = list(comp_reasons.values())
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(reason_labels, reason_counts, color="coral", alpha=0.8)
            ax.bar_label(bars, fmt="%d")
            ax.set_xlabel("Count")
            ax.set_title("Compile / template error reasons")
            fig.tight_layout()
            fig.savefig(report_dir / "compile_fail_reasons.png", dpi=150, bbox_inches="tight")
            plt.close()

        # 6) Exec fail reasons (entity not in KB, bad reference, etc.)
        exec_reasons = pipeline_stats.get("exec_fail_reasons") or {}
        if exec_reasons:
            reason_labels = list(exec_reasons.keys())
            reason_counts = list(exec_reasons.values())
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(reason_labels, reason_counts, color="indianred", alpha=0.8)
            ax.bar_label(bars, fmt="%d")
            ax.set_xlabel("Count")
            ax.set_title("Execution error reasons")
            fig.tight_layout()
            fig.savefig(report_dir / "exec_fail_reasons.png", dpi=150, bbox_inches="tight")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare success.json kg_results to gold answers; Jaccard + per-hop stats.")
    parser.add_argument("run_dir", type=Path, help="Decomposer run directory (e.g. runs/decomposer/20260123_072902)")
    parser.add_argument("--data-dir", type=Path, default=Path("Data"), help="Data directory with refined_*hop.txt and answers_*hop.txt")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility note")
    parser.add_argument("--no-write", action="store_true", help="Do not write metrics or notes to disk")
    parser.add_argument("--details", action="store_true", help="Write per-question details to analysis/answer_details.json")
    parser.add_argument("--report", action="store_true", help="Generate visualizations in reports/<run_id>/")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"), help="Base directory for report output (default: reports)")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir}")

    summary, per_item = run_analysis(run_dir, data_dir, args.seed)

    # Print summary
    print("\n" + "=" * 50)
    print("   ANSWER ACCURACY (success.json vs gold)")
    print("=" * 50)
    print(f"Run dir:       {run_dir}")
    print(f"Data dir:     {data_dir}")
    print(f"Total in success.json: {summary['total_in_success']}")
    print(f"With gold:    {summary['total_with_gold']}")
    print(f"Overall exact match: {summary['total_exact_match']} ({summary['overall_pct_exact']}%)")
    print(f"Overall mean Jaccard: {summary['overall_mean_jaccard']}")
    print("\nPer hop:")
    for hop in ("1", "2", "3"):
        p = summary["per_hop"][hop]
        total_g = p.get("total_gold_questions")
        cov = f", coverage={p.get('coverage_pct')}% of {total_g}" if total_g else ""
        print(f"  {hop}-hop: answered={p['answered_count']}{cov}, with_gold={p['with_gold_count']}, "
              f"exact={p['exact_match_count']} ({p['pct_exact']}%), mean_jaccard={p['mean_jaccard']}, median_jaccard={p.get('median_jaccard')}")
    print("=" * 50)

    if not args.no_write:
        analysis_dir = run_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = analysis_dir / "answer_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "config": {"seed": args.seed}}, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {metrics_path}")

        notes_path = analysis_dir / "answer_accuracy_notes.md"
        notes = [
            "# Answer accuracy (success.json vs gold)",
            "",
            f"- Run: `{run_dir}`",
            f"- Gold: `{data_dir}` (refined_*hop + answers_*hop)",
            f"- Metric: Jaccard similarity (set overlap); exact = Jaccard 1.0",
            "",
            "## Summary",
            f"- Total in success: {summary['total_in_success']}, with gold: {summary['total_with_gold']}",
            f"- Overall % exact: {summary['overall_pct_exact']}%, mean Jaccard: {summary['overall_mean_jaccard']}",
            "",
            "## Per hop",
        ]
        for hop in ("1", "2", "3"):
            p = summary["per_hop"][hop]
            cov = f", coverage {p.get('coverage_pct')}% of {p.get('total_gold_questions')}" if p.get("total_gold_questions") else ""
            notes.append(f"- **{hop}-hop**: answered {p['answered_count']}{cov}, with_gold {p['with_gold_count']}, "
                        f"exact {p['exact_match_count']} ({p['pct_exact']}%), mean Jaccard {p['mean_jaccard']}, median {p.get('median_jaccard')}")
        notes.append("")
        notes.append(f"(Seed: {args.seed})")
        notes_path.write_text("\n".join(notes), encoding="utf-8")
        print(f"Wrote {notes_path}")

        if args.details:
            details_path = analysis_dir / "answer_details.json"
            with open(details_path, "w", encoding="utf-8") as f:
                json.dump(per_item, f, indent=2, ensure_ascii=False)
            print(f"Wrote {details_path} ({len(per_item)} items)")

    if args.report:
        report_dir = args.reports_dir.resolve() / run_dir.name
        if _HAS_MATPLOTLIB:
            analysis_dir = run_dir / "analysis"
            pipeline_stats = load_decomposition_pipeline_stats(analysis_dir) if analysis_dir.exists() else None
            generate_visualizations(summary, per_item, report_dir, pipeline_stats=pipeline_stats)
            print(f"Wrote visualizations to {report_dir}")
        else:
            print("Skipping visualizations: matplotlib not available. Install matplotlib and numpy for --report.")


if __name__ == "__main__":
    main()
