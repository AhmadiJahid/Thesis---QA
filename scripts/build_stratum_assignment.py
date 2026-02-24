#!/usr/bin/env python3
"""
Assign questions in few_shot_decompositions.json to strata based on question patterns.
Outputs Pool/stratum_assignment.json and Pool/stratum_report.md for inspection.

Usage:
    python scripts/build_stratum_assignment.py
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# Stratum rules: (regex_pattern, stratum_name). First match wins.
# Order: more specific patterns first to avoid genre/types matching "directed" etc.
STRATUM_RULES = [
    # Release / years / dates (check before "years" in genre)
    (re.compile(r"\b(released?|release\s+date|release\s+years?|released\s+in\s+which\s+years?)\b", re.I), "release"),
    (re.compile(r"\bwhen\s+(were|did)\b", re.I), "release"),
    (re.compile(r"\b(in\s+which\s+years?|which\s+years?)\b", re.I), "release"),
    # Language
    (re.compile(r"\b(language|languages)\b", re.I), "language"),
    # Genre / types
    (re.compile(r"\b(genre|genres|type|types)\b", re.I), "genre"),
    # Writer / screenwriter / co-writer (before director - "co-directed" vs "directed")
    (re.compile(r"\b(co-wrote|co-writer|co-writers|screenwriter|screenwriters|scriptwriter)\b", re.I), "writer"),
    (re.compile(r"\b(wrote|written|writer|writers)\b", re.I), "writer"),
    # Director / co-director
    (re.compile(r"\b(co-directed|co-director|co-directors)\b", re.I), "director"),
    (re.compile(r"\b(directed|director|directors)\b", re.I), "director"),
    # Actor / star / co-star
    (re.compile(r"\b(co-starred|co-star|acted\s+together)\b", re.I), "actor"),
    (re.compile(r"\b(actor|actors|acted|starred|star|appeared|appear)\b", re.I), "actor"),
]

STRATA_ORDER = ["director", "actor", "writer", "genre", "release", "language"]


def assign_stratum(question: str) -> str:
    """Return stratum name for question. First matching rule wins."""
    for pattern, stratum in STRATUM_RULES:
        if pattern.search(question):
            return stratum
    return "other"


def main():
    repo_root = Path(__file__).resolve().parents[1]
    pool_path = repo_root / "Pool" / "few_shot_decompositions.json"
    if not pool_path.exists():
        print(f"Error: {pool_path} not found")
        return

    with open(pool_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assignment = {"2hop": defaultdict(list), "3hop": defaultdict(list)}
    for hop_key in ["2hop", "3hop"]:
        for item in data.get(hop_key, []):
            q = item["question"]
            stratum = assign_stratum(q)
            assignment[hop_key][stratum].append({"question": q, "decomposition": item["decomposition"]})

    # Convert to plain dict, ordered by strata
    order = STRATA_ORDER + ["other"]
    result = {}
    for hop_key in ["2hop", "3hop"]:
        result[hop_key] = {}
        for s in order:
            result[hop_key][s] = assignment[hop_key].get(s, [])
        # Add any stratum not in order (e.g. from future rules)
        for s in assignment[hop_key]:
            if s not in result[hop_key]:
                result[hop_key][s] = assignment[hop_key][s]
    assignment = result

    out_json = repo_root / "Pool" / "stratum_assignment.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(assignment, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_json}")

    # Human-readable report
    lines = [
        "# Stratum Assignment Report",
        "",
        "Questions from `few_shot_decompositions.json` assigned to strata by keyword rules.",
        "",
    ]
    for hop_key in ["2hop", "3hop"]:
        lines.append(f"## {hop_key}")
        lines.append("")
        for stratum, items in assignment[hop_key].items():
            lines.append(f"### {stratum} ({len(items)} questions)")
            lines.append("")
            for i, it in enumerate(items, 1):
                lines.append(f"{i}. {it['question']}")
            lines.append("")
        lines.append("---")
        lines.append("")

    out_md = repo_root / "Pool" / "stratum_report.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")

    # Summary table
    print("\nSummary:")
    for hop_key in ["2hop", "3hop"]:
        print(f"  {hop_key}: " + ", ".join(f"{s}={len(assignment[hop_key][s])}" for s in assignment[hop_key]))


if __name__ == "__main__":
    main()
