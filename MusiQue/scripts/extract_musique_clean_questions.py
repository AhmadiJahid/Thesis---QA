#!/usr/bin/env python3
"""
Extract id, question, index from MusiQue clean JSONL chunks and split by id stratum.

Writes one file per (chunk, stratum), e.g. musique_ans_v1.0_train_0_questions_2_hop.jsonl.

Run with the project venv::

    source .venv/bin/activate
    python MusiQue/scripts/extract_musique_clean_questions.py --seed 42
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from musique_ids import stratum_from_id, stratum_to_questions_slug  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        help="Explicit clean JSONL paths (ordered)",
    )
    p.add_argument(
        "--input-glob",
        type=str,
        default=None,
        help="Glob for clean chunks (default: clean_data/musique_ans_v1.0_train_*_clean.jsonl)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "MusiQue" / "Data" / "chunks_only_question",
        help="Directory for stratum question JSONLs",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "musique_question_extract",
        help="Directory for metrics.json and notes.md",
    )
    p.add_argument("--seed", type=int, default=42, help="Logged in metrics.json")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    p.add_argument(
        "--also-write-merged",
        action="store_true",
        help="Also write one ..._questions_all.jsonl per input (debug)",
    )
    return p.parse_args()


def _resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.inputs:
        return [Path(p).resolve() for p in args.inputs]
    g = args.input_glob
    if g is None:
        g = str(REPO_ROOT / "MusiQue" / "Data" / "clean_data" / "musique_ans_v1.0_train_*_clean.jsonl")
    paths = sorted(glob.glob(g, recursive=False))
    if not paths:
        raise SystemExit(f"No files matched glob: {g}")
    resolved = [Path(p).resolve() for p in paths]
    resolved = [p for p in resolved if p.stem.endswith("_clean")]
    if not resolved:
        raise SystemExit("No inputs with stem ending in _clean (expected clean chunk files).")
    return resolved


def _base_stem_clean(stem: str) -> str:
    if stem.endswith("_clean"):
        return stem[: -len("_clean")]
    return stem


def main() -> None:
    args = _parse_args()
    inputs = _resolve_inputs(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict = {
        "script": "extract_musique_clean_questions.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "out_dir": str(args.out_dir.resolve()),
        "inputs": [str(p) for p in inputs],
        "per_file": {},
        "outputs": [],
    }

    for inp in inputs:
        base = _base_stem_clean(inp.stem)
        by_stratum: dict[str, list[dict]] = defaultdict(list)
        unknown_strata = 0

        with inp.open(encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rid = obj.get("id", "")
                st = stratum_from_id(rid)
                slug = stratum_to_questions_slug(st)
                if st == "unknown":
                    unknown_strata += 1
                row = {
                    "id": rid,
                    "question": obj.get("question"),
                    "index": obj.get("index"),
                }
                by_stratum[slug].append(row)

        per_stratum_counts = {k: len(v) for k, v in sorted(by_stratum.items())}
        metrics["per_file"][inp.name] = {
            "per_stratum_counts": per_stratum_counts,
            "total_rows": sum(per_stratum_counts.values()),
            "unknown_stratum_rows": unknown_strata,
        }

        for slug, rows in sorted(by_stratum.items()):
            out_name = f"{base}_questions_{slug}.jsonl"
            out_path = args.out_dir / out_name
            if out_path.exists() and not args.overwrite:
                raise SystemExit(f"Refusing to overwrite (use --overwrite): {out_path}")
            rows.sort(key=lambda r: (r.get("index") is None, r.get("index", 0)))
            with out_path.open("w", encoding="utf-8") as wf:
                for row in rows:
                    wf.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics["outputs"].append(str(out_path.resolve()))
            print(f"{inp.name} -> {out_name} ({len(rows)} rows)")

        if args.also_write_merged:
            merged_name = f"{base}_questions_all.jsonl"
            merged_path = args.out_dir / merged_name
            if merged_path.exists() and not args.overwrite:
                raise SystemExit(f"Refusing to overwrite (use --overwrite): {merged_path}")
            all_rows: list[dict] = []
            for slug in sorted(by_stratum.keys()):
                all_rows.extend(by_stratum[slug])
            all_rows.sort(key=lambda r: (r.get("index") is None, r.get("index", 0)))
            with merged_path.open("w", encoding="utf-8") as wf:
                for row in all_rows:
                    wf.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics["outputs"].append(str(merged_path.resolve()))
            print(f"{inp.name} -> {merged_name} ({len(all_rows)} rows, merged)")

    metrics_path = args.run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as mf:
        json.dump(metrics, mf, indent=2, ensure_ascii=False)
        mf.write("\n")

    notes = args.run_dir / "notes.md"
    notes.write_text(
        "# MusiQue question extract (stratified)\n\n"
        f"- Seed: {args.seed}\n"
        f"- Output directory: `{args.out_dir}`\n"
        f"- Metrics: `{metrics_path}`\n",
        encoding="utf-8",
    )
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
