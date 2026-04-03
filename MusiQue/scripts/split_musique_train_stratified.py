#!/usr/bin/env python3
"""
Stratified 4-way split of MusiQue train JSONL into quarter chunks.

Uses id-prefix strata (``2hop``, ``3hop1``, …) because ``hop_count`` is null in source.

Run with the project venv activated, e.g.::

    source .venv/bin/activate
    python MusiQue/scripts/split_musique_train_stratified.py --seed 42

Or: ``.venv/bin/python MusiQue/scripts/split_musique_train_stratified.py ...``
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from musique_ids import stratum_from_id  # noqa: E402


def _default_input() -> Path:
    return REPO_ROOT / "MusiQue" / "Data" / "musique_ans_v1.0_train.jsonl"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=_default_input(), help="Source train JSONL")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for chunk files (default: same directory as --input)",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="musique_ans_v1.0_train",
        help="Output basename prefix before _{0..3}.jsonl",
    )
    p.add_argument("--seed", type=int, required=True, help="Random seed for StratifiedKFold shuffle")
    p.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="Directory for split_manifest.json and notes.md (default: runs/musique_train_split under repo root)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    input_path: Path = args.input
    out_dir: Path = args.out_dir if args.out_dir is not None else input_path.parent
    manifest_dir: Path = (
        args.manifest_dir if args.manifest_dir is not None else REPO_ROOT / "runs" / "musique_train_split"
    )

    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    lines: list[str] = []
    strata: list[str] = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            lines.append(line)
            strata.append(stratum_from_id(obj.get("id", "")))

    n = len(lines)
    if n == 0:
        raise SystemExit("No records in input")

    y = np.array(strata, dtype=object)
    X = np.zeros(n, dtype=np.int8)

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=args.seed)
    assignments = np.full(n, -1, dtype=np.int8)
    for fold_idx, (_, test_idx) in enumerate(skf.split(X, y)):
        assignments[test_idx] = fold_idx

    if not np.all(assignments >= 0):
        raise SystemExit("Internal error: not all rows assigned to a fold")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    chunk_stratum_hist: dict[str, dict[str, int]] = {}
    chunk_line_counts: dict[str, int] = {}

    for chunk_id in range(4):
        indices = np.where(assignments == chunk_id)[0]
        indices.sort()
        out_name = f"{args.prefix}_{chunk_id}.jsonl"
        out_path = out_dir / out_name
        hist: Counter[str] = Counter()
        with out_path.open("w", encoding="utf-8") as wf:
            for i in indices:
                wf.write(lines[int(i)] + "\n")
                hist[strata[int(i)]] += 1
        key = str(chunk_id)
        chunk_stratum_hist[key] = dict(sorted(hist.items()))
        chunk_line_counts[key] = int(len(indices))

    manifest = {
        "script": "split_musique_train_stratified.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "input": str(input_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "prefix": args.prefix,
        "total_lines": n,
        "chunk_line_counts": chunk_line_counts,
        "stratum_histograms_per_chunk": chunk_stratum_hist,
        "global_stratum_histogram": dict(sorted(Counter(strata).items())),
    }
    manifest_path = manifest_dir / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)
        mf.write("\n")

    notes_path = manifest_dir / "notes.md"
    notes_path.write_text(
        "# MusiQue train stratified split\n\n"
        f"- Seed: {args.seed}\n"
        f"- Input: `{input_path}`\n"
        f"- Chunks written under: `{out_dir}`\n"
        f"- Total lines: {n}\n"
        f"- Per-chunk counts: {chunk_line_counts}\n"
        f"- Manifest: `{manifest_path}`\n",
        encoding="utf-8",
    )

    print(f"Wrote 4 chunks under {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Notes: {notes_path}")


if __name__ == "__main__":
    main()
