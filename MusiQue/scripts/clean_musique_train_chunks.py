#!/usr/bin/env python3
"""
Clean MusiQue chunk JSONL files: keep id, hop_count (derived), question,
question_decomposition, and per-file index.

Run with the project venv::

    source .venv/bin/activate
    python MusiQue/scripts/clean_musique_train_chunks.py --input-glob 'MusiQue/Data/musique_ans_v1.0_train_*.jsonl' --seed 42
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from musique_ids import coarse_hop_from_id  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        help="Explicit list of input JSONL files (ordered)",
    )
    p.add_argument(
        "--input-glob",
        type=str,
        default=None,
        help="Glob pattern for inputs (e.g. MusiQue/Data/musique_ans_v1.0_train_*.jsonl)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as first input's parent)",
    )
    p.add_argument(
        "--out-suffix",
        type=str,
        default="_clean",
        help="Suffix before .jsonl for output basenames (default: _clean)",
    )
    p.add_argument("--seed", type=int, default=42, help="Logged for reproducibility")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    p.add_argument(
        "--allow-clean-inputs",
        action="store_true",
        help="When using --input-glob, include *_clean.jsonl matches (default: skip stems ending with _clean)",
    )
    return p.parse_args()


def _resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.inputs:
        return [Path(p).resolve() for p in args.inputs]
    if args.input_glob:
        paths = sorted(glob.glob(args.input_glob, recursive=False))
        if not paths:
            raise SystemExit(f"No files matched glob: {args.input_glob}")
        resolved = [Path(p).resolve() for p in paths]
        if not args.allow_clean_inputs:
            resolved = [p for p in resolved if not p.stem.endswith("_clean")]
        if not resolved:
            raise SystemExit(
                "After excluding *_clean inputs, no files left. "
                "Tighten --input-glob (e.g. musique_ans_v1.0_train_[0-3].jsonl) or use --allow-clean-inputs."
            )
        return resolved
    raise SystemExit("Provide --inputs and/or --input-glob")


def main() -> None:
    args = _parse_args()
    inputs = _resolve_inputs(args)
    out_dir = args.out_dir if args.out_dir is not None else inputs[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for inp in inputs:
        stem = inp.stem
        out_name = f"{stem}{args.out_suffix}.jsonl"
        out_path = out_dir / out_name
        if out_path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite (use --overwrite): {out_path}")

        idx = 0
        with inp.open(encoding="utf-8") as rf, out_path.open("w", encoding="utf-8") as wf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rid = obj.get("id", "")
                cleaned = {
                    "id": rid,
                    "hop_count": coarse_hop_from_id(rid),
                    "question": obj.get("question"),
                    "question_decomposition": obj.get("question_decomposition"),
                    "index": idx,
                }
                wf.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                idx += 1

        print(f"{inp.name} -> {out_path.name} ({idx} rows, seed={args.seed})")


if __name__ == "__main__":
    main()
