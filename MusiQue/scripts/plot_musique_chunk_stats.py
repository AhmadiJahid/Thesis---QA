#!/usr/bin/env python3
"""
Plot hop / stratum counts per MusiQue chunk JSONL (raw split or cleaned).

Writes figures and plot_stats.json under runs/ by default.

Run with the project venv::

    source .venv/bin/activate
    python MusiQue/scripts/plot_musique_chunk_stats.py --input-glob 'MusiQue/Data/musique_ans_v1.0_train_*.jsonl' --seed 42
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from musique_ids import coarse_hop_from_record, stratum_from_id  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="*", type=Path, help="Ordered chunk JSONL paths")
    p.add_argument(
        "--input-glob",
        type=str,
        default=None,
        help="Glob for chunk files (sorted lexicographically)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "musique_train_plots",
        help="Directory for PNG/PDF and plot_stats.json",
    )
    p.add_argument("--seed", type=int, default=42, help="Logged in plot_stats.json")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--format", type=str, default="png", choices=("png", "pdf", "svg"))
    p.add_argument(
        "--no-stratum-figure",
        action="store_true",
        help="Skip the id-prefix stratum heatmap",
    )
    p.add_argument(
        "--allow-clean-inputs",
        action="store_true",
        help="When using --input-glob, include *_clean.jsonl (default: skip stems ending with _clean)",
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
                "Use a tighter glob (e.g. musique_ans_v1.0_train_[0-3].jsonl) or --allow-clean-inputs."
            )
        return resolved
    raise SystemExit("Provide --inputs and/or --input-glob")


def _chunk_label(path: Path) -> str:
    m = re.search(r"_(\d+)(?:_clean)?\.jsonl$", path.name)
    if m:
        return m.group(1)
    return path.stem


def main() -> None:
    args = _parse_args()
    inputs = _resolve_inputs(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    chunk_keys = [_chunk_label(p) for p in inputs]
    coarse_per_chunk: dict[str, Counter[int]] = {k: Counter() for k in chunk_keys}
    stratum_per_chunk: dict[str, Counter[str]] = {k: Counter() for k in chunk_keys}

    for path, ck in zip(inputs, chunk_keys):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ch = coarse_hop_from_record(obj)
                if ch >= 0:
                    coarse_per_chunk[ck][ch] += 1
                stratum_per_chunk[ck][stratum_from_id(obj.get("id", ""))] += 1

    hops_sorted = sorted(
        {h for c in coarse_per_chunk.values() for h in c.keys()},
    )

    # --- Figure 1: grouped bar coarse hop ---
    x = range(len(chunk_keys))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, h in enumerate(hops_sorted):
        offsets = [xi + (i - (len(hops_sorted) - 1) / 2) * width for xi in x]
        heights = [coarse_per_chunk[ck][h] for ck in chunk_keys]
        ax.bar(offsets, heights, width=width, label=str(h))
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"chunk {k}" for k in chunk_keys])
    ax.set_ylabel("count")
    ax.set_title("Coarse hop counts per chunk")
    ax.legend(title="hop")
    fig.tight_layout()
    p1 = args.out_dir / f"hop_counts_per_chunk.{args.format}"
    fig.savefig(p1, dpi=args.dpi)
    plt.close(fig)

    # --- Figure 2: heatmap chunk x stratum ---
    p2 = None
    if not args.no_stratum_figure:
        strata_sorted = sorted({s for c in stratum_per_chunk.values() for s in c.keys()})
        mat = [[stratum_per_chunk[ck][s] for ck in chunk_keys] for s in strata_sorted]
        fig2, ax2 = plt.subplots(figsize=(max(8, len(chunk_keys) * 1.2), max(4, len(strata_sorted) * 0.35)))
        im = ax2.imshow(mat, aspect="auto", cmap="Blues")
        ax2.set_xticks(range(len(chunk_keys)))
        ax2.set_xticklabels([f"c{k}" for k in chunk_keys])
        ax2.set_yticks(range(len(strata_sorted)))
        ax2.set_yticklabels(strata_sorted)
        ax2.set_xlabel("chunk")
        ax2.set_ylabel("id stratum")
        ax2.set_title("Stratum counts per chunk (heatmap)")
        fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        fig2.tight_layout()
        p2 = args.out_dir / f"stratum_heatmap_per_chunk.{args.format}"
        fig2.savefig(p2, dpi=args.dpi)
        plt.close(fig2)

    stats = {
        "script": "plot_musique_chunk_stats.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "input_files": [str(p) for p in inputs],
        "chunk_keys": chunk_keys,
        "coarse_hop_counts_per_chunk": {k: {str(h): coarse_per_chunk[k][h] for h in hops_sorted} for k in chunk_keys},
        "stratum_counts_per_chunk": {k: dict(sorted(stratum_per_chunk[k].items())) for k in chunk_keys},
        "figures": {
            "hop_counts_per_chunk": str(p1),
            **({"stratum_heatmap_per_chunk": str(p2)} if p2 else {}),
        },
    }
    out_json = args.out_dir / "plot_stats.json"
    with out_json.open("w", encoding="utf-8") as jf:
        json.dump(stats, jf, indent=2, ensure_ascii=False)
        jf.write("\n")

    notes = args.out_dir / "notes.md"
    notes.write_text(
        "# MusiQue chunk stats plots\n\n"
        f"- Seed: {args.seed}\n"
        f"- Outputs: `{p1}`" + (f", `{p2}`" if p2 else "") + "\n"
        f"- Stats: `{out_json}`\n",
        encoding="utf-8",
    )
    print(f"Wrote {p1}")
    if p2:
        print(f"Wrote {p2}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
