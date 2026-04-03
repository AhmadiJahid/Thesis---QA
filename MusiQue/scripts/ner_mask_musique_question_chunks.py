#!/usr/bin/env python3
"""
Apply NER masking to MusiQue stratum question JSONLs (id, question, index).

Default: one JSONL per input under <model_slug>/ with columns side-by-side::

    id, index, question, question_masked_typed, question_masked_uniform

Use --split-typed-uniform-dirs for legacy <model_slug>/typed/ and uniform/ subdirs.

Default models: dslim/bert-large-NER and Jean-Baptiste/roberta-large-ner-english.

Loads each JSONL into a Hugging Face ``datasets.Dataset`` and passes it to the
transformers pipeline via ``KeyDataset``, enabling proper ``DataLoader``-based GPU
batching (``--batch-size``, default 16).  Entity spans are reused for both typed
and uniform masks (one forward pass per row, not two).

Run with the project venv::

    source .venv/bin/activate
    python MusiQue/scripts/ner_mask_musique_question_chunks.py --seed 42

Train chunk 0 only::

    python MusiQue/scripts/ner_mask_musique_question_chunks.py --seed 42 \\
      --input-glob 'MusiQue/Data/chunks_only_question/musique_ans_v1.0_train_0_questions_*.jsonl'
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

_DEFAULT_MODELS = (
    "dslim/bert-large-NER",
    "Jean-Baptiste/roberta-large-ner-english",
)

# Conservative DATE patterns (English); heuristic only.
_DATE_RES = (
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ),
)
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _sanitize_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s.split("/")[-1]).strip("_")


def _load_tokenizer_for_ner(model_name: str) -> Any:
    """
    Some Hub checkpoints (e.g. certain DeBERTa NER models) ship without a usable
    fast tokenizer file; fall back to the Python (slow) tokenizer.
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print(
            "  Note: using slow tokenizer (use_fast=False); "
            "fast tokenizer file missing or invalid for this checkpoint."
        )
        return tok


def _typed_placeholders() -> dict[str, str]:
    return {
        "PER": "[PERSON]",
        "PERSON": "[PERSON]",
        "ORG": "[ORG]",
        "LOC": "[PLACE]",
        "GPE": "[PLACE]",
        "MISC": "[WORK]",
    }

_UNIFORM_TOKEN = "[MASK]"
_FALLBACK_TYPED = "[ENTITY]"


def _normalize_entity_label(label: object) -> str:
    s = str(label).strip()
    if len(s) >= 2 and s[1] == "-":
        s = s[2:]
    return s.upper()


_MAX_MERGE_GAP = 3  # merge adjacent same-type NER spans separated by <= 3 chars


def _merge_ner_spans(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """
    Sort, drop overlaps, and merge adjacent same-type spans whose gap is
    at most ``_MAX_MERGE_GAP`` characters (handles "Kyeon Mi-ri" split into
    two PER spans with a short gap).
    """
    spans = sorted(spans, key=lambda x: x[0])
    merged: list[tuple[int, int, str]] = []
    for s, e, ph in spans:
        if merged:
            ps, pe, pph = merged[-1]
            if s < pe:
                continue
            if pph == ph and (s - pe) <= _MAX_MERGE_GAP:
                merged[-1] = (ps, e, ph)
                continue
        merged.append((s, e, ph))
    return merged


def _overlaps_blocked(s: int, e: int, blocked: list[tuple[int, int]]) -> bool:
    for bs, be in blocked:
        if s < be and bs < e:
            return True
    return False


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    out: list[tuple[int, int]] = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def _regex_spans(text: str, ner_blocked: list[tuple[int, int]], typed: bool) -> list[tuple[int, int, str, int]]:
    """DATE (priority 1) in gaps of NER; NUM (priority 2) in gaps of NER+DATE."""
    blocked = sorted(ner_blocked)
    date_ivs: list[tuple[int, int]] = []
    for rx in _DATE_RES:
        for m in rx.finditer(text):
            if not _overlaps_blocked(m.start(), m.end(), blocked):
                date_ivs.append((m.start(), m.end()))
    date_merged = _merge_intervals(date_ivs)
    blocked2 = blocked + list(date_merged)
    out: list[tuple[int, int, str, int]] = []
    for s, e in date_merged:
        ph = "[DATE]" if typed else _UNIFORM_TOKEN
        out.append((s, e, ph, 1))
    num_ivs: list[tuple[int, int]] = []
    for m in _NUM_RE.finditer(text):
        if not _overlaps_blocked(m.start(), m.end(), sorted(blocked2)):
            num_ivs.append((m.start(), m.end()))
    num_merged = _merge_intervals(num_ivs)
    for s, e in num_merged:
        ph = "[NUM]" if typed else _UNIFORM_TOKEN
        out.append((s, e, ph, 2))
    return out


def _merge_prioritized(spans_pri: list[tuple[int, int, str, int]]) -> list[tuple[int, int, str]]:
    """Earlier span wins on overlap; tie-break by priority then longer span."""
    spans_pri = sorted(
        spans_pri,
        key=lambda x: (x[0], x[3], -(x[1] - x[0])),
    )
    merged: list[tuple[int, int, str]] = []
    last_end = -1
    for s, e, ph, _ in spans_pri:
        if s < last_end:
            continue
        merged.append((s, e, ph))
        last_end = e
    return merged


def _apply_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    if not spans:
        return text
    spans = sorted(spans, key=lambda x: x[0])
    parts: list[str] = []
    last = 0
    for s, e, ph in spans:
        parts.append(text[last:s])
        parts.append(ph)
        last = e
    parts.append(text[last:])
    return "".join(parts)


def _ner_placeholder(label: object, typed_mode: bool) -> str:
    if not typed_mode:
        return _UNIFORM_TOKEN
    key = _normalize_entity_label(label)
    typed = _typed_placeholders()
    return typed.get(key, _FALLBACK_TYPED)


def _entities_to_spans(entities: list[dict[str, Any]] | None, typed_mode: bool) -> list[tuple[int, int, str]]:
    """Build merged spans from one pipeline entity list (grouped_entities)."""
    if not entities:
        return []
    spans: list[tuple[int, int, str]] = []
    for ent in entities:
        start = ent.get("start")
        end = ent.get("end")
        if start is None or end is None:
            continue
        label = ent.get("entity_group") or ent.get("entity") or ent.get("label")
        ph = _ner_placeholder(label, typed_mode)
        spans.append((int(start), int(end), ph))
    return _merge_ner_spans(spans)


def _mask_from_entities(
    entities: list[dict[str, Any]] | None,
    text: str,
    *,
    typed_mode: bool,
    use_regex: bool,
) -> str:
    """Apply NER spans + optional regex; one NER prediction set per row (reuse for typed + uniform)."""
    if not text:
        return ""
    ner_spans = _entities_to_spans(entities, typed_mode)
    blocked = [(s, e) for s, e, _ in ner_spans]
    spans_pri: list[tuple[int, int, str, int]] = [(s, e, ph, 0) for s, e, ph in ner_spans]
    if use_regex:
        for rs, re_, ph, pri in _regex_spans(text, blocked, typed_mode):
            spans_pri.append((rs, re_, ph, pri))
    merged = _merge_prioritized(spans_pri)
    return _apply_spans(text, merged)


def _predict_entities_dataset(
    ner_pipe: Any,
    ds: Dataset,
    batch_size: int,
) -> list[list[dict[str, Any]]]:
    """
    Run token-classification pipeline over a ``datasets.Dataset`` using the
    transformers ``KeyDataset`` wrapper.  This feeds a real ``DataLoader`` to
    the pipeline, enabling proper GPU batching (no "sequential on GPU" warning).
    Returns one entity list per row (same order as input).
    """
    from transformers.pipelines.pt_utils import KeyDataset

    out: list[list[dict[str, Any]]] = []
    for item in ner_pipe(KeyDataset(ds, "question"), batch_size=batch_size):
        if item is None:
            out.append([])
        elif isinstance(item, list):
            out.append(item)
        elif isinstance(item, dict):
            out.append([item])
        else:
            out.append([])
    if len(out) != len(ds):
        raise RuntimeError(f"NER output length mismatch: got {len(out)}, expected {len(ds)}")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="*", type=Path, help="Question JSONL files (ordered)")
    p.add_argument(
        "--input-glob",
        type=str,
        default=None,
        help="Glob under chunks_only_question (excludes *_questions_all.jsonl by default)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "MusiQue" / "Data" / "chunks_only_question_masked",
        help="Root directory; each model writes under <slug>/ (combined JSONL by default)",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "musique_question_ner_mask",
        help="metrics.json + notes.md",
    )
    p.add_argument(
        "--ner-model",
        action="append",
        dest="ner_models",
        default=None,
        help="HF model id (repeatable). If omitted, defaults to bert-large-NER + roberta-large-ner-english",
    )
    p.add_argument("--device", type=int, default=-1, help="Transformers device id (-1 = CPU)")
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Sequences per forward pass for GPU NER (pipeline batching)",
    )
    p.add_argument("--seed", type=int, default=42, help="Logged in metrics.json")
    p.add_argument(
        "--no-regex-num-date",
        action="store_true",
        help="Disable heuristic [NUM]/[DATE] regex pass in gaps after NER",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output JSONLs",
    )
    p.add_argument(
        "--split-typed-uniform-dirs",
        action="store_true",
        help="Write <slug>/typed/ and <slug>/uniform/ (two files per input) instead of one combined file",
    )
    return p.parse_args()


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("question")
            if not isinstance(q, str):
                q = "" if q is None else str(q)
            obj = dict(obj)
            obj["question"] = q
            rows.append(obj)
    return rows


def _resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.inputs:
        return [Path(p).resolve() for p in args.inputs]
    g = args.input_glob
    if g is None:
        g = str(
            REPO_ROOT
            / "MusiQue"
            / "Data"
            / "chunks_only_question"
            / "musique_ans_v1.0_train_*_questions_*.jsonl"
        )
    paths = sorted(glob.glob(g, recursive=False))
    paths = [p for p in paths if not Path(p).stem.endswith("_questions_all")]
    if not paths:
        raise SystemExit(f"No inputs matched (glob: {g})")
    return [Path(p).resolve() for p in paths]


def main() -> None:
    args = _parse_args()
    inputs = _resolve_inputs(args)
    models = args.ner_models if args.ner_models else list(_DEFAULT_MODELS)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import pipeline

    use_regex = not args.no_regex_num_date

    def _build_ner_pipe(name: str) -> Any:
        tokenizer = _load_tokenizer_for_ner(name)
        return pipeline(
            "token-classification",
            model=name,
            tokenizer=tokenizer,
            grouped_entities=True,
            device=args.device,
        )
    per_model_stats: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []

    split_dirs = args.split_typed_uniform_dirs

    for model_name in models:
        slug = _sanitize_slug(model_name)
        model_root = args.out_dir / slug
        typed_root = model_root / "typed"
        uniform_root = model_root / "uniform"
        if split_dirs:
            typed_root.mkdir(parents=True, exist_ok=True)
            uniform_root.mkdir(parents=True, exist_ok=True)
        else:
            model_root.mkdir(parents=True, exist_ok=True)

        print(
            f"Loading NER pipeline: {model_name} "
            f"(batch_size={args.batch_size}, KeyDataset + DataLoader GPU batching) ..."
        )
        ner_pipe = _build_ner_pipe(model_name)

        total_rows = 0
        files_out = 0

        for inp in inputs:
            base = inp.name
            if split_dirs:
                t_path = typed_root / base
                u_path = uniform_root / base
                if (t_path.exists() or u_path.exists()) and not args.overwrite:
                    raise SystemExit(f"Refusing to overwrite (use --overwrite): {t_path} or {u_path}")
            else:
                out_path = model_root / base
                if out_path.exists() and not args.overwrite:
                    raise SystemExit(f"Refusing to overwrite (use --overwrite): {out_path}")

            records = _load_jsonl_records(inp)
            ds = Dataset.from_list(records)
            questions = ds["question"]
            try:
                entities_list = _predict_entities_dataset(ner_pipe, ds, args.batch_size)
            except Exception as ex:  # noqa: BLE001
                errors.append({"model": model_name, "file": str(inp), "error": f"batch_ner: {ex}"})
                continue

            if split_dirs:
                dest_tf = open(t_path, "w", encoding="utf-8")
                dest_uf = open(u_path, "w", encoding="utf-8")
            else:
                dest_tf = open(out_path, "w", encoding="utf-8")
                dest_uf = None

            try:
                tf = dest_tf
                uf = dest_uf
                for i, obj in enumerate(records):
                    try:
                        q = questions[i]
                        ent_i = entities_list[i]
                        mt = _mask_from_entities(
                            ent_i, q, typed_mode=True, use_regex=use_regex
                        )
                        mu = _mask_from_entities(
                            ent_i, q, typed_mode=False, use_regex=use_regex
                        )
                        row_t = {
                            "id": obj.get("id"),
                            "index": obj.get("index"),
                            "question": q,
                            "question_masked_typed": mt,
                        }
                        row_u = {
                            "id": obj.get("id"),
                            "index": obj.get("index"),
                            "question": q,
                            "question_masked_uniform": mu,
                        }
                        if split_dirs:
                            assert uf is not None
                            tf.write(json.dumps(row_t, ensure_ascii=False) + "\n")
                            uf.write(json.dumps(row_u, ensure_ascii=False) + "\n")
                        else:
                            combined = {
                                "id": row_t["id"],
                                "index": row_t["index"],
                                "question": row_t["question"],
                                "question_masked_typed": row_t["question_masked_typed"],
                                "question_masked_uniform": row_u["question_masked_uniform"],
                            }
                            tf.write(json.dumps(combined, ensure_ascii=False) + "\n")
                        total_rows += 1
                    except Exception as ex:  # noqa: BLE001
                        errors.append({"model": model_name, "file": str(inp), "error": str(ex)})
            finally:
                tf.close()
                if uf is not None:
                    uf.close()

            files_out += 1
            if split_dirs:
                print(
                    f"  {model_name}: {inp.name} ({len(records)} rows) -> {slug}/typed|uniform"
                )
            else:
                print(f"  {model_name}: {inp.name} ({len(records)} rows) -> {slug}/{base}")

        per_model_stats[model_name] = {
            "slug": slug,
            "files": files_out,
            "total_rows": total_rows,
            "batch_size": args.batch_size,
        }

    metrics = {
        "script": "ner_mask_musique_question_chunks.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "ner_models": models,
        "batch_size": args.batch_size,
        "ner_inference": "Dataset.from_list(rows); batched pipeline forward; one entity prediction per row reused for typed + uniform masks",
        "output_layout": "split_typed_uniform_dirs" if split_dirs else "combined_single_file",
        "device": args.device,
        "regex_num_date_enabled": use_regex,
        "typed_mapping_note": "PER/PERSON->[PERSON], ORG->[ORG], LOC/GPE->[PLACE], MISC->[WORK]; unknown->[ENTITY] or [MASK]; adjacent same-type spans merged (gap<=3 chars)",
        "inputs": [str(p) for p in inputs],
        "per_model": per_model_stats,
        "errors": errors,
    }
    metrics_path = args.run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as mf:
        json.dump(metrics, mf, indent=2, ensure_ascii=False)
        mf.write("\n")

    notes = args.run_dir / "notes.md"
    notes.write_text(
        "# MusiQue question NER mask\n\n"
        f"- Seed: {args.seed}\n"
        f"- Models: {', '.join(models)}\n"
        f"- Output root: `{args.out_dir}`\n"
        f"- Regex NUM/DATE pass: {use_regex}\n"
        f"- Batch size (pipeline): {args.batch_size}\n"
        f"- Output layout: {'typed/ + uniform/ dirs' if split_dirs else 'single JSONL with typed + uniform columns'}\n"
        f"- Metrics: `{metrics_path}`\n",
        encoding="utf-8",
    )
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
