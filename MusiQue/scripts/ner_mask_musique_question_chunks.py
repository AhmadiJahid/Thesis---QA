#!/usr/bin/env python3
"""
Improved NER masking for MusiQue question JSONLs.

Key changes from the user's original script:
- defaults to RoBERTa-large NER first (cleaner spans in your sample outputs)
- uses a safer typed label set: PERSON / ORG / PLACE / DATE / NUM / ENTITY
  instead of mapping all MISC spans to [WORK]
- adds span filtering for generic/common false positives
- compresses broken adjacent placeholder runs such as
  [PERSON][PLACE] [PERSON][PLACE][PERSON] -> [PERSON]
- removes artifacts like [ENTITY]3 -> [ENTITY]
- supports a typed output and a uniform [MASK] output from the same spans
- keeps the same JSONL I/O style as your current pipeline

Example:
    python MusiQue/scripts/ner_mask_musique_question_chunks_fixed.py \
      --seed 42 \
      --device 0 \
      --input-glob 'MusiQue/Data/chunks_only_question/musique_ans_v1.0_train_0_questions_*.jsonl' \
      --overwrite
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

_DEFAULT_MODELS = (
    "Jean-Baptiste/roberta-large-ner-english",
    "dslim/bert-large-NER",
)

_UNIFORM_TOKEN = "[MASK]"
_FALLBACK_TYPED = "[ENTITY]"
_TYPED_PRIORITY = {
    "[PERSON]": 6,
    "[ORG]": 5,
    "[PLACE]": 4,
    "[DATE]": 3,
    "[NUM]": 2,
    "[ENTITY]": 1,
}
_PLACEHOLDER_RE = re.compile(r"\[(?:PERSON|ORG|PLACE|DATE|NUM|ENTITY|MASK)\]")

# Conservative English date patterns.
_DATE_RES = (
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{4}s\b"),
    re.compile(r"\b\d{4}\b"),
)
_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?(?:\s?(?:%|percent|million|billion|thousand))?\b", re.IGNORECASE)

# Protect common generic phrases that often should not be masked as entities.
_GENERIC_REJECTIONS = {
    "senate",
    "president of the senate",
    "house",
    "parliament",
    "government",
    "earth",
    "world",
}

# Ambiguous but common phrases we would rather keep unmasked than mask incorrectly.
_LITERAL_KEEP_LOWER = {
    "lok sabha",
    "doctor who",
    "la liga",
    "united states",
    "us",
    "u.s.",
}

# Simple literal spans that are often worth masking if the NER model misses them.
_LITERAL_TYPE_HINTS = {
    "united states": "[PLACE]",
    "u.s.": "[PLACE]",
    "us": "[PLACE]",
    "india": "[PLACE]",
    "england": "[PLACE]",
}


def _sanitize_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s.split("/")[-1]).strip("_")


def _load_tokenizer_for_ner(model_name: str) -> Any:
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print("  Note: using slow tokenizer (use_fast=False).")
        return tok


def _typed_placeholders() -> dict[str, str]:
    return {
        "PER": "[PERSON]",
        "PERSON": "[PERSON]",
        "ORG": "[ORG]",
        "LOC": "[PLACE]",
        "GPE": "[PLACE]",
        "FAC": "[PLACE]",
        "DATE": "[DATE]",
        "TIME": "[DATE]",
        "CARDINAL": "[NUM]",
        "ORDINAL": "[NUM]",
        "QUANTITY": "[NUM]",
        "PERCENT": "[NUM]",
        "MONEY": "[NUM]",
        # Ambiguous classes go to a safe fallback instead of [WORK].
        "MISC": "[ENTITY]",
        "NORP": "[ENTITY]",
        "EVENT": "[ENTITY]",
        "WORK_OF_ART": "[ENTITY]",
        "PRODUCT": "[ENTITY]",
        "LAW": "[ENTITY]",
        "LANGUAGE": "[ENTITY]",
    }


def _normalize_entity_label(label: object) -> str:
    s = str(label).strip()
    if len(s) >= 2 and s[1] == "-":
        s = s[2:]
    return s.upper()


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
        default=REPO_ROOT / "MusiQue" / "Data" / "chunks_only_question_masked_fixed",
        help="Root directory; each model writes under <slug>/",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "musique_question_ner_mask_fixed",
        help="metrics.json + notes.md",
    )
    p.add_argument(
        "--ner-model",
        action="append",
        dest="ner_models",
        default=None,
        help="HF model id (repeatable). If omitted, defaults to roberta-large then bert-large.",
    )
    p.add_argument("--device", type=int, default=-1, help="Transformers device id (-1 = CPU)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--split-typed-uniform-dirs",
        action="store_true",
        help="Write <slug>/typed/ and <slug>/uniform/ instead of one combined file",
    )
    p.add_argument(
        "--no-regex-num-date",
        action="store_true",
        help="Disable heuristic [NUM]/[DATE] regex pass",
    )
    p.add_argument(
        "--literal-place-hints",
        action="store_true",
        help="Also mask a small literal list such as U.S. / United States when the NER model misses them",
    )
    return p.parse_args()


def _ner_placeholder(label: object, typed_mode: bool) -> str:
    if not typed_mode:
        return _UNIFORM_TOKEN
    key = _normalize_entity_label(label)
    return _typed_placeholders().get(key, _FALLBACK_TYPED)


def _is_probably_generic_or_bad_span(span_text: str, label: str) -> bool:
    raw = span_text.strip()
    low = raw.lower()
    if not raw:
        return True
    if low in _GENERIC_REJECTIONS:
        return True
    # Reject single lowercase common tokens unless they are strongly typed.
    if raw.islower() and " " not in raw and label not in {"[DATE]", "[NUM]"}:
        return True
    # Reject pure punctuation.
    if not any(ch.isalnum() for ch in raw):
        return True
    return False


def _score(ent: dict[str, Any]) -> float:
    s = ent.get("score")
    try:
        return float(s)
    except Exception:
        return 0.0


def _entities_to_spans(
    entities: list[dict[str, Any]] | None,
    text: str,
    *,
    typed_mode: bool,
) -> list[tuple[int, int, str]]:
    if not entities:
        return []

    spans: list[tuple[int, int, str, float]] = []
    for ent in entities:
        start = ent.get("start")
        end = ent.get("end")
        if start is None or end is None:
            continue
        s = int(start)
        e = int(end)
        if s >= e or s < 0 or e > len(text):
            continue
        ph = _ner_placeholder(ent.get("entity_group") or ent.get("entity") or ent.get("label"), typed_mode)
        piece = text[s:e]
        if piece.strip().lower() in _LITERAL_KEEP_LOWER:
            continue
        if _is_probably_generic_or_bad_span(piece, ph):
            continue
        spans.append((s, e, ph, _score(ent)))

    return _merge_and_clean_spans(text, spans)


def _merge_and_clean_spans(
    text: str,
    spans: list[tuple[int, int, str, float]],
) -> list[tuple[int, int, str]]:
    if not spans:
        return []

    spans = sorted(spans, key=lambda x: (x[0], x[1], -x[3]))
    merged: list[tuple[int, int, str, float]] = []
    for s, e, ph, sc in spans:
        if not merged:
            merged.append((s, e, ph, sc))
            continue

        ps, pe, pph, psc = merged[-1]
        gap = text[pe:s] if s >= pe else ""
        # Overlap: keep the longer span, or the higher score if similar.
        if s < pe:
            prev_len = pe - ps
            cur_len = e - s
            choose_cur = (cur_len > prev_len) or (cur_len == prev_len and sc > psc)
            if choose_cur:
                merged[-1] = (s, e, ph, sc)
            continue

        # Merge same placeholder across tiny punctuation/space gaps.
        if pph == ph and s >= pe and gap and len(gap) <= 4 and not any(ch.isalnum() for ch in gap):
            merged[-1] = (ps, e, ph, max(psc, sc))
            continue

        merged.append((s, e, ph, sc))

    return [(s, e, ph) for s, e, ph, _ in merged]


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


def _regex_spans(text: str, blocked: list[tuple[int, int]], typed: bool) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []

    date_ivs: list[tuple[int, int]] = []
    for rx in _DATE_RES:
        for m in rx.finditer(text):
            if not _overlaps_blocked(m.start(), m.end(), blocked):
                date_ivs.append((m.start(), m.end()))
    date_merged = _merge_intervals(date_ivs)
    blocked2 = blocked + date_merged
    for s, e in date_merged:
        out.append((s, e, "[DATE]" if typed else _UNIFORM_TOKEN))

    num_ivs: list[tuple[int, int]] = []
    for m in _NUM_RE.finditer(text):
        if not _overlaps_blocked(m.start(), m.end(), blocked2):
            num_ivs.append((m.start(), m.end()))
    for s, e in _merge_intervals(num_ivs):
        out.append((s, e, "[NUM]" if typed else _UNIFORM_TOKEN))

    return out


def _literal_hint_spans(text: str, blocked: list[tuple[int, int]], typed: bool) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    text_lower = text.lower()
    for lit, placeholder in _LITERAL_TYPE_HINTS.items():
        start = 0
        while True:
            idx = text_lower.find(lit, start)
            if idx == -1:
                break
            end = idx + len(lit)
            if not _overlaps_blocked(idx, end, blocked):
                out.append((idx, end, placeholder if typed else _UNIFORM_TOKEN))
            start = end
    return out


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


def _best_placeholder(placeholders: Iterable[str], uniform: bool) -> str:
    vals = list(placeholders)
    if uniform:
        return _UNIFORM_TOKEN
    counts = Counter(vals)
    # choose most common; break ties with priority
    best = sorted(counts.items(), key=lambda kv: (kv[1], _TYPED_PRIORITY.get(kv[0], 0)), reverse=True)[0][0]
    return best


def _compress_placeholder_runs(text: str, typed_mode: bool) -> str:
    # Collapse runs such as [PERSON][PLACE] [PERSON] -> [PERSON]
    pattern = re.compile(
        r"((?:\[(?:PERSON|ORG|PLACE|DATE|NUM|ENTITY|MASK)\](?:[\s'’.,;/:-]*)?){2,})"
    )

    def repl(match: re.Match[str]) -> str:
        chunk = match.group(1)
        tokens = _PLACEHOLDER_RE.findall(chunk)
        return _best_placeholder(tokens, uniform=not typed_mode)

    out = pattern.sub(repl, text)
    # Remove attached digits: [ENTITY]3 -> [ENTITY]
    out = re.sub(r"(\[(?:PERSON|ORG|PLACE|DATE|NUM|ENTITY|MASK)\])(?=\d)", r"\1 ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _cleanup_masked_text(text: str, typed_mode: bool) -> str:
    out = _compress_placeholder_runs(text, typed_mode=typed_mode)
    # Normalize spaces before punctuation and possessives.
    out = re.sub(r"\s+([,.;:?!])", r"\1", out)
    out = re.sub(r"\[(PERSON|ORG|PLACE|DATE|NUM|ENTITY|MASK)\]\s+'s", r"[\1]'s", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def _mask_from_entities(
    entities: list[dict[str, Any]] | None,
    text: str,
    *,
    typed_mode: bool,
    use_regex: bool,
    use_literal_hints: bool,
) -> str:
    if not text:
        return ""

    spans = _entities_to_spans(entities, text, typed_mode=typed_mode)
    blocked = [(s, e) for s, e, _ in spans]

    extra: list[tuple[int, int, str]] = []
    if use_regex:
        extra.extend(_regex_spans(text, blocked, typed=typed_mode))
        blocked = blocked + [(s, e) for s, e, _ in extra]
    if use_literal_hints:
        extra.extend(_literal_hint_spans(text, blocked, typed=typed_mode))

    all_spans = sorted(spans + extra, key=lambda x: (x[0], x[1]))
    # Final non-overlap pass.
    final_spans: list[tuple[int, int, str]] = []
    last_end = -1
    for s, e, ph in all_spans:
        if s < last_end:
            continue
        final_spans.append((s, e, ph))
        last_end = e

    masked = _apply_spans(text, final_spans)
    return _cleanup_masked_text(masked, typed_mode=typed_mode)


def _predict_entities_dataset(ner_pipe: Any, ds: Dataset, batch_size: int) -> list[list[dict[str, Any]]]:
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


def main() -> None:
    args = _parse_args()
    inputs = _resolve_inputs(args)
    models = args.ner_models if args.ner_models else list(_DEFAULT_MODELS)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import pipeline

    use_regex = not args.no_regex_num_date
    use_literal_hints = args.literal_place_hints

    def _build_ner_pipe(name: str) -> Any:
        tokenizer = _load_tokenizer_for_ner(name)
        return pipeline(
            "token-classification",
            model=name,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=args.device,
        )

    per_model_stats: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []

    for model_name in models:
        slug = _sanitize_slug(model_name)
        model_root = args.out_dir / slug
        typed_root = model_root / "typed"
        uniform_root = model_root / "uniform"
        split_dirs = args.split_typed_uniform_dirs
        if split_dirs:
            typed_root.mkdir(parents=True, exist_ok=True)
            uniform_root.mkdir(parents=True, exist_ok=True)
        else:
            model_root.mkdir(parents=True, exist_ok=True)

        print(f"Loading NER pipeline: {model_name} (batch_size={args.batch_size}) ...")
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
            except Exception as ex:
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
                            ent_i,
                            q,
                            typed_mode=True,
                            use_regex=use_regex,
                            use_literal_hints=use_literal_hints,
                        )
                        mu = _mask_from_entities(
                            ent_i,
                            q,
                            typed_mode=False,
                            use_regex=use_regex,
                            use_literal_hints=use_literal_hints,
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
                    except Exception as ex:
                        errors.append({"model": model_name, "file": str(inp), "error": str(ex)})
            finally:
                tf.close()
                if uf is not None:
                    uf.close()

            files_out += 1
            if split_dirs:
                print(f"  {model_name}: {inp.name} ({len(records)} rows) -> {slug}/typed|uniform")
            else:
                print(f"  {model_name}: {inp.name} ({len(records)} rows) -> {slug}/{base}")

        per_model_stats[model_name] = {
            "slug": slug,
            "files": files_out,
            "total_rows": total_rows,
            "batch_size": args.batch_size,
        }

    metrics = {
        "script": "ner_mask_musique_question_chunks_fixed.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "ner_models": models,
        "batch_size": args.batch_size,
        "device": args.device,
        "regex_num_date_enabled": use_regex,
        "literal_place_hints_enabled": use_literal_hints,
        "typed_mapping_note": "PER/PERSON->[PERSON], ORG->[ORG], LOC/GPE/FAC->[PLACE], DATE/TIME->[DATE], numeric labels->[NUM], ambiguous labels->[ENTITY]",
        "cleanup_note": "collapses broken adjacent placeholder runs and removes artifacts like [ENTITY]3",
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
        "# MusiQue question NER mask (fixed)\n\n"
        f"- Seed: {args.seed}\n"
        f"- Models: {', '.join(models)}\n"
        f"- Output root: `{args.out_dir}`\n"
        f"- Regex NUM/DATE pass: {use_regex}\n"
        f"- Literal hints: {use_literal_hints}\n"
        f"- Batch size: {args.batch_size}\n"
        f"- Metrics: `{metrics_path}`\n",
        encoding="utf-8",
    )
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
