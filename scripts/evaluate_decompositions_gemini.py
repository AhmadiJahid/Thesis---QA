#!/usr/bin/env python3
"""
Evaluate question decompositions using Gemini API.

For each (question, decomposition) pair, asks Gemini to classify:
  0 = decomposition is correct
  1 = decomposition is wrong

Outputs: metrics.json, per-item results, notes.md (runs/).
Requires GOOGLE_API_KEY in environment (e.g. from .env).
"""
from __future__ import annotations

import json
import os
import re
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

with suppress(ImportError):
    from dotenv import load_dotenv
    load_dotenv()


EVAL_PROMPT = """You are evaluating multi-hop question decompositions for a movie QA system.

Given an ORIGINAL QUESTION and its DECOMPOSITION (numbered sub-questions with [#k] referencing prior steps), determine if the decomposition is semantically correct.

A decomposition is CORRECT (0) when:
- The sub-questions logically break down the original question
- Each step is answerable and feeds into the next
- [#1], [#2], etc. correctly refer to results of previous steps
- The final step would yield the intended answer to the original question

A decomposition is WRONG (1) when:
- Steps do not match the original question's intent
- Logic is flawed (wrong order, wrong entities, missing steps)
- References [#k] are inconsistent or invalid
- The decomposition would not lead to the correct answer

Respond with:
- If correct: 0
- If wrong: 1, followed by a short sentence explaining why (e.g. "1. Step 2 asks about directors but the question asks about writers.")

---
ORIGINAL QUESTION: {question}

DECOMPOSITION:
{decomposition}
---
Answer:"""


@dataclass
class EvalItem:
    question: str
    decomposition: str
    hop_count: int | None = None
    index: int = 0


@dataclass
class GeminiEvalMetrics:
    total: int = 0
    correct: int = 0  # label 0
    wrong: int = 0    # label 1
    parse_fail: int = 0
    api_error: int = 0
    items: list[dict[str, Any]] = field(default_factory=list)


def load_items_from_json(path: Path, max_items: int | None = None) -> list[EvalItem]:
    """Load (question, decomposition) items from various JSON formats."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    items: list[EvalItem] = []

    def add(q: str, d: str, hop: int | None = None) -> None:
        if not q or not d:
            return
        items.append(EvalItem(question=q, decomposition=d, hop_count=hop, index=len(items) + 1))

    if isinstance(raw, list):
        # decomposer results.json: [{"question", "hop_count", "decomposition"}, ...]
        for i, it in enumerate(raw):
            if max_items and len(items) >= max_items:
                break
            q = it.get("question") or ""
            d = it.get("decomposition") or ""
            hop = it.get("hop_count")
            add(q, d, hop)

    elif isinstance(raw, dict):
        # stratum_assignment.json or few_shot_decompositions.json
        # {"2hop": [{"question", "decomposition"}], "3hop": [...], ...}
        # or {"2hop": {"director": [...], "actor": [...]}, ...}
        for hop_key, val in raw.items():
            if max_items and len(items) >= max_items:
                break
            hop_num = int(hop_key.replace("hop", "")) if "hop" in hop_key else None

            if isinstance(val, list):
                for it in val:
                    if max_items and len(items) >= max_items:
                        break
                    q = (it.get("question") or "") if isinstance(it, dict) else ""
                    d = (it.get("decomposition") or "") if isinstance(it, dict) else ""
                    add(q, d, hop_num)
            elif isinstance(val, dict):
                for stratum_list in val.values():
                    if not isinstance(stratum_list, list):
                        continue
                    for it in stratum_list:
                        if max_items and len(items) >= max_items:
                            break
                        q = (it.get("question") or "") if isinstance(it, dict) else ""
                        d = (it.get("decomposition") or "") if isinstance(it, dict) else ""
                        add(q, d, hop_num)

    return items


def parse_response(text: str) -> tuple[int | None, str | None]:
    """Extract label (0 or 1) and optional reason from model output.
    Returns (label, reason). reason is non-empty only when label=1.
    """
    text = (text or "").strip()
    if not (m := re.search(r"\b([01])\b", text)):
        return (None, None)
    label = int(m.group(1))
    reason: str | None = None
    if label == 1 and (after := text[m.end() :].strip()):
        reason = re.sub(r"^[.\-:]\s*", "", after).strip() or None
        if reason and len(reason) > 200:
            reason = f"{reason[:197]}..."
    return (label, reason)


def _call_gemini_with_retry(client, model_id: str, prompt: str, max_retries: int = 5) -> str:
    """Call Gemini API with exponential backoff on 429 / rate limit."""
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
            )
            return (response.text or "").strip()
        except Exception as e:
            last_err = e
            err_str = str(e)
            is_retryable = "429" in err_str or "503" in err_str or "rate" in err_str.lower()
            if is_retryable and attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                raise
    raise last_err or RuntimeError("Unexpected retry loop exit")


def evaluate_with_gemini(
    items: list[EvalItem],
    api_key: str | None = None,
    model_id: str = "gemini-2.5-flash",
    request_delay: float = 1.0,
    verbose: bool = True,
) -> GeminiEvalMetrics:
    """Call Gemini for each item; return metrics and per-item results."""
    try:
        from google import genai
    except ImportError:
        raise ImportError("Install google-genai: pip install google-genai") from None

    key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY in environment or pass api_key="
        )

    client = genai.Client(api_key=key)
    metrics = GeminiEvalMetrics(total=len(items))
    total = len(items)
    start_time = time.time()

    for i, item in enumerate(items):
        q_short = f"{item.question[:55]}…" if len(item.question) > 55 else item.question
        if verbose:
            print(f"[{i + 1}/{total}] Q: {q_short!r} …", end=" ", flush=True)

        prompt = EVAL_PROMPT.format(
            question=item.question,
            decomposition=item.decomposition,
        )
        row: dict[str, Any] = {
            "index": item.index,
            "question": item.question,
            "decomposition": item.decomposition,
            "hop_count": item.hop_count,
            "label": None,
            "wrong_reason": None,
            "raw_response": None,
            "error": None,
        }

        try:
            raw = _call_gemini_with_retry(client, model_id, prompt)
            row["raw_response"] = raw[:200]
            label, reason = parse_response(raw)
            if label is not None:
                row["label"] = label
                if label == 1 and reason:
                    row["wrong_reason"] = reason
                if label == 0:
                    metrics.correct += 1
                    if verbose:
                        print("-> 0 (correct)", flush=True)
                else:
                    metrics.wrong += 1
                    if verbose:
                        msg = f"-> 1 (wrong): {reason}" if reason else "-> 1 (wrong)"
                        print(msg[:120] + ("…" if len(msg) > 120 else ""), flush=True)
            else:
                metrics.parse_fail += 1
                row["error"] = "parse_fail"
                if verbose:
                    print(f"-> parse_fail (raw: {raw[:50]!r}…)", flush=True)
        except Exception as e:
            metrics.api_error += 1
            row["error"] = str(e)[:200]
            if verbose:
                err_short = str(e)[:60].replace("\n", " ")
                print(f"-> ERROR: {err_short}…", flush=True)

        metrics.items.append(row)

        if verbose and (i + 1) % 10 == 0 and i + 1 < total:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f"    --- {i + 1}/{total} done, ~{remaining:.0f}s left ---", flush=True)

        if request_delay > 0 and i < len(items) - 1:
            time.sleep(request_delay)

    client.close()
    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate decompositions with Gemini API (0=correct, 1=wrong)"
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=Path("Pool/stratum_assignment.json"),
        help="Path to JSON with question/decomposition (stratum, few_shot, or results.json)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit number of items to evaluate (for quick runs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: runs/gemini_eval/<timestamp>)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model ID (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for logging (evaluation is LLM-based, non-deterministic)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between API requests (default: 1.0, use 0 to disable)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-item progress output",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    input_path = args.input_path if args.input_path.is_absolute() else repo_root / args.input_path

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or repo_root / "runs" / "gemini_eval" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading items from {input_path}...")
    items = load_items_from_json(input_path, max_items=args.max_items)
    print(f"Loaded {len(items)} items.")

    if not items:
        print("No items to evaluate.")
        return

    print(f"Evaluating with Gemini ({args.model})...")
    t0 = time.time()
    metrics = evaluate_with_gemini(
        items,
        model_id=args.model,
        request_delay=args.delay,
        verbose=not args.quiet,
    )
    elapsed_sec = time.time() - t0

    # Metrics JSON
    wrong_2hop = sum(1 for r in metrics.items if r.get("label") == 1 and r.get("hop_count") == 2)
    wrong_3hop = sum(1 for r in metrics.items if r.get("label") == 1 and r.get("hop_count") == 3)
    metrics_data = {
        "run_id": run_id,
        "seed": args.seed,
        "model": args.model,
        "input_path": str(input_path),
        "total": metrics.total,
        "correct": metrics.correct,
        "wrong": metrics.wrong,
        "wrong_2hop": wrong_2hop,
        "wrong_3hop": wrong_3hop,
        "parse_fail": metrics.parse_fail,
        "api_error": metrics.api_error,
        "accuracy": metrics.correct / metrics.total if metrics.total else 0,
        "elapsed_seconds": round(elapsed_sec, 1),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_data, indent=2),
        encoding="utf-8",
    )

    # Per-item results
    (output_dir / "results.json").write_text(
        json.dumps(metrics.items, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Config snapshot
    config = {
        "run_id": run_id,
        "seed": args.seed,
        "model": args.model,
        "input_path": str(input_path),
        "max_items": args.max_items,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    # Notes
    evaluated = metrics.correct + metrics.wrong
    acc_str = f"{metrics.correct / evaluated:.2%}" if evaluated > 0 else "N/A (no items evaluated)"
    mins, secs = divmod(int(elapsed_sec), 60)
    elapsed_str = f"{mins}m {secs}s" if mins else f"{elapsed_sec:.1f}s"
    notes = f"""Gemini Decomposition Eval - {run_id}
Model: {args.model}
Input: {input_path}
Total: {metrics.total}
Correct (0): {metrics.correct}
Wrong (1): {metrics.wrong}
  - 2-hop: {wrong_2hop}
  - 3-hop: {wrong_3hop}
Parse fail: {metrics.parse_fail}
API error: {metrics.api_error}
Accuracy (of evaluated): {acc_str}
Total time: {elapsed_str} ({elapsed_sec:.1f}s)
"""
    (output_dir / "notes.md").write_text(notes, encoding="utf-8")

    print(f"\nResults saved to {output_dir}")
    print(f"  Total time: {elapsed_str}")
    print(f"  Correct: {metrics.correct}  Wrong: {metrics.wrong}")
    print(f"  Parse fail: {metrics.parse_fail}  API error: {metrics.api_error}")
    if evaluated:
        print(f"  Accuracy: {metrics.correct / evaluated:.2%}")
    else:
        print("  Accuracy: N/A (no items evaluated successfully)")


if __name__ == "__main__":
    main()
