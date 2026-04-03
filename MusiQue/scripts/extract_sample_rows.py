#!/usr/bin/env python3
"""Extract specific rows (by 1-based line number) from BERT and RoBERTa masked JSONL files."""

from pathlib import Path
import json

LINES = [
    3427, 516, 2562, 1677, 1423, 2601, 2900, 1494, 2001, 1039,
    464, 878, 1043, 1881, 2543, 2958, 2632, 1701, 1338, 2541,
    2504, 2768, 3070, 946, 3241, 2422, 1197, 1616, 1813, 1368,
    658, 444, 2110, 3086, 388, 404, 1837, 656, 2848, 796,
]



BASE_DIR = Path("/cta/users/fyilmaz/Thesis---QA/MusiQue/Data/chunks_only_question_masked")
FILENAME = "musique_ans_v1.0_train_0_questions_2_hop.jsonl"
OUT_DIR = Path("/cta/users/fyilmaz/Thesis---QA/MusiQue/Data/sample_extracts")

SOURCES = {
    "bert_large_NER": BASE_DIR / "bert_large_NER" / FILENAME,
    "roberta_large_ner_english": BASE_DIR / "roberta_large_ner_english" / FILENAME,
}

line_set = set(LINES)

OUT_DIR.mkdir(parents=True, exist_ok=True)

for tag, src in SOURCES.items():
    rows: dict[int, dict] = {}
    with open(src, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            if lineno in line_set:
                rows[lineno] = json.loads(raw)
    out_path = OUT_DIR / f"sample_{tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        for ln in LINES:
            row = rows.get(ln)
            if row is None:
                print(f"  WARNING: line {ln} not found in {src.name}")
                continue
            row["_line"] = ln
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows -> {out_path}")
