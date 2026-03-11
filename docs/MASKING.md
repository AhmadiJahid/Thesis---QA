# Entity Masking for Questions

## Purpose

Replace movie titles and person names in questions with placeholders `[MOVIE]` and `[PERSON]` to enable **structure-based similarity** (e.g. router few-shot matching). Without masking, similarity is inflated by entity overlap; with masking, the model matches on question structure instead.

## Implementation

- **Source**: [`src/entity_masking.py`](../src/entity_masking.py)
- **Config**: [`configs/masking.json`](../configs/masking.json)
- **Data**: Entities from MetaQA `Data/kb.txt` (movies as subjects; people from `directed_by`, `written_by`, `starred_actors`)

## Algorithm

1. **Load entities** from KB (movies, people).
2. **Optional corpus filter**: Keep only entities that appear in a reference corpus (pool + refined questions) to reduce set size and build time.
3. **Single combined automaton**: Movies and people are merged into one Aho-Corasick automaton. Longest-match across all entity types handles overlap (e.g. "Michael Tully" person beats "Michael"/"Tully" movies; "Joe Thomas" beats "Joe").
4. **Word-boundary check** so we match whole entities, not substrings (e.g. avoid "Ted" in "directed").

## Usage

```python
from pathlib import Path
from entity_masking import build_masker, mask_question

mask_fn = build_masker(Path("Data/kb.txt"))
masked = mask_fn("who directed The Godfather")  # -> "who directed [MOVIE]"
```

With config and corpus filtering:

```python
import json
cfg = json.loads(Path("configs/masking.json").read_text())
mask_fn = build_masker(
    Path(cfg["kb_path"]),
    corpus_paths=[Path(p) for p in cfg["corpus_paths"]],
    repo_root=Path("."),
)
```

## Integration

- **Router**: Use the pre-masked pool file (`Pool/few_shot_router_masked.json`) when available; apply `mask_fn` only to queries (e.g. refined questions). Do not dynamically mask the pool, as entity overlap can corrupt pool items (e.g. "who stars in Baby Face" -> "who [MOVIE] in [MOVIE]" when "Stars" is a KB movie).
- **Test script**: `scripts/test_similarity_router.py` runs a three-way comparison; B and C use the pre-masked pool, and mask_fn is used for query masking in Mode C only.

## Dependencies

- `pyahocorasick` for multi-pattern matching
