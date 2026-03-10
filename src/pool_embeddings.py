"""
Embedding-based similarity for few-shot pool selection.

Loads pool, computes embeddings (with disk cache), returns top-k most similar
examples for a given question.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _pool_hash(questions: list[str]) -> str:
    """Content hash for cache invalidation."""
    blob = json.dumps(sorted(questions), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _model_slug(model_id: str) -> str:
    """Short slug for cache paths (e.g. minilm, e5-small)."""
    return model_id.split("/")[-1].replace(".", "_")[:20]


def _needs_e5_prefix(model_id: str) -> bool:
    """E5 models expect 'query:' and 'passage:' prefixes."""
    return "e5" in model_id.lower()


def load_pool(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load few_shot_decompositions.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if k in ("2hop", "3hop") and isinstance(v, list)}


def load_router_pool(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load few_shot_router.json (1hop, 2hop, 3hop)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if k in ("1hop", "2hop", "3hop") and isinstance(v, list)}


def get_router_pool_embeddings(
    pool_path: Path,
    cache_dir: Path | None = None,
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[list[dict], np.ndarray, Any]:
    """
    Return (all_items, all_embeddings, model) for router pool.
    Combines 1hop, 2hop, 3hop into one pool for similarity search when hop is unknown.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers") from None

    pool = load_router_pool(pool_path)
    cache_dir = cache_dir or pool_path.parent / "embeddings_cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_items: list[dict] = []
    for hop_key in ("1hop", "2hop", "3hop"):
        all_items.extend(pool.get(hop_key, []))

    questions = [it["question"] for it in all_items]
    content_hash = _pool_hash(questions)
    model_slug = _model_slug(model_id)
    cache_file = cache_dir / f"embeddings_router_{model_slug}_{content_hash}.npz"

    model = SentenceTransformer(model_id)
    use_prefix = _needs_e5_prefix(model_id)

    if cache_file.exists():
        loaded = np.load(cache_file, allow_pickle=True)
        emb = loaded["embeddings"]
        items = loaded["items"].tolist()
        return (items, emb, model)

    to_encode = [f"passage: {q}" for q in questions] if use_prefix else questions
    emb = model.encode(to_encode, normalize_embeddings=True)
    np.savez_compressed(cache_file, embeddings=emb, items=np.array(all_items, dtype=object))
    for f in cache_dir.glob(f"embeddings_router_{model_slug}_*.npz"):
        if f != cache_file:
            f.unlink(missing_ok=True)
    return (all_items, emb, model)


def top_k_similar_router(
    query: str,
    items: list[dict],
    embeddings: np.ndarray,
    model: Any,
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    k: int = 6,
) -> list[tuple[dict, float]]:
    """Top-k similar from combined router pool (hop unknown)."""
    use_prefix = _needs_e5_prefix(model_id)
    to_encode = [f"query: {query}"] if use_prefix else [query]
    q_emb = model.encode(to_encode, normalize_embeddings=True)[0]
    scores = np.dot(embeddings, q_emb)
    idx_scores = sorted(enumerate(scores), key=lambda x: -x[1])
    return [(items[idx], float(sim)) for idx, sim in idx_scores[:k]]


def get_pool_embeddings(
    pool_path: Path,
    cache_dir: Path | None = None,
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, tuple[list[dict], np.ndarray]]:
    """
    Return {hop_key: (items, embeddings)}.
    Uses disk cache; invalidates when pool content changes.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers") from None

    pool = load_pool(pool_path)
    cache_dir = cache_dir or pool_path.parent / "embeddings_cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, tuple[list[dict], np.ndarray]] = {}
    model = SentenceTransformer(model_id)

    model_slug = _model_slug(model_id)
    use_prefix = _needs_e5_prefix(model_id)

    for hop_key, items in pool.items():
        questions = [it["question"] for it in items]
        content_hash = _pool_hash(questions)
        cache_file = cache_dir / f"embeddings_{model_slug}_{hop_key}_{content_hash}.npz"

        if cache_file.exists():
            loaded = np.load(cache_file, allow_pickle=True)
            emb = loaded["embeddings"]
            cached_items = loaded["items"].tolist()
            result[hop_key] = (cached_items, emb)
            continue

        to_encode = [f"passage: {q}" for q in questions] if use_prefix else questions
        emb = model.encode(to_encode, normalize_embeddings=True)
        np.savez_compressed(
            cache_file,
            embeddings=emb,
            items=np.array(items, dtype=object),
        )
        # Remove stale cache files for this hop+model
        for f in cache_dir.glob(f"embeddings_{model_slug}_{hop_key}_*.npz"):
            if f != cache_file:
                f.unlink(missing_ok=True)
        result[hop_key] = (items, emb)

    return result


def top_k_similar(
    query: str,
    hop_key: str,
    pool_embeddings: dict[str, tuple[list[dict], np.ndarray]],
    k: int = 6,
    exclude_question: str | None = None,
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    model: Any = None,
) -> list[tuple[dict, float]]:
    """
    Return top-k most similar pool items for query.
    Each element is (item, cosine_similarity).
    Pass model to avoid reloading when calling repeatedly.
    """
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_id)
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers") from None

    if hop_key not in pool_embeddings:
        return []
    items, emb = pool_embeddings[hop_key]
    if not items:
        return []
    use_prefix = _needs_e5_prefix(model_id)
    to_encode = [f"query: {query}"] if use_prefix else [query]
    q_emb = model.encode(to_encode, normalize_embeddings=True)[0]
    scores = np.dot(emb, q_emb)  # cosine sim (embeddings normalized)

    idx_scores = list(enumerate(scores))
    idx_scores.sort(key=lambda x: -x[1])
    top: list[tuple[dict, float]] = []
    for idx, sim in idx_scores:
        it = items[idx]
        if exclude_question and it["question"] == exclude_question:
            continue
        top.append((it, float(sim)))
        if len(top) >= k:
            break
    return top
