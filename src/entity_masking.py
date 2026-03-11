"""
Entity masking for questions: replace movie titles and person names with [MOVIE], [PERSON].

Loads entities from MetaQA kb.txt. Used to test similarity on structure rather than
entity overlap (e.g. for router few-shot matching).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable


def load_entities_from_kb(kb_path: Path) -> tuple[set[str], set[str]]:
    """
    Extract movies and people from kb.txt.
    Returns (movies, people). Movies are subjects; people are objects of
    directed_by, written_by, starred_actors.
    """
    movies: set[str] = set()
    people: set[str] = set()
    person_relations = {"directed_by", "written_by", "starred_actors"}

    for line in kb_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        subj, pred, obj = parts[0].strip(), parts[1].strip(), parts[2].strip()
        movies.add(subj)
        if pred in person_relations:
            people.add(obj)

    return movies, people


def build_masker(
    kb_path: Path,
    movie_placeholder: str = "[MOVIE]",
    person_placeholder: str = "[PERSON]",
) -> Callable[[str], str]:
    """
    Build a mask function that replaces movies and people with placeholders.
    Uses longest-match first to avoid partial matches (e.g. "The Godfather" before "The").
    """
    movies, people = load_entities_from_kb(kb_path)

    # Sort by length descending for longest-match-first
    movie_patterns = sorted(movies, key=len, reverse=True)
    person_patterns = sorted(people, key=len, reverse=True)

    def _replacer(text: str, patterns: list[str], placeholder: str) -> str:
        result = text
        for p in patterns:
            if not p:
                continue
            # Word-boundary-ish: avoid matching inside longer names
            escaped = re.escape(p)
            result = re.sub(rf"\b{escaped}\b", placeholder, result, flags=re.IGNORECASE)
        return result

    def mask(text: str) -> str:
        out = _replacer(text, movie_patterns, movie_placeholder)
        out = _replacer(out, person_patterns, person_placeholder)
        return out

    return mask
