from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, DefaultDict


@dataclass
class MetaQAKG:
    """
    Faithful MetaQA knowledge graph built directly from kb.txt (no inverse edges, no normalization).

    - entity_to_id: maps entity string -> int id
    - id_to_entity: list where index -> entity string
    - triples: list of (subj_id, relation, obj_id)
    - out_adj: subj_id -> list of (relation, obj_id)  (fast forward lookups)
    - rel_counts: relation -> frequency
    """
    entity_to_id: Dict[str, int]
    id_to_entity: List[str]
    triples: List[Tuple[int, str, int]]
    out_adj: DefaultDict[int, List[Tuple[str, int]]]
    rel_counts: Counter


def build_metaqa_kg(kb_path: str | Path) -> MetaQAKG:
    kb_path = Path(kb_path)
    if not kb_path.exists():
        raise FileNotFoundError(f"kb.txt not found at: {kb_path.resolve()}")

    entity_to_id: Dict[str, int] = {}
    id_to_entity: List[str] = []
    triples: List[Tuple[int, str, int]] = []
    out_adj: DefaultDict[int, List[Tuple[str, int]]] = defaultdict(list)
    rel_counts: Counter = Counter()

    def get_id(entity: str) -> int:
        """Assign a stable integer id to each unique entity string (exactly as it appears)."""
        if entity not in entity_to_id:
            entity_to_id[entity] = len(id_to_entity)
            id_to_entity.append(entity)
        return entity_to_id[entity]

    with open(kb_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) != 3:
                raise ValueError(f"Bad triple at line {line_no}: {line!r}")

            subj, rel, obj = (p.strip() for p in parts)
            if not subj or not rel or not obj:
                raise ValueError(f"Empty field at line {line_no}: {line!r}")

            sid = get_id(subj)
            oid = get_id(obj)

            triples.append((sid, rel, oid))
            out_adj[sid].append((rel, oid))
            rel_counts[rel] += 1

    return MetaQAKG(
        entity_to_id=entity_to_id,
        id_to_entity=id_to_entity,
        triples=triples,
        out_adj=out_adj,
        rel_counts=rel_counts,
    )


# --- Example usage ---
if __name__ == "__main__":
    kg = build_metaqa_kg("Data/kb.txt")

    print("Entities:", len(kg.id_to_entity))
    print("Triples:", len(kg.triples))
    print("Relations:", len(kg.rel_counts))
    print("Top relations:", kg.rel_counts.most_common(10))

    # Example: show first 5 outgoing edges for entity "Kismet"
    name = "Kismet"
    if name in kg.entity_to_id:
        sid = kg.entity_to_id[name]
        for rel, oid in kg.out_adj[sid][:5]:
            print(f"{name} --{rel}--> {kg.id_to_entity[oid]}")
