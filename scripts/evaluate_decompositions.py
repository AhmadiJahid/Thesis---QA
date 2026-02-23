import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Literal, Any
from collections import defaultdict, Counter
from pathlib import Path
from scripts.kg import build_metaqa_kg, MetaQAKG


# ============================================================
# EXPECTED KG INTERFACE
# ============================================================
# kg must have:
# - kg.triples: List[Tuple[int, str, int]]        (sid, rel, oid)
# - kg.out_adj: Dict[int, List[Tuple[str, int]]]  sid -> [(rel, oid)...]
# - kg.entity_to_id: Dict[str, int]
# - kg.id_to_entity: List[str]
# ============================================================


# ============================================================
# 1) Reverse index (derived; does not change KG)
# ============================================================

def build_reverse_index(kg: MetaQAKG) -> Dict[Tuple[str, int], Set[int]]:
    rev: Dict[Tuple[str, int], Set[int]] = defaultdict(set)
    for sid, rel, oid in kg.triples:
        rev[(rel, oid)].add(sid)
    return rev


# ============================================================
# 2) Compiler: Decomposition English -> Query Ops
# ============================================================

OpType = Literal[
    "VALUES_OF_MOVIE",            # movie -> values
    "MOVIES_WITH_VALUE",          # value -> movies (reverse index)
    "PROJECT_VALUES_FROM_MOVIES"  # movieset -> values
]

@dataclass(frozen=True)
class StepOp:
    op: OpType
    relation: str
    movie: Optional[str] = None        # for VALUES_OF_MOVIE
    entity: Optional[str] = None       # for MOVIES_WITH_VALUE (explicit value entity)
    ref_step: Optional[int] = None     # for MOVIES_WITH_VALUE (values from previous step) or PROJECT_VALUES_FROM_MOVIES


PLACEHOLDER_RX = re.compile(r"\[#(\d+)\]")
STEP_PREFIX_RX = re.compile(r"^\s*\d+\.\s*")

def normalize_step_text(step_line: str) -> str:
    step = STEP_PREFIX_RX.sub("", step_line).strip()
    step = step.replace("’", "'").strip()
    return step


# ---- Relation inference (improved) ----
# We infer the *target KG relation* needed for the step.
# IMPORTANT: Many “actor” wordings all map to MetaQA’s starred_actors relation.
REL_RULES: List[Tuple[re.Pattern, str]] = [
    # Actors (movie <-> actor)
    (re.compile(r"\b(actor|actors|acted|act|starred|star|appeared|appear|played|cast|co-starred|co-star)\b", re.I), "starred_actors"),

    # Directors
    (re.compile(r"\b(director|directed|direct)\b", re.I), "directed_by"),

    # Writers / screenwriters
    (re.compile(r"\b(writer|writers|wrote|written|screenwriter|screenwriters|scriptwriter|scriptwriters|screenplay)\b", re.I), "written_by"),

    # Genres / types (MetaQA uses has_genre)
    (re.compile(r"\b(genre|genres|type|types)\b", re.I), "has_genre"),

    # Languages
    (re.compile(r"\b(language|languages)\b", re.I), "in_language"),

    # Years / release dates (MetaQA uses release_year)
    (re.compile(r"\b(release year|release years|released|release date|release dates|year|years)\b", re.I), "release_year"),

    # Tags / ratings / votes
    (re.compile(r"\b(tags?)\b", re.I), "has_tags"),
    (re.compile(r"\b(imdb rating)\b", re.I), "has_imdb_rating"),
    (re.compile(r"\b(imdb votes)\b", re.I), "has_imdb_votes"),
]

def infer_relation(step: str) -> str:
    for rx, rel in REL_RULES:
        if rx.search(step):
            return rel
    raise ValueError(f"Cannot infer relation from: {step!r}")


# ---- Step template patterns (compiler) ----

# A) Person -> Movies (reverse lookup)
# "What movies did PERSON star in?"
RX_PERSON_DID_VERB_IN = re.compile(
    r"^what\s+(movies|films).*\b(did|does|was|were)\b\s+(?P<person>.+?)\s+(?:an\s+)?(?:actor\s+)?(?:act|acted|star|starred|appear|appeared).*\bin\??$",
    re.I
)
# Also handle: "What movies was PERSON in?"
RX_PERSON_WAS_IN = re.compile(
    r"^what\s+(movies|films).*\bwas\b\s+(?P<person>.+?)\s+\bin\??$",
    re.I
)
# Also handle: "What movies did PERSON direct / write?"
RX_PERSON_DIRECT_WRITE = re.compile(
    r"^what\s+(movies|films).*\b(did|does)\b\s+(?P<person>.+?)\s+\b(direct|directed|write|wrote)\b.*\??$",
    re.I
)

# B) Movie -> Values (forward lookup)
# "Who directed MOVIE?" / "Who wrote MOVIE?" / "Who starred in MOVIE?" / "What actors were in MOVIE?"
RX_WHO_DIRECTED_MOVIE = re.compile(r"^who\s+directed\s+(?P<movie>.+?)\??$", re.I)
RX_WHO_WROTE_MOVIE = re.compile(r"^who\s+(?:wrote|is\s+the\s+writer\s+of|is\s+the\s+screenwriter\s+of|is\s+listed\s+as\s+screenwriter\s+of)\s+(?P<movie>.+?)\??$", re.I)
RX_WHO_STARRED_IN_MOVIE = re.compile(r"^who\s+(?:starred|acted|appeared)\s+in\s+(?P<movie>.+?)\??$", re.I)
RX_WHO_ARE_ACTORS_IN_MOVIE = re.compile(r"^who\s+(?:are|were)\s+the\s+(?:actors|actor|cast)\s+in\s+(?P<movie>.+?)\??$", re.I)
RX_WHAT_ACTORS_IN_MOVIE = re.compile(r"^what\s+(?:actors|actor|cast)\s+(?:were|was|are)\s+in\s+(?P<movie>.+?)\??$", re.I)

# C) Movie -> Values with “of”
RX_WHO_IS_THE_X_OF_MOVIE = re.compile(r"^who\s+is\s+the\s+.+?\s+of\s+(?P<movie>.+?)\??$", re.I)

# D) Movie -> Values with “was MOVIE directed by / written by”
RX_WHAT_MOVIES_WAS_MOVIE_DIRECTED_BY = re.compile(r"^what\s+movies?\s+was\s+(?P<movie>.+?)\s+directed\s+by\??$", re.I)
RX_WHAT_MOVIES_WAS_MOVIE_WRITTEN_BY = re.compile(r"^what\s+movies?\s+was\s+(?P<movie>.+?)\s+(?:screenplay\s+)?written\s+by\??$", re.I)

# E) Placeholder-based steps
# "What other movies were directed by [#1]?" -> MOVIES_WITH_VALUE(ref_step=1)
RX_OTHER_MOVIES_BY_PLACEHOLDER = re.compile(r"^what\s+other\s+(movies|films).*\bby\b\s+\[#(?P<k>\d+)\]\??$", re.I)

# "Who directed [#1]?" / "Who wrote [#1]?" / "Who starred in [#1]?" / "What languages are [#1] in?"
# -> PROJECT_VALUES_FROM_MOVIES(ref_step=k)
RX_PROJECT_VALUES_FROM_MOVIESET = re.compile(r"^(who|what|when).*\[#(?P<k>\d+)\].*$", re.I)

# Note: if step contains [#k] and is NOT “other movies by [#k]”, it’s almost always projection from movieset.


def compile_decomposition(decomposition: str) -> List[StepOp]:
    lines = [ln.strip() for ln in decomposition.split("\n") if ln.strip()]
    ops: List[StepOp] = []

    for ln in lines:
        step = normalize_step_text(ln)
        rel = infer_relation(step)

        # ----- Placeholder count
        ph = PLACEHOLDER_RX.search(step)
        has_ph = ph is not None

        # ======================================================
        # 1) Placeholder steps
        # ======================================================

        # "What other movies were ... by [#k]?"
        m = RX_OTHER_MOVIES_BY_PLACEHOLDER.match(step)
        if m:
            k = int(m.group("k"))
            # reverse lookup: values from step k -> movies
            ops.append(StepOp(op="MOVIES_WITH_VALUE", relation=rel, ref_step=k))
            continue

        # If step has placeholder and isn't "other movies by", treat as projection from movieset
        if has_ph:
            k = int(ph.group(1))
            ops.append(StepOp(op="PROJECT_VALUES_FROM_MOVIES", relation=rel, ref_step=k))
            continue

        # ======================================================
        # 2) Non-placeholder steps
        # ======================================================

        # Person -> movies
        m = RX_PERSON_DID_VERB_IN.match(step)
        if m:
            person = m.group("person").strip().rstrip("?")
            ops.append(StepOp(op="MOVIES_WITH_VALUE", relation=rel, entity=person))
            continue

        m = RX_PERSON_WAS_IN.match(step)
        if m:
            person = m.group("person").strip().rstrip("?")
            # This is almost always asking filmography => starred_actors
            ops.append(StepOp(op="MOVIES_WITH_VALUE", relation="starred_actors", entity=person))
            continue

        m = RX_PERSON_DIRECT_WRITE.match(step)
        if m:
            person = m.group("person").strip().rstrip("?")
            # rel inference already gives directed_by or written_by depending on wording
            ops.append(StepOp(op="MOVIES_WITH_VALUE", relation=rel, entity=person))
            continue

        # Movie -> values (direct patterns)
        m = RX_WHO_DIRECTED_MOVIE.match(step)
        if m:
            movie = m.group("movie").strip().rstrip("?")
            ops.append(StepOp(op="VALUES_OF_MOVIE", relation="directed_by", movie=movie))
            continue

        m = RX_WHO_WROTE_MOVIE.match(step)
        if m:
            movie = m.group("movie").strip().rstrip("?")
            ops.append(StepOp(op="VALUES_OF_MOVIE", relation="written_by", movie=movie))
            continue

        m = RX_WHO_STARRED_IN_MOVIE.match(step)
        if m:
            movie = m.group("movie").strip().rstrip("?")
            ops.append(StepOp(op="VALUES_OF_MOVIE", relation="starred_actors", movie=movie))
            continue

        m = RX_WHO_ARE_ACTORS_IN_MOVIE.match(step) or RX_WHAT_ACTORS_IN_MOVIE.match(step)
        if m:
            movie = m.group("movie").strip().rstrip("?")
            ops.append(StepOp(op="VALUES_OF_MOVIE", relation="starred_actors", movie=movie))
            continue

        m = RX_WHO_IS_THE_X_OF_MOVIE.match(step)
        if m:
            movie = m.group("movie").strip().rstrip("?")
            ops.append(StepOp(op="VALUES_OF_MOVIE", relation=rel, movie=movie))
            continue

        m = RX_WHAT_MOVIES_WAS_MOVIE_DIRECTED_BY.match(step)
        if m:
            movie = m.group("movie").strip().rstrip("?")
            ops.append(StepOp(op="VALUES_OF_MOVIE", relation="directed_by", movie=movie))
            continue

        m = RX_WHAT_MOVIES_WAS_MOVIE_WRITTEN_BY.match(step)
        if m:
            movie = m.group("movie").strip().rstrip("?")
            ops.append(StepOp(op="VALUES_OF_MOVIE", relation="written_by", movie=movie))
            continue

        # If we reach here: template not supported
        raise NotImplementedError(f"Unsupported step template: {step!r}")

    return ops


# ============================================================
# 3) Executor
# ============================================================

class KGExecutionError(Exception):
    pass

def must_entity_id(kg: MetaQAKG, name: str) -> int:
    if name not in kg.entity_to_id:
        raise KeyError(name)
    return kg.entity_to_id[name]

def exec_ops(kg: MetaQAKG, reverse_index: Dict[Tuple[str, int], Set[int]], ops: List[StepOp]) -> Set[int]:
    """
    Executes ops sequentially.
    Returns a set of IDs (movies or values depending on last op).
    """
    results: Dict[int, Set[int]] = {}

    for i, op in enumerate(ops, start=1):
        if op.op == "VALUES_OF_MOVIE":
            if not op.movie:
                raise KGExecutionError(f"Step {i} missing movie")
            mid = must_entity_id(kg, op.movie)
            out_vals = {oid for rel, oid in kg.out_adj[mid] if rel == op.relation}
            results[i] = out_vals
            continue

        if op.op == "MOVIES_WITH_VALUE":
            movie_ids: Set[int] = set()

            if op.entity is not None:
                vid = must_entity_id(kg, op.entity)
                movie_ids |= reverse_index.get((op.relation, vid), set())
                results[i] = movie_ids
                continue

            if op.ref_step is not None:
                prev_vals = results.get(op.ref_step)
                if prev_vals is None:
                    raise KGExecutionError(f"Step {i} refers to missing step #{op.ref_step}")
                for vid in prev_vals:
                    movie_ids |= reverse_index.get((op.relation, vid), set())
                results[i] = movie_ids
                continue

            raise KGExecutionError(f"Step {i} missing entity/ref")
        
        if op.op == "PROJECT_VALUES_FROM_MOVIES":
            if op.ref_step is None:
                raise KGExecutionError(f"Step {i} missing ref_step")
            prev_movies = results.get(op.ref_step)
            if prev_movies is None:
                raise KGExecutionError(f"Step {i} refers to missing step #{op.ref_step}")

            vals: Set[int] = set()
            for mid in prev_movies:
                for rel, oid in kg.out_adj[mid]:
                    if rel == op.relation:
                        vals.add(oid)

            results[i] = vals
            continue

        raise KGExecutionError(f"Unknown op: {op.op}")

    return results[len(ops)]


# ============================================================
# 4) Decomposition rate (compile + execute) with reason breakdown
# ============================================================

@dataclass
class DecompositionMetrics:
    total: int
    compiled_ok: int
    executed_ok: int
    compile_fail: int
    exec_fail: int
    compile_fail_reasons: Counter
    exec_fail_reasons: Counter

def evaluate_decomposition_rate(
    kg: MetaQAKG, 
    results_json_path: str, 
    max_items: Optional[int] = None,
    output_dir: Optional[str] = None
) -> DecompositionMetrics:
    with open(results_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError("results.json must be a list of dicts")

    if max_items is not None:
        items = items[:max_items]

    reverse_index = build_reverse_index(kg)

    compile_fail_reasons = Counter()
    exec_fail_reasons = Counter()

    compiled_ok = executed_ok = compile_fail = exec_fail = 0
    
    success_items = []
    compile_fail_items = []
    exec_fail_items = []

    for it in items:
        dec = it.get("decomposition")
        if not isinstance(dec, str) or not dec.strip():
            compile_fail += 1
            compile_fail_reasons["missing_decomposition"] += 1
            compile_fail_items.append({**it, "error_reason": "missing_decomposition"})
            continue

        # Compile
        try:
            ops = compile_decomposition(dec)
            # We don't increment compiled_ok yet, we wait for execution success
        except NotImplementedError:
            compile_fail += 1
            compile_fail_reasons["unsupported_template"] += 1
            compile_fail_items.append({**it, "error_reason": "unsupported_template"})
            continue
        except ValueError:
            compile_fail += 1
            compile_fail_reasons["cannot_infer_relation"] += 1
            compile_fail_items.append({**it, "error_reason": "cannot_infer_relation"})
            continue
        except Exception as e:
            compile_fail += 1
            compile_fail_reasons["compile_error_other"] += 1
            compile_fail_items.append({**it, "error_reason": f"compile_error_other: {str(e)}"})
            continue

        compiled_ok += 1

        # Execute
        try:
            results = exec_ops(kg, reverse_index, ops)
            executed_ok += 1
            readable_results = [kg.id_to_entity[rid] for rid in results]
            success_items.append({**it, "ops": [str(o) for o in ops], "kg_results": readable_results})
        except KeyError as e:
            exec_fail += 1
            exec_fail_reasons["entity_not_in_kb"] += 1
            exec_fail_items.append({**it, "ops": [str(o) for o in ops], "error_reason": f"entity_not_in_kb: {str(e)}"})
        except KGExecutionError as e:
            exec_fail += 1
            exec_fail_reasons["bad_reference_or_plan"] += 1
            exec_fail_items.append({**it, "ops": [str(o) for o in ops], "error_reason": f"bad_reference_or_plan: {str(e)}"})
        except Exception as e:
            exec_fail += 1
            exec_fail_reasons["exec_error_other"] += 1
            exec_fail_items.append({**it, "ops": [str(o) for o in ops], "error_reason": f"exec_error_other: {str(e)}"})

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        with open(out_path / "success.json", "w", encoding="utf-8") as f:
            json.dump(success_items, f, indent=2, ensure_ascii=False)
        with open(out_path / "compile_fail.json", "w", encoding="utf-8") as f:
            json.dump(compile_fail_items, f, indent=2, ensure_ascii=False)
        with open(out_path / "exec_fail.json", "w", encoding="utf-8") as f:
            json.dump(exec_fail_items, f, indent=2, ensure_ascii=False)
        print(f"\nSaved error analysis files to: {output_dir}")

    total = len(items)
    return DecompositionMetrics(
        total=total,
        compiled_ok=compiled_ok,
        executed_ok=executed_ok,
        compile_fail=compile_fail,
        exec_fail=exec_fail,
        compile_fail_reasons=compile_fail_reasons,
        exec_fail_reasons=exec_fail_reasons,
    )


# ============================================================
# 5) Convenience: run + print
# ============================================================

def print_metrics(m: DecompositionMetrics) -> None:
    if m.total == 0:
        print("No items.")
        return
    print("\n" + "="*40)
    print("   DECOMPOSITION EXECUTION METRICS")
    print("="*40)
    print(f"Total Questions:  {m.total}")
    print(f"Compiled OK:      {m.compiled_ok} ({m.compiled_ok/m.total:.2%})")
    print(f"Executed OK:      {m.executed_ok} ({m.executed_ok/m.total:.2%})")
    print(f"Compile fail:     {m.compile_fail} ({m.compile_fail/m.total:.2%})")
    print(f"Exec fail:        {m.exec_fail} ({m.exec_fail/m.total:.2%})")
    print("\nCompile fail reasons:")
    for k, v in m.compile_fail_reasons.most_common():
        print(f"  - {k}: {v}")
    print("\nExec fail reasons:")
    for k, v in m.exec_fail_reasons.most_common():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate decompositions on MetaQA KG")
    parser.add_argument("results_path", type=Path, nargs="?", default=Path("runs/decomposer/20260123_072902/results.json"),
                        help="Path to decomposer results.json")
    parser.add_argument("--kb", type=Path, default=Path("Data/kb.txt"), help="Path to KG kb.txt")
    args = parser.parse_args()

    # 1) Build KG
    print(f"Loading KG from {args.kb}...")
    kg = build_metaqa_kg(str(args.kb))

    # 2) Path to your results.json
    results_path = args.results_path
    
    # Automatically set output directory based on results_path
    results_dir = Path(results_path).parent
    output_analysis_dir = results_dir / "analysis"
    
    if not Path(results_path).exists():
        print(f"Error: Results file not found at {results_path}")
    else:
        print(f"Evaluating decompositions in {results_path}...")
        metrics = evaluate_decomposition_rate(
            kg, 
            results_path, 
            output_dir=output_analysis_dir
        )
        print_metrics(metrics)


        print("\n" + "="*40)
        print("   DECOMPOSITION EXECUTION METRICS")
        print("="*40)
        print(f"Total Questions:  {metrics.total}")
        print(f"Compiled OK:      {metrics.compiled_ok} ({metrics.compiled_ok/metrics.total:.2%})")
        print(f"Executed OK:      {metrics.executed_ok} ({metrics.executed_ok/metrics.total:.2%})")
        
        print("\nCompile Fail Reasons:")
        for reason, count in metrics.compile_fail_reasons.items():
            print(f"  - {reason}: {count}")
            
        print("\nExecution Fail Reasons:")
        for reason, count in metrics.exec_fail_reasons.items():
            print(f"  - {reason}: {count}")
