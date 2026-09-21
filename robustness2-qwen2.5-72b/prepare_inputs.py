#!/usr/bin/env python3
"""Freeze the exact concepts used in the cross-evaluator robustness check."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any

from prompt import EVALUATION_CRITERIA, EVALUATOR_PROMPT


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
CANONICAL_RUN = "2026-08-08_deepseekv32-evalrank15-full"
INPUT_DIR = HERE / "inputs"
CONCEPT_FIELDS = [
    "slot_id",
    "group",
    "lineage_id",
    "parent_slot_id",
    "text_sha256",
    "source_label",
    "source_path",
    "deepseek_win_rate",
    "is_seed",
    "realized_raised_usd",
    "plan_text",
]


def normalized_text(text: str) -> str:
    return " ".join(text.split())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_firm_set(path: Path) -> list[str]:
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        firm_id = line.split("#", 1)[0].strip()
        if firm_id:
            ids.append(firm_id)
    return ids


def load_canonical_prompt() -> tuple[str, str]:
    source = BASE / "src" / "local_search_prompts.py"
    spec = importlib.util.spec_from_file_location("canonical_prompts", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import canonical prompt from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EVALUATION_CRITERIA, module.EVALUATOR_PROMPT


def add_concept(
    rows: list[dict[str, Any]],
    *,
    slot_id: str,
    group: str,
    lineage_id: str,
    parent_slot_id: str,
    source_label: str,
    source_path: Path,
    deepseek_win_rate: float,
    is_seed: bool,
    plan_text: str,
    realized_raised_usd: str = "",
) -> None:
    if not normalized_text(plan_text):
        raise ValueError(f"Empty plan text for {slot_id}")
    rows.append(
        {
            "slot_id": slot_id,
            "group": group,
            "lineage_id": lineage_id,
            "parent_slot_id": parent_slot_id,
            "text_sha256": sha256_text(normalized_text(plan_text)),
            "source_label": source_label,
            "source_path": str(source_path.relative_to(BASE)),
            "deepseek_win_rate": f"{deepseek_win_rate:.6f}",
            "is_seed": "true" if is_seed else "false",
            "realized_raised_usd": realized_raised_usd,
            "plan_text": plan_text,
        }
    )


def main() -> None:
    projects_path = BASE / "in" / "projects.csv"
    firm_set_path = BASE / "in" / "firm-set-15-evalrank.txt"
    polish_path = BASE / "out" / f"{CANONICAL_RUN}-polish-entrants.csv"
    local_candidates_path = BASE / "out" / f"{CANONICAL_RUN}-local-candidates.csv"
    baseline_path = BASE / "out" / f"{CANONICAL_RUN}-baseline15-entrants.csv"
    original_scores_path = BASE / "analysis" / "evalrank-30.csv"
    main_scores_path = BASE / "analysis" / "universe-champ-wr.json"
    baseline_scores_path = BASE / "analysis" / "baseline15-wr.json"
    canonical_prompt_path = BASE / "src" / "local_search_prompts.py"
    source_paths = [
        projects_path,
        firm_set_path,
        polish_path,
        local_candidates_path,
        baseline_path,
        original_scores_path,
        main_scores_path,
        baseline_scores_path,
        canonical_prompt_path,
    ]
    missing = [str(path) for path in source_paths if not path.exists()]
    if missing:
        raise SystemExit("Missing canonical inputs:\n" + "\n".join(missing))

    canonical_criteria, canonical_prompt = load_canonical_prompt()
    if canonical_criteria != EVALUATION_CRITERIA or canonical_prompt != EVALUATOR_PROMPT:
        raise SystemExit(
            "The frozen robustness prompt differs from src/local_search_prompts.py; "
            "review the change before preparing inputs."
        )

    seed_ids = read_firm_set(firm_set_path)
    if len(seed_ids) != 15 or len(set(seed_ids)) != 15:
        raise SystemExit(f"Expected 15 distinct seeds, found {len(seed_ids)}")
    seed_set = set(seed_ids)

    original_scores = {
        row["id"]: float(row["universe_win_rate"])
        for row in read_csv(original_scores_path)
    }
    main_scores = json.loads(main_scores_path.read_text(encoding="utf-8"))
    baseline_scores = json.loads(baseline_scores_path.read_text(encoding="utf-8"))

    concepts: list[dict[str, Any]] = []
    projects = read_csv(projects_path)
    if len(projects) != 30 or len({row["id"] for row in projects}) != 30:
        raise SystemExit("The reference-set file must contain 30 distinct ventures")
    if set(original_scores) != {row["id"] for row in projects}:
        raise SystemExit("Original-score ids do not match the reference-set ids")
    for row in projects:
        firm_id = row["id"]
        add_concept(
            concepts,
            slot_id=f"original:{firm_id}",
            group="original",
            lineage_id=firm_id,
            parent_slot_id="",
            source_label=firm_id,
            source_path=projects_path,
            deepseek_win_rate=original_scores[firm_id],
            is_seed=firm_id in seed_set,
            plan_text=row["description"],
            realized_raised_usd=row["raised"],
        )

    polish_rows = read_csv(polish_path)
    if len(polish_rows) != 30:
        raise SystemExit(f"Expected 30 polish entrant rows, found {len(polish_rows)}")
    polish_slots: dict[tuple[str, str], dict[str, str]] = {}
    for row in polish_rows:
        match = re.match(r"(g12-s\d+)-(parent|polished)", row["id"])
        if not match:
            raise SystemExit(f"Unexpected polish entrant id: {row['id']}")
        lineage, stage = match.groups()
        polish_slots[(lineage, stage)] = row
    if len(polish_slots) != 30:
        raise SystemExit("Polish entrants do not contain 15 parent-descendant pairs")

    polish_score_by_text: dict[str, float] = {}
    for row in polish_rows:
        if row["id"] in main_scores:
            polish_score_by_text[
                sha256_text(normalized_text(row["plan_text"]))
            ] = float(main_scores[row["id"]]["universe_wr"])

    for lineage in sorted({key[0] for key in polish_slots}):
        parent = polish_slots[(lineage, "parent")]
        polished = polish_slots[(lineage, "polished")]
        parent_hash = sha256_text(normalized_text(parent["plan_text"]))
        polished_hash = sha256_text(normalized_text(polished["plan_text"]))
        if parent_hash not in polish_score_by_text or polished_hash not in polish_score_by_text:
            raise SystemExit(f"Missing canonical DeepSeek score for {lineage}")
        add_concept(
            concepts,
            slot_id=f"global-final:{lineage}",
            group="global_final",
            lineage_id=lineage,
            parent_slot_id="",
            source_label=parent["id"],
            source_path=polish_path,
            deepseek_win_rate=polish_score_by_text[parent_hash],
            is_seed=False,
            plan_text=parent["plan_text"],
        )
        add_concept(
            concepts,
            slot_id=f"polished:{lineage}",
            group="polished",
            lineage_id=lineage,
            parent_slot_id=f"global-final:{lineage}",
            source_label=polished["id"],
            source_path=polish_path,
            deepseek_win_rate=polish_score_by_text[polished_hash],
            is_seed=False,
            plan_text=polished["plan_text"],
        )

    baseline_rows = read_csv(baseline_path)
    if len(baseline_rows) != 14:
        raise SystemExit(f"Expected 14 additional local-only finals, found {len(baseline_rows)}")
    baseline_ids = set()
    for row in baseline_rows:
        match = re.match(r"baseline-(.+?) \(", row["id"])
        if not match:
            raise SystemExit(f"Unexpected baseline entrant id: {row['id']}")
        seed_id = match.group(1)
        baseline_ids.add(seed_id)
        if row["id"] not in baseline_scores:
            raise SystemExit(f"Missing canonical DeepSeek baseline score for {row['id']}")
        add_concept(
            concepts,
            slot_id=f"local-only:{seed_id}",
            group="local_only",
            lineage_id=seed_id,
            parent_slot_id=f"original:{seed_id}",
            source_label=row["id"],
            source_path=baseline_path,
            deepseek_win_rate=float(baseline_scores[row["id"]]),
            is_seed=False,
            plan_text=row["plan_text"],
        )

    selected_local = [
        row for row in read_csv(local_candidates_path) if row.get("selected") == "True"
    ]
    if not selected_local or selected_local[-1].get("step") != "12":
        raise SystemExit("Could not identify the focal round-12 local final")
    focal = selected_local[-1]
    focal_label = "baseline-clockchain (main local run)"
    if focal_label not in baseline_scores:
        raise SystemExit("Missing canonical score for the focal clockchain local final")
    add_concept(
        concepts,
        slot_id="local-only:clockchain",
        group="local_only",
        lineage_id="clockchain",
        parent_slot_id="original:clockchain",
        source_label=f"local-final ({focal['candidate_id']})",
        source_path=local_candidates_path,
        deepseek_win_rate=float(baseline_scores[focal_label]),
        is_seed=False,
        plan_text=focal["plan_text"],
    )

    expected_groups = {"original": 30, "global_final": 15, "polished": 15, "local_only": 15}
    group_counts = {
        group: sum(row["group"] == group for row in concepts)
        for group in expected_groups
    }
    if group_counts != expected_groups:
        raise SystemExit(f"Unexpected group counts: {group_counts}")
    if baseline_ids | {"clockchain"} != seed_set:
        raise SystemExit("The local-only finals do not correspond exactly to the 15 seeds")

    slots_by_hash: dict[str, list[str]] = {}
    for row in concepts:
        slots_by_hash.setdefault(row["text_sha256"], []).append(row["slot_id"])
    duplicates = {
        text_hash: slots
        for text_hash, slots in slots_by_hash.items()
        if len(slots) > 1
    }
    generated = [row for row in concepts if row["group"] != "original"]
    unique_generated = {row["text_sha256"] for row in generated}
    if len(concepts) != 75 or len(unique_generated) != 44:
        raise SystemExit(
            f"Expected 75 slots and 44 distinct generated texts; found "
            f"{len(concepts)} and {len(unique_generated)}"
        )
    if len(duplicates) != 1:
        raise SystemExit(f"Expected one duplicate text across slots, found {len(duplicates)}")

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    concept_path = INPUT_DIR / "concepts.csv"
    with concept_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONCEPT_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(concepts)

    prompt_payload = json.dumps(
        {"criteria": EVALUATION_CRITERIA, "template": EVALUATOR_PROMPT},
        ensure_ascii=False,
        sort_keys=True,
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "canonical_run": CANONICAL_RUN,
        "sample": {
            "slot_count": len(concepts),
            "unique_text_count": len(slots_by_hash),
            "group_counts": group_counts,
            "unique_generated_texts": len(unique_generated),
            "duplicate_texts": duplicates,
            "expected_paid_calls": {
                "originals_double_round_robin": 30 * 29,
                "generated_vs_originals_both_orders": len(unique_generated) * 30 * 2,
                "total": 30 * 29 + len(unique_generated) * 30 * 2,
            },
        },
        "prompt_sha256": sha256_text(prompt_payload),
        "concepts_csv_sha256": sha256_file(concept_path),
        "sources": {
            str(path.relative_to(BASE)): sha256_file(path)
            for path in source_paths
        },
    }
    manifest_path = INPUT_DIR / "sample-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {concept_path}")
    print(f"Wrote {manifest_path}")
    print(json.dumps(manifest["sample"], indent=2))


if __name__ == "__main__":
    main()
