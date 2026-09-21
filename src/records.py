"""Normalized audit-record writers shared by local and global search."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


CANDIDATE_FIELDNAMES = [
    "mode",
    "step",
    "slot",
    "candidate_id",
    "parent_a",
    "parent_b",
    "origin",
    "transformation",
    "requested_change",
    "plan_text",
    "generated_plan",
    "rank",
    "win_rate",
    "quality_percentile",
    "novelty",
    "novelty_rank",
    "novelty_percentile",
    "combined_score",
    "selection_rank",
    "selected",
    "fidelity_passed",
    "fidelity_action",
    "unsupported_claims",
    "fidelity_analysis",
    "evidence_notes",
    "prompt",
    "root_lineage",
]

MATCH_FIELDNAMES = [
    "mode",
    "step",
    "plan_a_slot",
    "plan_a_id",
    "plan_b_slot",
    "plan_b_id",
    "winner_id",
    "winner_label",
    "analysis",
]

AUDIT_FIELDNAMES = [
    "mode",
    "step",
    "candidate_id",
    "operator",
    "parent_a",
    "parent_b",
    "model_passed",
    "decision_basis",
    "model_unsupported_claims",
    "passed",
    "action",
    "unsupported_claims",
    "analysis",
    "generated_plan",
    "evaluated_plan",
]


def initialize_csv(path: str, fieldnames: list[str]) -> None:
    """Create a fresh normalized record with a stable header."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL).writeheader()


def append_rows(path: str, fieldnames: list[str], rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writerows(rows)


def claims_json(claims: list[str] | None) -> str:
    return json.dumps(claims or [], ensure_ascii=False)
