#!/usr/bin/env python3
"""Score every unique plan with the locked component-embedding construct.

This is a local, deterministic analysis stage: it makes no OpenRouter calls.
It applies the exact encoder and seed archive used by revised global search to
all local, global, and optional comparison-run candidates so R can place them
on one common novelty scale.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import config
from semantic_novelty import COMPONENTS, SemanticNovelty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute locked semantic novelty for normalized candidate records."
    )
    parser.add_argument("--input-file", required=True, help="Source projects CSV.")
    parser.add_argument("--firm-set", required=True, help="Locked seed IDs, one per line.")
    parser.add_argument(
        "--candidate-csv",
        action="append",
        required=True,
        help="Normalized candidate CSV; repeat for additional runs or modes.",
    )
    parser.add_argument("--output", required=True, help="Destination semantic-score CSV.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the deterministic hash encoder for integration tests only.",
    )
    return parser.parse_args()


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").split())


def read_firm_set(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        ids = [line.split("#", 1)[0].strip() for line in handle]
    return [firm_id for firm_id in ids if firm_id]


def read_seed_plans(path: str, firm_ids: list[str]) -> list[str]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        by_id = {row["id"]: row["description"] for row in csv.DictReader(handle)}
    missing = [firm_id for firm_id in firm_ids if firm_id not in by_id]
    if missing:
        raise SystemExit(f"Firm IDs missing from {path}: {', '.join(missing)}")
    return [by_id[firm_id] for firm_id in firm_ids]


def collect_unique_plans(paths: list[str]) -> list[dict[str, str]]:
    by_key: dict[str, dict[str, str]] = {}
    for source_path in paths:
        path = Path(source_path)
        if not path.exists():
            raise SystemExit(f"Candidate CSV not found: {path}")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if "plan_text" not in (reader.fieldnames or []):
                raise SystemExit(f"Candidate CSV lacks plan_text: {path}")
            for row in reader:
                plan_text = row.get("plan_text", "")
                text_key = normalize_text(plan_text)
                if text_key and text_key not in by_key:
                    by_key[text_key] = {
                        "text_key": text_key,
                        "plan_text": plan_text,
                        "first_source_file": path.name,
                    }
    return list(by_key.values())


def output_fieldnames() -> list[str]:
    fields = [
        "text_key",
        "plan_text",
        "first_source_file",
        "semantic_novelty",
        "nearest_seed_id",
    ]
    for component in COMPONENTS:
        fields.extend(
            [
                f"{component}_distance_to_nearest_seed",
                f"{component}_nearest_seed_id",
                f"{component}_nearest_seed_distance",
            ]
        )
    return fields


def main() -> None:
    args = parse_args()
    firm_ids = read_firm_set(args.firm_set)
    try:
        config.validate_firm_set(firm_ids)
    except ValueError as exc:
        raise SystemExit(str(exc))
    seed_plans = read_seed_plans(args.input_file, firm_ids)
    plans = collect_unique_plans(args.candidate_csv)
    if not plans:
        raise SystemExit("No nonempty candidate plans were found.")

    use_mock = args.mock or config.MOCK_LLM
    novelty = SemanticNovelty(
        firm_ids,
        seed_plans,
        model_name=config.NOVELTY_EMBEDDING_MODEL,
        model_revision=config.NOVELTY_EMBEDDING_REVISION,
        device=config.NOVELTY_EMBEDDING_DEVICE,
        mock=use_mock,
    )
    details = novelty.novelty_details([row["plan_text"] for row in plans])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=output_fieldnames(),
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for row, detail in zip(plans, details):
            output_row: dict[str, str] = {
                **row,
                "semantic_novelty": f"{float(detail['novelty']):.8f}",
                "nearest_seed_id": str(detail["nearest_seed_id"]),
            }
            for component in COMPONENTS:
                output_row[f"{component}_distance_to_nearest_seed"] = (
                    f"{float(detail['component_distances'][component]):.8f}"
                )
                output_row[f"{component}_nearest_seed_id"] = str(
                    detail["component_nearest_seed_ids"][component]
                )
                output_row[f"{component}_nearest_seed_distance"] = (
                    f"{float(detail['component_nearest_distances'][component]):.8f}"
                )
            writer.writerow(output_row)

    backend = "mock-hash" if use_mock else "sentence-transformers"
    print(f"Semantic scores: {output}")
    print(f"Unique plans: {len(plans)}")
    print(f"Encoder: {config.NOVELTY_EMBEDDING_MODEL}@{config.NOVELTY_EMBEDDING_REVISION}")
    print(f"Backend: {backend}")


if __name__ == "__main__":
    main()
