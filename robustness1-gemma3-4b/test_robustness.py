#!/usr/bin/env python3
"""Deterministic tests for the robustness-check machinery."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import prepare_inputs
from prompt import render_prompt
from run_robustness import build_tasks, parse_content, select_pilot


HERE = Path(__file__).resolve().parent


class RobustnessTests(unittest.TestCase):
    def test_prompt_contains_both_plans_and_json_schema(self) -> None:
        prompt = render_prompt("PLAN ALPHA", "PLAN BETA")
        self.assertIn("PLAN ALPHA", prompt)
        self.assertIn("PLAN BETA", prompt)
        self.assertIn('"winner": "A" or "B"', prompt)

    def test_parser_accepts_exact_and_fenced_json(self) -> None:
        exact = '{"analysis":"clear","winner":"A"}'
        fenced = '```json\n{"analysis":"clear","winner":"B"}\n```'
        self.assertEqual(parse_content(exact), ("A", "clear", "exact"))
        self.assertEqual(parse_content(fenced), ("B", "clear", "markdown_fence"))

    def test_parser_recovers_only_an_explicit_winner_field(self) -> None:
        malformed = '''```json
{"analysis": {"Project A:", "Pros:"}, "winner": "B"}
```'''
        winner, _analysis, mode = parse_content(malformed)
        self.assertEqual(winner, "B")
        self.assertEqual(mode, "explicit_winner_recovered_from_malformed_json")

    def test_parser_normalizes_project_prefix(self) -> None:
        content = '{"analysis":"clear","winner":"Project A"}'
        self.assertEqual(parse_content(content), ("A", "clear", "normalized_project_winner"))

    def test_parser_rejects_missing_winner(self) -> None:
        with self.assertRaises(ValueError):
            parse_content('{"analysis":"clear"}')

    def test_frozen_sample_and_task_arithmetic(self) -> None:
        prepare_inputs.main()
        concepts_path = HERE / "inputs" / "concepts.csv"
        with concepts_path.open(newline="", encoding="utf-8") as handle:
            concepts = list(csv.DictReader(handle))
        tasks, texts = build_tasks(concepts, "google/gemma-3-4b-it", 0.5, 42)
        self.assertEqual(len(concepts), 75)
        self.assertEqual(len({row["text_sha256"] for row in concepts}), 74)
        self.assertEqual(len(texts), 74)
        self.assertEqual(len(tasks), 3510)
        self.assertEqual(sum(row["category"] == "original_original" for row in tasks), 870)
        self.assertEqual(sum(row["category"] == "generated_reference" for row in tasks), 2640)
        pilot = select_pilot(tasks, 12)
        self.assertEqual(len(pilot), 12)
        self.assertGreaterEqual(len({row["bucket"] for row in pilot}), 4)


if __name__ == "__main__":
    unittest.main()
