"""Deterministic safeguards for parsing, ranking, design, and fallbacks."""

import os
from pathlib import Path
import sys
import unittest
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from agents_common import Evaluator, FidelityAuditor, _chat, rank_from_wins
from global_search_agents import (
    Mutator,
    _make_mutant_id,
    _make_offspring_id,
    filter_crossover_audit,
    validate_component_map,
)
from global_search_prompts import CROSSOVER_PROMPT, MUTATOR_PROMPT
from global_search import (
    _build_fieldnames as build_global_fieldnames,
    _unavailable_audit_slots,
    pick_recombination_partner,
    select_elites_with_lineage_cap,
)
from local_search_agents import SpecializedGenerator
from local_search_prompts import (
    FACTUAL_FIDELITY_RULES,
    FIDELITY_AUDITOR_PROMPT,
    SPECIALIZED_GENERATOR_PROMPT,
)
from local_search import _build_fieldnames, _build_match_pairs, load_seed_plan
from semantic_novelty import (
    COMPONENTS,
    SemanticNovelty,
    combine_quality_and_novelty,
    component_texts,
)


class _ReverseRng:
    def shuffle(self, seq):
        seq.reverse()


class _TopicEncoder:
    """Deterministic semantic stand-in for unit tests; never downloads a model."""

    def encode(self, texts, **kwargs):
        topics = ("apple", "robot", "music")
        return [
            [float(text.lower().split().count(topic)) for topic in topics]
            for text in texts
        ]


def _component_plan(topic: str) -> str:
    return "\n".join(
        f"# {component.replace('_', ' ').title()}\n{topic} {component}"
        for component in COMPONENTS
    )


@contextmanager
def patched_config(**overrides):
    original = {name: getattr(config, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(config, name, value)
        config.sync_step_aliases()
        yield
    finally:
        for name, value in original.items():
            setattr(config, name, value)
        config.sync_step_aliases()


class RankFromWinsTests(unittest.TestCase):
    def test_stable_ranks_without_rng(self):
        self.assertEqual(rank_from_wins([5, 5, 2, 2]), [1, 2, 3, 4])

    def test_randomized_ties_shuffle_within_equal_groups_only(self):
        ranks = rank_from_wins([5, 5, 2, 2], tie_break_rng=_ReverseRng())
        self.assertEqual(ranks, [2, 1, 4, 3])


class ConfigValidationTests(unittest.TestCase):
    def test_local_config_rejects_zero_steps(self):
        with patched_config(NUM_STEPS=0):
            with self.assertRaisesRegex(ValueError, "NUM_STEPS"):
                config.validate_local_search_config()

    def test_global_config_requires_selection_pool_of_at_least_two(self):
        with patched_config(
            NUM_STEPS=1,
            POP_SIZE=15,
            POP_ELITE=5,
            POP_MUTANT=5,
            POP_OFFSPRING=5,
            SELECTION_POOL_FRAC=0.05,
        ):
            with self.assertRaisesRegex(ValueError, "selection pool"):
                config.validate_global_search_config()

    def test_global_config_accepts_locked_smoke_test_settings(self):
        with patched_config(
            NUM_STEPS=1,
            POP_SIZE=15,
            POP_ELITE=5,
            POP_MUTANT=5,
            POP_OFFSPRING=5,
            SELECTION_POOL_FRAC=2 / 3,
        ):
            config.validate_global_search_config()
            self.assertEqual(config.get_selection_pool_size(), 10)

    def test_global_config_rejects_nonlocked_population(self):
        with patched_config(
            NUM_STEPS=1,
            POP_SIZE=4,
            POP_ELITE=1,
            POP_MUTANT=1,
            POP_OFFSPRING=2,
            SELECTION_POOL_FRAC=0.75,
        ):
            with self.assertRaisesRegex(ValueError, "locks global search"):
                config.validate_global_search_config()

    def test_local_config_accepts_disabled_stuck_threshold(self):
        with patched_config(NUM_STEPS=2, STUCK_THRESHOLD=None):
            config.validate_local_search_config()

    def test_local_config_rejects_zero_timeout(self):
        with patched_config(API_CALL_TIMEOUT=0):
            with self.assertRaisesRegex(ValueError, "API_CALL_TIMEOUT"):
                config.validate_local_search_config()

    def test_global_config_rejects_zero_batch_size(self):
        with patched_config(MATCH_BATCH_SIZE=0):
            with self.assertRaisesRegex(ValueError, "MATCH_BATCH_SIZE"):
                config.validate_global_search_config()

    def test_firm_set_validation_is_structural(self):
        valid = [f"firm-{i}" for i in range(14)] + [config.SEED_PROJECT_ID]
        config.validate_firm_set(valid)

        with self.assertRaisesRegex(ValueError, "exactly 15"):
            config.validate_firm_set(valid[:10])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            config.validate_firm_set(valid[:-1] + [valid[0]])
        with self.assertRaisesRegex(ValueError, "local-search seed"):
            config.validate_firm_set([f"firm-{i}" for i in range(15)])

    def test_global_config_rejects_invalid_lineage_cap(self):
        with patched_config(ELITE_LINEAGE_CAP=0):
            with self.assertRaisesRegex(ValueError, "ELITE_LINEAGE_CAP"):
                config.validate_global_search_config()
        with patched_config(ELITE_LINEAGE_CAP=6):
            with self.assertRaisesRegex(ValueError, "ELITE_LINEAGE_CAP"):
                config.validate_global_search_config()


class ChildIdTests(unittest.TestCase):
    def test_mutant_ids_are_unique_per_slot(self):
        self.assertNotEqual(
            _make_mutant_id("seed", 1, 1),
            _make_mutant_id("seed", 1, 2),
        )
        self.assertEqual(_make_mutant_id("seed", 1, 2), "seed.M1.2")

    def test_offspring_ids_are_unique_per_slot(self):
        self.assertNotEqual(
            _make_offspring_id("a", "b", 2, 1),
            _make_offspring_id("a", "b", 2, 2),
        )
        self.assertEqual(_make_offspring_id("a", "b", 2, 2), "(aXb).G2.2")


class SeedLoadingTests(unittest.TestCase):
    def test_load_seed_plan_honors_requested_project_id(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("id,description\ntoolkaiser,plan A text\ng12-s03,plan B text\n")
            path = f.name
        try:
            self.assertEqual(load_seed_plan(path, "g12-s03"), "plan B text")
            self.assertEqual(load_seed_plan(path, "toolkaiser"), "plan A text")
        finally:
            os.unlink(path)


class LocalCsvShapeTests(unittest.TestCase):
    def test_local_csv_shape_tracks_active_num_ideas(self):
        fieldnames = _build_fieldnames(2)
        match_pairs = _build_match_pairs(3)

        self.assertEqual(fieldnames[0], "step")
        self.assertNotIn("time_step", fieldnames)
        self.assertIn("plan_2", fieldnames)
        self.assertNotIn("plan_3", fieldnames)
        self.assertEqual(len(match_pairs), 6)
        self.assertIn("analysis_2_vs_1", fieldnames)
        self.assertNotIn("analysis_3_vs_0", fieldnames)

    def test_global_csv_shape_uses_shared_step_field(self):
        fieldnames = build_global_fieldnames()

        self.assertEqual(fieldnames[0], "step")
        self.assertNotIn("generation", fieldnames)
        self.assertIn("plan_id_0", fieldnames)
        self.assertIn("novelty_0", fieldnames)
        self.assertIn("selection_rank_14", fieldnames)


class SemanticNoveltyTests(unittest.TestCase):
    def test_component_projection_maps_legacy_source_headings(self):
        plan = """# One-Sentence Summary
Distinct idea.
# Problem
Pain point.
# Solution / Product Overview
Product.
# Key Features & Capabilities
Feature.
# Target Users & Use Cases
Buyer.
# Roadmap & Delivery Plan
Launch plan.
# Why Crowdfunding / Use of Funds
Funding model.
"""
        projected = component_texts(plan)

        self.assertEqual(tuple(projected), COMPONENTS)
        self.assertIn("Pain point", projected["problem"])
        self.assertIn("Buyer", projected["customer"])
        self.assertIn("Product", projected["solution"])
        self.assertIn("Launch plan", projected["delivery_model"])
        self.assertIn("Funding model", projected["revenue_logic"])
        self.assertIn("Distinct idea", projected["distinctiveness"])

    def test_novelty_is_distance_from_nearest_fixed_seed(self):
        apple = _component_plan("apple")
        robot = _component_plan("robot")
        measure = SemanticNovelty(
            ["apple-seed"],
            [apple],
            model_name="test",
            model_revision="test",
            encoder=_TopicEncoder(),
        )

        scores = measure.novelty_scores([apple, robot])

        self.assertAlmostEqual(scores[0], 0.0)
        self.assertAlmostEqual(scores[1], 1.0)

    def test_novelty_details_distinguish_aggregate_and_component_matches(self):
        apple = _component_plan("apple")
        robot = _component_plan("robot")
        mixed_lines = []
        for index, component in enumerate(COMPONENTS):
            topic = "apple" if index < 3 else "robot"
            mixed_lines.append(
                f"# {component.replace('_', ' ').title()}\n{topic} {component}"
            )
        mixed = "\n".join(mixed_lines)
        measure = SemanticNovelty(
            ["apple-seed", "robot-seed"],
            [apple, robot],
            model_name="test",
            model_revision="test",
            encoder=_TopicEncoder(),
        )

        detail = measure.novelty_details([mixed])[0]

        self.assertAlmostEqual(detail["novelty"], 0.5)
        self.assertEqual(detail["nearest_seed_id"], "apple-seed")
        self.assertEqual(detail["component_nearest_seed_ids"]["problem"], "apple-seed")
        self.assertEqual(
            detail["component_nearest_seed_ids"]["distinctiveness"],
            "robot-seed",
        )

    def test_quality_and_novelty_percentiles_receive_equal_weight(self):
        metrics = combine_quality_and_novelty(
            quality_ranks=[1, 2, 3],
            novelty_scores=[0.0, 2.0, 1.0],
        )

        self.assertEqual(metrics.quality_percentiles, [1.0, 0.5, 0.0])
        self.assertEqual(metrics.novelty_percentiles, [0.0, 1.0, 0.5])
        self.assertEqual(metrics.combined_scores, [0.5, 0.75, 0.25])
        self.assertEqual(metrics.selection_ranks, [2, 1, 3])

    def test_quality_only_weights_reduce_selection_to_quality_rank(self):
        metrics = combine_quality_and_novelty(
            quality_ranks=[2, 1, 3],
            novelty_scores=[9.0, 0.0, 5.0],
            quality_weight=1.0,
            novelty_weight=0.0,
        )

        self.assertEqual(metrics.combined_scores, [0.5, 1.0, 0.0])
        self.assertEqual(metrics.selection_ranks, [2, 1, 3])

    def test_recombination_partner_prefers_best_ranked_new_lineage(self):
        eligible = [
            {"candidate_id": "a-child", "plan": "plan alpha", "roots": frozenset({"a"})},
            {"candidate_id": "b-child", "plan": "plan beta", "roots": frozenset({"b"})},
            {"candidate_id": "c-seed", "plan": "plan gamma", "roots": frozenset({"c"})},
        ]

        second = pick_recombination_partner(
            "a-parent", "plan of a", frozenset({"a"}), eligible, set()
        )

        self.assertEqual(second["candidate_id"], "b-child")

    def test_recombination_partner_skips_used_pairs_then_allows_repeats(self):
        eligible = [
            {"candidate_id": "b-child", "plan": "plan beta", "roots": frozenset({"b"})},
            {"candidate_id": "c-seed", "plan": "plan gamma", "roots": frozenset({"c"})},
        ]
        used = {("a-parent", "b-child")}

        second = pick_recombination_partner(
            "a-parent", "plan of a", frozenset({"a"}), eligible, used
        )
        self.assertEqual(second["candidate_id"], "c-seed")

        used.add(("a-parent", "c-seed"))
        third = pick_recombination_partner(
            "a-parent", "plan of a", frozenset({"a"}), eligible, used
        )
        self.assertEqual(third["candidate_id"], "b-child")

    def test_recombination_partner_can_reintroduce_archive_seed(self):
        eligible = [
            {"candidate_id": "a-elite", "plan": "plan alpha two", "roots": frozenset({"a"})},
            {"candidate_id": "seed-b", "plan": "plan beta", "roots": frozenset({"b"})},
        ]

        second = pick_recombination_partner(
            "a-parent", "plan of a", frozenset({"a"}), eligible, set()
        )

        self.assertEqual(second["candidate_id"], "seed-b")

    def test_elite_lineage_cap_keeps_multiple_lineages_alive(self):
        sorted_indices = [0, 1, 2, 3, 4, 5]
        root_sets = [
            frozenset({"a"}),
            frozenset({"a"}),
            frozenset({"a"}),
            frozenset({"b"}),
            frozenset({"a"}),
            frozenset({"c"}),
        ]

        elites = select_elites_with_lineage_cap(sorted_indices, root_sets, 5, 2)

        # slots 0,1 fill lineage a's cap; slot 2 and 4 are displaced in the
        # first pass, admitting b and c, then the best displaced tops up.
        self.assertEqual(elites, [0, 1, 3, 5, 2])

    def test_elite_lineage_cap_tops_up_when_one_lineage_remains(self):
        sorted_indices = [0, 1, 2]
        root_sets = [frozenset({"a"})] * 3

        elites = select_elites_with_lineage_cap(sorted_indices, root_sets, 3, 2)

        self.assertEqual(elites, [0, 1, 2])


class GlobalOperatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_mutation_component_is_selected_by_algorithm_and_recorded(self):
        parent = "# Problem\nA source plan that is comfortably longer than fifty characters for testing."
        child = parent + "\nA more sharply defined pain point for the focal user."
        mutator = Mutator()
        mutator.auditor.audit = AsyncMock(
            return_value={"passed": True, "unsupported_claims": [], "analysis": "grounded"}
        )
        response = __import__("json").dumps({"component": "solution", "plan": child})
        with patch("global_search_agents._chat", new=AsyncMock(return_value=response)) as chat:
            result = await mutator.mutate(parent, "seed", "problem", 1, 1)

        self.assertEqual(result["dimension"], "problem")
        prompt = chat.await_args.args[1]
        self.assertIn("problem: the unmet need", prompt)
        self.assertNotIn("Pick ONE", prompt)

    def test_crossover_prompt_requires_component_level_recombination(self):
        self.assertIn("RECOMBINING the two parents at the level of", CROSSOVER_PROMPT)
        self.assertIn("at least two components from each parent", CROSSOVER_PROMPT)
        self.assertIn("brand-new venture", CROSSOVER_PROMPT)
        self.assertIn("do NOT carry claims of existing traction", CROSSOVER_PROMPT)
        self.assertIn("component_map", CROSSOVER_PROMPT)
        self.assertIn("must not be a restatement", CROSSOVER_PROMPT)

    def test_crossover_audit_filter_distinguishes_concept_from_evidence(self):
        candidate = (
            "The venture plans an app-free bedside hub. "
            "The product would combine calming audio with sunrise lighting. "
            "The team has already shipped 5,000 units to early customers."
        )
        concept_flags = {
            "passed": False,
            "unsupported_claims": [
                "The venture plans an app-free bedside hub.",
                "The product would combine calming audio with sunrise lighting.",
            ],
            "analysis": "flags concept statements",
        }
        out = filter_crossover_audit(concept_flags, candidate)
        self.assertTrue(out["passed"])
        self.assertEqual(out["decision_basis"], "crossover_concept_filter")
        self.assertFalse(out["model_passed"])

        evidence_flags = {
            "passed": False,
            "unsupported_claims": [
                "The team has already shipped 5,000 units to early customers.",
            ],
            "analysis": "flags real evidence",
        }
        out = filter_crossover_audit(evidence_flags, candidate)
        self.assertFalse(out["passed"])
        self.assertEqual(out["decision_basis"], "crossover_marker_confirmed")

        passing = {"passed": True, "unsupported_claims": [], "analysis": "ok"}
        self.assertTrue(filter_crossover_audit(passing, candidate)["passed"])

    def test_component_map_validation_enforces_two_from_each_parent(self):
        balanced = {
            "problem": "A", "customer": "a", "solution": "B",
            "delivery_model": "b", "revenue_logic": "A", "distinctiveness": "B",
        }
        normalized = validate_component_map(balanced)
        self.assertEqual(normalized["customer"], "A")
        self.assertEqual(normalized["delivery_model"], "B")

        lopsided = dict.fromkeys(
            ("problem", "customer", "solution", "delivery_model",
             "revenue_logic"), "A") | {"distinctiveness": "B"}
        self.assertIsNone(validate_component_map(lopsided))
        self.assertIsNone(validate_component_map({"problem": "A"}))
        self.assertIsNone(validate_component_map("problem from A, rest from B"))

    def test_mutator_prompt_uses_algorithm_selected_component(self):
        rendered = MUTATOR_PROMPT.format(
            plan="source",
            component="customer",
            component_guidance="the focal user",
            fidelity_rules=FACTUAL_FIDELITY_RULES,
        )
        self.assertIn('"component": "customer"', rendered)


class ChatRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_sets_max_tokens_cap_and_timeout(self):
        response = AsyncMock()
        response.choices = [type("Choice", (), {"message": type("Msg", (), {"content": "ok"})()})()]
        response.usage = None

        async def passthrough(awaitable, timeout):
            return await awaitable

        with patch(
            "agents_common.client.chat.completions.create",
            new=AsyncMock(return_value=response),
        ) as create_mock:
            with patch(
                "agents_common.asyncio.wait_for",
                new=AsyncMock(side_effect=passthrough),
            ) as wait_for_mock:
                text = await _chat(
                    "deepseek/deepseek-v3.2",
                    "hello",
                    role="Test",
                    context="smoke",
                    temperature=0.5,
                )

        self.assertEqual(text, "ok")
        self.assertEqual(create_mock.await_args.kwargs["max_tokens"], config.MAX_RESPONSE_TOKENS)
        self.assertEqual(create_mock.await_args.kwargs["temperature"], 0.5)
        self.assertEqual(
            create_mock.await_args.kwargs["extra_body"],
            {"reasoning": {"effort": "none"}},
        )
        self.assertEqual(wait_for_mock.await_args.kwargs["timeout"], config.API_CALL_TIMEOUT)


class EvaluatorRobustnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_duplicate_is_tied_without_live_call(self):
        evaluator = Evaluator()
        with patch("agents_common._chat", new=AsyncMock()) as chat_mock:
            winner, analysis = await evaluator._match("same  plan", " same plan ")
        self.assertEqual((winner, analysis), ("TIE", "Exact same normalized text; skipped LLM call."))
        chat_mock.assert_not_awaited()

    async def test_evaluate_batches_matches(self):
        evaluator = Evaluator()

        async def fake_match(plan_a, plan_b, label_a="", label_b="", log_extra=""):
            return ("A" if plan_a < plan_b else "B", f"{plan_a}>{plan_b}")

        evaluator._match = AsyncMock(side_effect=fake_match)

        with patched_config(MATCH_BATCH_SIZE=2):
            ranks, win_rates, match_details = await evaluator.evaluate(["a", "b", "c"])

        self.assertEqual(evaluator._match.await_count, 6)
        self.assertEqual(ranks, [1, 2, 3])
        self.assertEqual(win_rates, [1.0, 0.5, 0.0])
        self.assertEqual(match_details[(1, 0)][0], "plan_0")
        self.assertEqual(match_details[(2, 1)][0], "plan_1")

    async def test_match_defaults_after_chat_failure(self):
        evaluator = Evaluator()

        with patch(
            "agents_common._chat",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ) as chat_mock:
            winner, analysis = await evaluator._match("plan a", "plan b", "incumbent", "alt 1")

        self.assertEqual(chat_mock.await_count, 1)
        self.assertEqual((winner, analysis), ("TIE", ""))

    async def test_tie_fallback_splits_wins_evenly(self):
        evaluator = Evaluator()
        evaluator._match = AsyncMock(side_effect=[("TIE", ""), ("TIE", "")])

        ranks, win_rates, match_details = await evaluator.evaluate(["a", "b"])

        self.assertEqual(ranks, [1, 2])
        self.assertEqual(win_rates, [0.5, 0.5])
        self.assertEqual(match_details[(0, 1)][0], "tie")
        self.assertEqual(match_details[(1, 0)][0], "tie")


class FactualFidelityTests(unittest.IsolatedAsyncioTestCase):
    def test_global_search_detects_unavailable_audit_slots(self):
        mutants = [
            {"audit": {"passed": True}},
            {"audit": {"passed": False, "unavailable": True}},
        ]
        offspring = [{"audit": {"passed": False, "unavailable": True}}]

        self.assertEqual(
            _unavailable_audit_slots(mutants, offspring),
            ["mutation 2", "crossover 1"],
        )

    def test_specialized_prompt_contains_evidence_rules(self):
        prompt = SPECIALIZED_GENERATOR_PROMPT.format(
            plan="source",
            idea="proposal",
            fidelity_rules=FACTUAL_FIDELITY_RULES,
        )
        self.assertIn("Do not invent or imply existing traction", prompt)
        self.assertIn('"evidence_notes"', prompt)

    def test_auditor_allows_specific_future_proposals(self):
        self.assertIn("Specificity does not turn a future proposal", FIDELITY_AUDITOR_PROMPT)
        self.assertIn("we plan to test with 50 participants", FIDELITY_AUDITOR_PROMPT)
        self.assertIn("the candidate must pass", FIDELITY_AUDITOR_PROMPT)

    def test_prospective_safeguard_accepts_explicit_future_claims(self):
        candidate = (
            "We plan to conduct a usability test with 50 participants. "
            "A hypothetical partner would pilot the feature next year."
        )
        self.assertTrue(
            FidelityAuditor._is_clearly_prospective_claim(
                candidate,
                "We plan to conduct a usability test with 50 participants.",
            )
        )
        self.assertTrue(
            FidelityAuditor._is_clearly_prospective_claim(
                candidate,
                "A hypothetical partner would pilot the feature next year.",
            )
        )

    def test_prospective_safeguard_does_not_accept_mixed_existing_claims(self):
        candidate = "We have already tested with 50 participants and plan to expand the test."
        self.assertFalse(
            FidelityAuditor._is_clearly_prospective_claim(candidate, candidate)
        )
        customer_claim = "We will launch the feature for our current customers."
        self.assertFalse(
            FidelityAuditor._is_clearly_prospective_claim(customer_claim, customer_claim)
        )

    async def test_model_rejection_of_only_proposals_is_overridden(self):
        rejected = {
            "passed": False,
            "unsupported_claims": ["We plan to test with 50 participants."],
            "analysis": "The plan is absent from the source.",
        }
        with patch(
            "agents_common._chat",
            new=AsyncMock(return_value=__import__("json").dumps(rejected)),
        ):
            result = await FidelityAuditor().audit(
                [("PARENT", "Original source plan.")],
                "We plan to test with 50 participants.",
                context="test",
            )

        self.assertTrue(result["passed"])
        self.assertEqual(result["decision_basis"], "prospective_framing_safeguard")
        self.assertFalse(result["model_passed"])
        self.assertEqual(result["unsupported_claims"], [])

    async def test_failed_audit_falls_back_to_parent(self):
        parent = "Original source plan with no customers or launch claims."
        generated = {
            "plan": "We already serve 10,000 customers through a signed university partnership.",
            "evidence_notes": "",
        }
        audit = {
            "passed": False,
            "unsupported_claims": ["10,000 customers", "signed university partnership"],
            "analysis": "Neither claim appears in the source.",
        }
        generator = SpecializedGenerator()
        with patch(
            "local_search_agents._chat",
            new=AsyncMock(return_value=__import__("json").dumps(generated)),
        ):
            with patch(
                "agents_common._chat",
                new=AsyncMock(return_value=__import__("json").dumps(audit)),
            ):
                record = await generator.generate(parent, "Clarify the evidence plan", alt_num=1)

        self.assertEqual(record["plan"], parent)
        self.assertEqual(record["action"], "fallback_parent")
        self.assertFalse(record["audit"]["passed"])

    async def test_unavailable_audit_is_distinct_from_substantive_rejection(self):
        auditor = FidelityAuditor()
        with patch(
            "agents_common._chat",
            new=AsyncMock(side_effect=RuntimeError("provider unavailable")),
        ):
            result = await auditor.audit(
                [("PARENT", "Original source plan.")],
                "Candidate plan.",
                context="test",
            )

        self.assertFalse(result["passed"])
        self.assertTrue(result["unavailable"])
        self.assertIn("Audit unavailable", result["unsupported_claims"][0])


if __name__ == "__main__":
    unittest.main()
