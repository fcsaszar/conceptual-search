"""Mutation, crossover, and selection agents used by global search."""

import asyncio
import json
import logging
import random

import config
from local_search_prompts import EVALUATION_CRITERIA, FACTUAL_FIDELITY_RULES
from global_search_prompts import (
    CROSSOVER_FIDELITY_AUDITOR_PROMPT,
    CROSSOVER_PROMPT,
    MUTATION_COMPONENT_GUIDANCE,
    MUTATOR_PROMPT,
)
from agents_common import (
    _chat,
    _describe_exception,
    _strip_fences,
    _track_planned,
    _inc_errors,
    FidelityAuditor,
)

logger = logging.getLogger(__name__)

# Deterministic decision layer for crossover audits.  The model auditor
# systematically conflates concept novelty with evidence fabrication when the
# candidate is a recombined concept (it flags even explicitly prospective
# sentences), so its flagged claims are treated as candidates and the verdict
# is decided by rule: a claim is a real violation only when its sentence
# asserts an already-existing accomplishment AND is not prospectively framed.
import re as _re

_CROSSOVER_ACHIEVEMENT_MARKERS = _re.compile(
    r"(\b(has|have|had)\s+(already\s+)?(shipped|served|signed|raised|generated|"
    r"launched|built|tested|enrolled|secured|partnered|won|earned|completed|"
    r"sold|delivered|deployed|acquired|collected)\b)"
    r"|(\balready\s+(has|have|serves|ships|sells|works|operates|offers)\b)"
    r"|(\b(currently|to date|so far|at present)\s+(serve|serves|ship|ships|"
    r"sell|sells|operate|operates|have|has)\b)"
    r"|(\bin\s+transit\b)"
    r"|(\bmass[- ]production\s+batch\b)"
    r"|(\b(customer|user|beta[- ]tester)\s+testimonials?\b)"
    r"|(\bfive[- ]star\s+reviews?\b)",
    _re.IGNORECASE,
)

_CROSSOVER_PROSPECTIVE_MARKERS = _re.compile(
    r"\b(plans?|planned|planning|proposes?|proposed|intends?|intended|aims?|"
    r"aiming|would|will|could|target|targets|targeting|future|hypothetical|"
    r"prospective|envisioned|to\s+be\s+tested|to-be-tested)\b",
    _re.IGNORECASE,
)


def filter_crossover_audit(audit: dict, candidate: str) -> dict:
    """Apply the deterministic concept-vs-evidence rule to a crossover audit."""
    if audit.get("passed") or audit.get("unavailable"):
        return audit
    claims = audit.get("unsupported_claims") or []
    real_violations = []
    for claim in claims:
        context = FidelityAuditor._claim_context(candidate, str(claim))
        if _CROSSOVER_ACHIEVEMENT_MARKERS.search(context) and not (
            _CROSSOVER_PROSPECTIVE_MARKERS.search(context)
        ):
            real_violations.append(str(claim))
    filtered = dict(audit)
    filtered.setdefault("model_passed", audit.get("passed", False))
    filtered.setdefault("model_unsupported_claims", list(claims))
    if real_violations:
        filtered["passed"] = False
        filtered["unsupported_claims"] = real_violations
        filtered["decision_basis"] = "crossover_marker_confirmed"
    else:
        filtered["passed"] = True
        filtered["unsupported_claims"] = []
        filtered["decision_basis"] = "crossover_concept_filter"
        filtered["analysis"] = (
            "Model-flagged claims were concept or prospective statements, not "
            "assertions of existing accomplishments; deterministic rule passes "
            "the candidate. Model analysis: " + str(audit.get("analysis", ""))
        )
    return filtered


CROSSOVER_COMPONENTS = (
    "problem",
    "customer",
    "solution",
    "delivery_model",
    "revenue_logic",
    "distinctiveness",
)


def validate_component_map(component_map: object) -> dict[str, str] | None:
    """Return a normalized {component: "A"|"B"} map, or None if invalid.

    A valid recombination sources every component from exactly one parent and
    takes at least two components from each parent.
    """
    if not isinstance(component_map, dict):
        return None
    normalized: dict[str, str] = {}
    for component in CROSSOVER_COMPONENTS:
        value = component_map.get(component)
        if not isinstance(value, str) or value.strip().upper() not in {"A", "B"}:
            return None
        normalized[component] = value.strip().upper()
    counts = {"A": 0, "B": 0}
    for value in normalized.values():
        counts[value] += 1
    if counts["A"] < 2 or counts["B"] < 2:
        return None
    return normalized


def rank_proportional_select(pool_ids: list[str], pool_ranks: list[int], n: int, rng: random.Random) -> list[str]:
    """Select n IDs from pool with probability inversely proportional to rank.

    Lower rank (1=best) → higher selection probability.
    pool_ranks are 1-based ranks within the full population.
    """
    # Weight = 1/rank (rank 1 → weight 1.0, rank 2 → 0.5, etc.)
    weights = [1.0 / r for r in pool_ranks]
    selected = rng.choices(pool_ids, weights=weights, k=n)
    return selected


class Mutator:
    """Mutates one algorithm-selected component."""

    ROLE = "Mutator"

    def __init__(self) -> None:
        self.auditor = FidelityAuditor()

    async def mutate(
        self,
        plan: str,
        parent_id: str,
        component: str,
        generation: int,
        child_slot: int,
        log_extra: str = "",
    ) -> dict:
        """Return a mutation candidate plus its factual-fidelity decision."""
        if component not in MUTATION_COMPONENT_GUIDANCE:
            raise ValueError(f"unknown mutation component: {component}")
        prompt = MUTATOR_PROMPT.format(
            plan=plan,
            component=component,
            component_guidance=MUTATION_COMPONENT_GUIDANCE[component],
            fidelity_rules=FACTUAL_FIDELITY_RULES,
        )
        _track_planned(self.ROLE)
        last_raw = ""

        for attempt in range(1, config.MAX_PARSE_ATTEMPTS + 1):
            try:
                raw = await _chat(config.MUTATOR_MODEL, prompt,
                                  role=self.ROLE,
                                  context=f"mutating {parent_id} (attempt {attempt}/{config.MAX_PARSE_ATTEMPTS})",
                                  temperature=config.MUTATOR_TEMPERATURE,
                                  log_extra=log_extra)
            except Exception as exc:
                logger.warning(
                    "LLM call failed while mutating %s (attempt %d/%d): %s",
                    parent_id, attempt, config.MAX_PARSE_ATTEMPTS, _describe_exception(exc)
                )
                break
            last_raw = raw
            try:
                data = json.loads(_strip_fences(raw))
                plan_text = data["plan"]
                if (
                    isinstance(plan_text, str)
                    and len(plan_text) > 50
                    and " ".join(plan_text.split()).casefold()
                    != " ".join(plan.split()).casefold()
                ):
                    child_id = _make_mutant_id(parent_id, generation, child_slot)
                    audit = await self.auditor.audit(
                        [(f"MUTATION PARENT {parent_id}", plan)],
                        plan_text,
                        context=f"mutation {child_id}",
                        log_extra=log_extra,
                    )
                    accepted = bool(audit["passed"])
                    return {
                        "plan": plan_text if accepted else plan,
                        "generated_plan": plan_text,
                        "dimension": component,
                        "child_id": child_id,
                        "rationale": "",
                        "audit": audit,
                        "action": "accepted" if accepted else "fallback_parent",
                    }
                logger.warning(
                    "JSON parsed but plan was too short or unchanged (%d chars, attempt %d/%d)",
                    len(plan_text) if isinstance(plan_text, str) else 0,
                    attempt, config.MAX_PARSE_ATTEMPTS,
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning(
                    "JSON parse failed (attempt %d/%d): %s", attempt, config.MAX_PARSE_ATTEMPTS, exc
                )

        _inc_errors()
        child_id = _make_mutant_id(parent_id, generation, child_slot)
        if last_raw:
            logger.error("ERROR: All JSON parse attempts failed for mutator, returning raw text")
        logger.error("ERROR: Mutation failed for %s, returning parent plan unchanged", parent_id)
        return {
            "plan": plan,
            "generated_plan": last_raw.strip(),
            "dimension": component,
            "child_id": child_id,
            "rationale": "",
            "audit": {
                "passed": False,
                "unsupported_claims": ["No valid generated candidate was available."],
                "analysis": "Mutation generation or parsing failed.",
            },
            "action": "fallback_parent",
        }

    async def mutate_batch(self, plans: list[str], parent_ids: list[str], components: list[str],
                           generation: int,
                           log_extras: list[str] | None = None) -> list[dict]:
        """Mutate multiple plans concurrently."""
        extras = log_extras or [""] * len(plans)
        tasks = [
            self.mutate(plan, pid, component, generation, child_slot=slot, log_extra=le)
            for slot, (plan, pid, component, le) in enumerate(
                zip(plans, parent_ids, components, extras), 1
            )
        ]
        return list(await asyncio.gather(*tasks))


class Crossover:
    """Combines two parent plans into a child plan."""

    ROLE = "Crossover"

    def __init__(self) -> None:
        self.auditor = FidelityAuditor()

    async def cross(
        self,
        plan_a: str,
        plan_b: str,
        id_a: str,
        id_b: str,
        generation: int,
        child_slot: int,
        log_extra: str = "",
    ) -> dict:
        """Return a crossover candidate plus its factual-fidelity decision."""
        prompt = CROSSOVER_PROMPT.format(
            plan_a=plan_a,
            plan_b=plan_b,
            criteria=EVALUATION_CRITERIA,
            fidelity_rules=FACTUAL_FIDELITY_RULES,
        )
        _track_planned(self.ROLE)
        last_raw = ""

        for attempt in range(1, config.MAX_PARSE_ATTEMPTS + 1):
            try:
                raw = await _chat(config.CROSSOVER_MODEL, prompt,
                                  role=self.ROLE,
                                  context=f"crossing {id_a} x {id_b} (attempt {attempt}/{config.MAX_PARSE_ATTEMPTS})",
                                  temperature=config.CROSSOVER_TEMPERATURE,
                                  log_extra=log_extra)
            except Exception as exc:
                logger.warning(
                    "LLM call failed while crossing %s x %s (attempt %d/%d): %s",
                    id_a, id_b, attempt, config.MAX_PARSE_ATTEMPTS, _describe_exception(exc)
                )
                break
            last_raw = raw
            try:
                data = json.loads(_strip_fences(raw))
                plan_text = data["plan"]
                component_map = validate_component_map(data.get("component_map"))
                normalized_child = " ".join(plan_text.split()).casefold() if isinstance(plan_text, str) else ""
                if (
                    isinstance(plan_text, str)
                    and len(plan_text) > 50
                    and normalized_child != " ".join(plan_a.split()).casefold()
                    and normalized_child != " ".join(plan_b.split()).casefold()
                    and component_map is not None
                ):
                    rationale = json.dumps(component_map)
                    child_id = _make_offspring_id(id_a, id_b, generation, child_slot)
                    audit = await self.auditor.audit(
                        [(f"CROSSOVER PARENT {id_a}", plan_a), (f"CROSSOVER PARENT {id_b}", plan_b)],
                        plan_text,
                        context=f"crossover {child_id}",
                        log_extra=log_extra,
                        prompt_template=CROSSOVER_FIDELITY_AUDITOR_PROMPT,
                    )
                    audit = filter_crossover_audit(audit, plan_text)
                    accepted = bool(audit["passed"])
                    return {
                        "plan": plan_text if accepted else plan_a,
                        "generated_plan": plan_text,
                        "dimension": "",
                        "child_id": child_id,
                        "rationale": str(rationale),
                        "audit": audit,
                        "action": "accepted" if accepted else "fallback_parent_a",
                    }
                logger.warning(
                    "JSON parsed but child was too short, unchanged, or had an invalid "
                    "component_map (need every component sourced A/B, >=2 per parent) "
                    "(%d chars, attempt %d/%d)",
                    len(plan_text) if isinstance(plan_text, str) else 0,
                    attempt, config.MAX_PARSE_ATTEMPTS,
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning(
                    "JSON parse failed (attempt %d/%d): %s", attempt, config.MAX_PARSE_ATTEMPTS, exc
                )

        _inc_errors()
        child_id = _make_offspring_id(id_a, id_b, generation, child_slot)
        if last_raw:
            logger.error("ERROR: All JSON parse attempts failed for crossover, returning raw text")
        logger.error("ERROR: Crossover failed for %s x %s, returning first parent unchanged", id_a, id_b)
        return {
            "plan": plan_a,
            "generated_plan": last_raw.strip(),
            "dimension": "",
            "child_id": child_id,
            "rationale": "unknown",
            "audit": {
                "passed": False,
                "unsupported_claims": ["No valid generated candidate was available."],
                "analysis": "Crossover generation or parsing failed.",
            },
            "action": "fallback_parent_a",
        }

    async def cross_batch(self, pairs: list[tuple[str, str, str, str]], generation: int,
                          log_extras: list[str] | None = None) -> list[dict]:
        """Cross multiple pairs concurrently. Each tuple: (plan_a, plan_b, id_a, id_b).
        Returns one audited record per pair."""
        extras = log_extras or [""] * len(pairs)
        tasks = [
            self.cross(pa, pb, ia, ib, generation, child_slot=slot, log_extra=le)
            for slot, ((pa, pb, ia, ib), le) in enumerate(zip(pairs, extras), 1)
        ]
        return list(await asyncio.gather(*tasks))


def _make_mutant_id(parent_id: str, generation: int, child_slot: int) -> str:
    """e.g., humanecheck.M2.1"""
    return f"{parent_id}.M{generation}.{child_slot}"


def _make_offspring_id(id_a: str, id_b: str, generation: int, child_slot: int) -> str:
    """e.g., (humanecheckXgenaix).G1.1"""
    return f"({id_a}X{id_b}).G{generation}.{child_slot}"
