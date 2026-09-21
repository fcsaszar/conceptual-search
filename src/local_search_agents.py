"""Generation and parsing agents used by local search."""

import asyncio
import json
import logging
import re

import config
from local_search_prompts import (
    EVALUATION_CRITERIA,
    FACTUAL_FIDELITY_RULES,
    GENERATOR_PROMPT,
    SPECIALIZED_GENERATOR_PROMPT,
)
from agents_common import (
    client,
    _chat,
    _describe_exception,
    _strip_fences,
    _short_model,
    _before_sleep,
    _track_planned,
    _track_chat,
    get_call_stats,
    _inc_errors,
    get_error_count,
    Evaluator,
    FidelityAuditor,
    Narrator,
)

logger = logging.getLogger(__name__)


_BACKUP_IMPROVEMENT_IDEAS = [
    "Clarify the value proposition with a concrete before-and-after example.",
    "Add stronger evidence of demand from a clearly defined target customer.",
    "Show more proof of feasibility and a clearer delivery plan.",
    "Explain the revenue model and path to sustainability more concretely.",
    "State more clearly what differentiates the product from existing alternatives.",
    "Specify a validation test and the evidence that would be needed, without claiming it already exists.",
    "Acknowledge major risks and explain how they will be mitigated.",
]


def _fallback_ideas(num_ideas: int) -> list[str]:
    ideas = []
    for idx in range(num_ideas):
        ideas.append(_BACKUP_IMPROVEMENT_IDEAS[idx % len(_BACKUP_IMPROVEMENT_IDEAS)])
    return ideas


class Generator:
    """Generates improvement ideas from a plan."""

    ROLE = "Generator"

    async def generate(self, plan: str, log_extra: str = "") -> list[str]:
        prompt = GENERATOR_PROMPT.format(
            plan=plan,
            criteria=EVALUATION_CRITERIA,
            num_ideas=config.NUM_IDEAS,
            fidelity_rules=FACTUAL_FIDELITY_RULES,
        )
        _track_planned(self.ROLE)
        last_raw = ""

        for attempt in range(1, config.MAX_PARSE_ATTEMPTS + 1):
            try:
                raw = await _chat(config.GENERATOR_MODEL, prompt,
                                  role=self.ROLE,
                                  context=f"generating {config.NUM_IDEAS} ideas (attempt {attempt}/{config.MAX_PARSE_ATTEMPTS})",
                                  temperature=config.GENERATOR_TEMPERATURE,
                                  log_extra=log_extra)
            except Exception as exc:
                logger.warning(
                    "LLM call failed while generating ideas (attempt %d/%d): %s",
                    attempt, config.MAX_PARSE_ATTEMPTS, _describe_exception(exc)
                )
                break
            last_raw = raw
            try:
                data = json.loads(_strip_fences(raw))
                ideas = data["ideas"]
                if isinstance(ideas, list) and len(ideas) == config.NUM_IDEAS and all(isinstance(s, str) for s in ideas):
                    for i, idea in enumerate(ideas, 1):
                        logger.debug("Idea %d: %.120s", i, idea)
                    return ideas
                logger.warning(
                    "JSON parsed but 'ideas' invalid (got %d items, attempt %d/%d)",
                    len(ideas) if isinstance(ideas, list) else -1,
                    attempt, config.MAX_PARSE_ATTEMPTS,
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning(
                    "JSON parse failed (attempt %d/%d): %s", attempt, config.MAX_PARSE_ATTEMPTS, exc
                )

        # Fallback: regex parsing (last resort)
        _inc_errors()
        if last_raw:
            logger.error("ERROR: All JSON parse attempts failed, falling back to regex parsing")
            ideas = re.split(r"\n\s*\d+[\.\)]\s*", last_raw)
            ideas = [idea.strip() for idea in ideas if idea.strip()]
            if len(ideas) == 0:
                ideas = [last_raw.strip()]
            while len(ideas) < config.NUM_IDEAS:
                ideas.append(ideas[len(ideas) % len(ideas)])
            ideas = ideas[:config.NUM_IDEAS]
        else:
            logger.error("ERROR: Idea generation failed, using deterministic backup ideas")
            ideas = _fallback_ideas(config.NUM_IDEAS)

        for i, idea in enumerate(ideas, 1):
            logger.debug("Idea %d (fallback): %.120s", i, idea)
        return ideas


class SpecializedGenerator:
    """Takes a plan and one idea, produces an updated plan."""

    ROLE = "Specialized Generator"

    def __init__(self) -> None:
        self.auditor = FidelityAuditor()

    async def generate(self, plan: str, idea: str, alt_num: int = 0, log_extra: str = "") -> dict:
        prompt = SPECIALIZED_GENERATOR_PROMPT.format(
            plan=plan,
            idea=idea,
            fidelity_rules=FACTUAL_FIDELITY_RULES,
        )
        _track_planned(self.ROLE)
        last_error = ""
        generated_plan = ""
        evidence_notes = ""
        for attempt in range(1, config.MAX_PARSE_ATTEMPTS + 1):
            try:
                raw = await _chat(
                    config.SPECIALIZED_GENERATOR_MODEL,
                    prompt,
                    role=self.ROLE,
                    context=f"alt {alt_num} (parse attempt {attempt})",
                    temperature=config.SPECIALIZED_GENERATOR_TEMPERATURE,
                    log_extra=log_extra,
                )
                data = json.loads(_strip_fences(raw))
                generated_plan = data.get("plan", "")
                evidence_notes = data.get("evidence_notes", "")
                if isinstance(generated_plan, str) and len(generated_plan) > 50:
                    break
                last_error = "generated plan was missing or too short"
            except (json.JSONDecodeError, TypeError, AttributeError) as exc:
                last_error = str(exc)
            except Exception as exc:
                last_error = _describe_exception(exc)
                break
            logger.warning(
                "Specialized generator parse failed for alt %d (attempt %d/%d): %s",
                alt_num,
                attempt,
                config.MAX_PARSE_ATTEMPTS,
                last_error,
            )

        if not generated_plan or len(generated_plan) <= 50:
            _inc_errors()
            return {
                "plan": plan,
                "generated_plan": generated_plan,
                "evidence_notes": evidence_notes,
                "audit": {
                    "passed": False,
                    "unsupported_claims": ["No valid generated candidate was available."],
                    "analysis": last_error or "Generation failed.",
                    "model_passed": "",
                    "model_unsupported_claims": [],
                    "decision_basis": "generation_unavailable",
                    "unavailable": False,
                },
                "action": "fallback_parent",
            }

        audit = await self.auditor.audit(
            [("LOCAL PARENT", plan)],
            generated_plan,
            context=f"local alt {alt_num}",
            log_extra=log_extra,
        )
        accepted = bool(audit["passed"])
        return {
            "plan": generated_plan if accepted else plan,
            "generated_plan": generated_plan,
            "evidence_notes": str(evidence_notes),
            "audit": audit,
            "action": "accepted" if accepted else "fallback_parent",
        }

    async def generate_all_variations(
        self, plan: str, ideas: list[str], log_extras: list[str] | None = None
    ) -> list[dict]:
        extras = log_extras or [""] * len(ideas)
        tasks = [self.generate(plan, idea, alt_num=i, log_extra=extras[i - 1]) for i, idea in enumerate(ideas, 1)]
        return list(await asyncio.gather(*tasks))
