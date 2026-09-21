"""Shared model client, retry, cache, evaluation, and audit utilities."""

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import sqlite3

from openai import AsyncOpenAI
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

import config
from local_search_prompts import (
    EVALUATION_CRITERIA,
    EVALUATOR_PROMPT,
    FACTUAL_FIDELITY_RULES,
    FIDELITY_AUDITOR_PROMPT,
    NARRATIVE_DIFF_PROMPT,
)

logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    base_url=config.OPENROUTER_BASE_URL,
    api_key=config.OPENROUTER_API_KEY or "missing-openrouter-api-key",
)

# --- Concurrency semaphore -----------------------------------------------------
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_CALLS)
    return _semaphore


# --- Call tracking ----------------------------------------------------------
# Each role maps to [planned, total_chat_calls].
# retries = total - planned.
_call_stats: dict[str, list[int]] = {}
_cache_hits: dict[str, int] = {}
_usage_stats = {
    "input_tokens": 0,
    "output_tokens": 0,
    "reported_cost_usd": 0.0,
}


def _track_planned(role: str):
    _call_stats.setdefault(role, [0, 0])
    _call_stats[role][0] += 1


def _track_chat(role: str):
    _call_stats.setdefault(role, [0, 0])
    _call_stats[role][1] += 1


def get_call_stats() -> dict[str, list[int]]:
    """Return {role: [planned, total]} for all roles."""
    return dict(_call_stats)


def get_cache_stats() -> dict[str, int]:
    """Return cache-hit counts by role for the current process."""
    return dict(_cache_hits)


def get_usage_stats() -> dict[str, int | float]:
    """Return token and provider-reported cost totals for live calls."""
    return dict(_usage_stats)


# --- Error tracking --------------------------------------------------------
_error_count = 0


def _inc_errors():
    global _error_count
    _error_count += 1


def get_error_count() -> int:
    return _error_count


# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) if present."""
    m = re.match(r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$", text, re.DOTALL)
    return m.group(1) if m else text


_cache_connection: sqlite3.Connection | None = None
_cache_connection_path = ""


def _call_cache_path() -> str:
    return os.environ.get("AI_ENTREP_CALL_CACHE", config.CALL_CACHE_PATH).strip()


def _get_cache_connection() -> sqlite3.Connection | None:
    """Open the per-run SQLite call ledger when caching is configured."""
    global _cache_connection, _cache_connection_path
    path = _call_cache_path()
    if not path:
        return None
    if _cache_connection is not None and _cache_connection_path == path:
        return _cache_connection

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS calls (
            cache_key TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            model TEXT NOT NULL,
            role TEXT NOT NULL,
            context TEXT NOT NULL,
            temperature REAL,
            max_tokens INTEGER NOT NULL,
            prompt_sha256 TEXT NOT NULL,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            reported_cost_usd REAL NOT NULL DEFAULT 0
        )
        """
    )
    connection.commit()
    _cache_connection = connection
    _cache_connection_path = path
    return connection


def _cache_key(
    model: str,
    prompt: str,
    role: str,
    context: str,
    temperature: float | None,
) -> str:
    request = {
        "schema": config.PROMPT_SCHEMA_VERSION,
        "model": model,
        "prompt": prompt,
        "role": role,
        "context": context,
        "temperature": temperature,
        "max_tokens": config.MAX_RESPONSE_TOKENS,
        "reasoning_effort": config.REASONING_EFFORT,
    }
    payload = json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cached_response(cache_key: str) -> str | None:
    connection = _get_cache_connection()
    if connection is None:
        return None
    row = connection.execute(
        "SELECT response FROM calls WHERE cache_key = ?", (cache_key,)
    ).fetchone()
    return None if row is None else str(row[0])


def _usage_value(usage, *names: str) -> int:
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)
        if isinstance(usage, dict) and usage.get(name) is not None:
            return int(usage[name])
    return 0


def _response_usage(response) -> tuple[int, int, float]:
    usage = getattr(response, "usage", None) or {}
    input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
    output_tokens = _usage_value(usage, "completion_tokens", "output_tokens")
    cost = getattr(usage, "cost", None)
    if cost is None and isinstance(usage, dict):
        cost = usage.get("cost")
    if cost is None:
        model_extra = getattr(response, "model_extra", None) or {}
        extra_usage = model_extra.get("usage", {}) if isinstance(model_extra, dict) else {}
        cost = extra_usage.get("cost", 0.0) if isinstance(extra_usage, dict) else 0.0
    return input_tokens, output_tokens, float(cost or 0.0)


def _store_cached_response(
    cache_key: str,
    model: str,
    prompt: str,
    response_text: str,
    role: str,
    context: str,
    temperature: float | None,
    input_tokens: int,
    output_tokens: int,
    reported_cost_usd: float,
) -> None:
    connection = _get_cache_connection()
    if connection is None:
        return
    connection.execute(
        """
        INSERT OR IGNORE INTO calls (
            cache_key, created_at, model, role, context, temperature,
            max_tokens, prompt_sha256, prompt, response, input_tokens,
            output_tokens, reported_cost_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cache_key,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model,
            role,
            context,
            temperature,
            config.MAX_RESPONSE_TOKENS,
            hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            prompt,
            response_text,
            input_tokens,
            output_tokens,
            reported_cost_usd,
        ),
    )
    connection.commit()


def _prompt_section(prompt: str, start: str, end: str) -> str:
    match = re.search(
        re.escape(start) + r"\s*(.*?)\s*" + re.escape(end),
        prompt,
        re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _mock_chat_response(role: str, prompt: str) -> str:
    """Return deterministic schema-valid responses for offline integration tests."""
    if role == "Evaluator":
        plan_a = _prompt_section(prompt, "--- PROJECT A ---", "--- END PROJECT A ---")
        plan_b = _prompt_section(prompt, "--- PROJECT B ---", "--- END PROJECT B ---")
        digest_a = hashlib.sha256(" ".join(plan_a.split()).encode("utf-8")).hexdigest()
        digest_b = hashlib.sha256(" ".join(plan_b.split()).encode("utf-8")).hexdigest()
        winner = "A" if digest_a <= digest_b else "B"
        return json.dumps(
            {
                "analysis": "Offline deterministic comparison used only to test the workflow.",
                "winner": winner,
            }
        )
    if role == "Generator":
        ideas = [
            "Frame a proposed customer interview that would test the value proposition.",
            "Clarify the target segment without claiming new demand evidence.",
            "Describe a proposed feasibility milestone and its decision criterion.",
            "State a hypothetical pricing test and the assumptions it would examine.",
            "Name a key risk and a proposed experiment for reducing it.",
        ]
        return json.dumps({"ideas": ideas[: config.NUM_IDEAS]})
    if role == "Specialized Generator":
        plan = _prompt_section(prompt, "--- BEGIN PROJECT ---", "--- END PROJECT ---")
        idea = _prompt_section(prompt, "--- IMPROVEMENT IDEA ---", "--- END IDEA ---")
        candidate = (
            plan
            + "\n\nProposed validation refinement (not completed evidence): "
            + idea
        )
        return json.dumps(
            {
                "plan": candidate,
                "evidence_notes": "The added material is explicitly a proposal, not an accomplished fact.",
            }
        )
    if role == "Mutator":
        plan = _prompt_section(prompt, "--- BEGIN PROJECT ---", "--- END PROJECT ---")
        component_match = re.search(
            r"selected this component for mutation:\s*\n([a-z_]+):",
            prompt,
        )
        component = component_match.group(1) if component_match else "unknown"
        return json.dumps(
            {
                "component": component,
                "plan": plan
                + f"\n\nProposed mutation for future testing: clarify the {component} "
                "through an explicitly planned validation exercise.",
            }
        )
    if role == "Crossover":
        plan_a = _prompt_section(prompt, "--- PARENT A ---", "--- END PARENT A ---")
        plan_b = _prompt_section(prompt, "--- PARENT B ---", "--- END PARENT B ---")
        return json.dumps(
            {
                "component_map": {
                    "problem": "A",
                    "customer": "A",
                    "solution": "A",
                    "delivery_model": "A",
                    "revenue_logic": "B",
                    "distinctiveness": "B",
                },
                "plan": plan_a
                + "\n\nRecombined revenue logic and distinctiveness carried from "
                "Parent B (offline mock):\n"
                + plan_b[-400:]
                + "\n\nAny cross-parent integration is framed as a proposal; no "
                "partnership, traction, or implementation is claimed.",
            }
        )
    if role == "Fidelity Auditor":
        return json.dumps(
            {
                "passed": True,
                "unsupported_claims": [],
                "analysis": "Offline mock candidate uses explicitly proposed language.",
            }
        )
    if role == "Narrator":
        return "Offline mock summary: the candidate adds a proposed validation step without asserting new evidence."
    raise RuntimeError(f"No offline mock response is defined for role {role!r}")

def _short_model(model: str) -> str:
    """Extract last segment of model ID: 'openai/gpt-5-mini' -> 'gpt-5-mini'."""
    return model.rsplit("/", 1)[-1]


def _describe_exception(exc: BaseException) -> str:
    """Return a readable error string for retries and logs."""
    if isinstance(exc, RetryError):
        last_exc = exc.last_attempt.exception()
        if last_exc is not None:
            return _describe_exception(last_exc)
    if isinstance(exc, asyncio.TimeoutError):
        return f"timed out after {config.API_CALL_TIMEOUT}s"
    message = str(exc).strip()
    return message or exc.__class__.__name__


def rank_from_wins(wins: list[float], tie_break_rng: random.Random | None = None) -> list[int]:
    """Convert wins to 1-based ranks, optionally randomizing equal-win groups."""
    grouped: dict[float, list[int]] = {}
    for idx, win_count in enumerate(wins):
        grouped.setdefault(win_count, []).append(idx)

    ordered_indices: list[int] = []
    for win_count in sorted(grouped, reverse=True):
        group = grouped[win_count][:]
        if tie_break_rng is not None and len(group) > 1:
            tie_break_rng.shuffle(group)
        ordered_indices.extend(group)

    ranks = [0] * len(wins)
    for rank, idx in enumerate(ordered_indices, 1):
        ranks[idx] = rank
    return ranks


def _before_sleep(retry_state):
    """Log a warning before each tenacity retry."""
    args = retry_state.args
    kw = retry_state.kwargs
    role = kw.get("role", "")
    model = args[0] if args else kw.get("model", "")
    context = kw.get("context", "")
    exc = retry_state.outcome.exception()
    n = retry_state.attempt_number
    wait = retry_state.next_action.sleep
    logger.warning(
        "  Retry %s (%s): %s -- %s (attempt %d, waiting %.0fs)",
        role, _short_model(model), context, _describe_exception(exc), n, wait,
    )


@retry(
    stop=stop_after_attempt(config.API_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=config.API_RETRY_WAIT_MIN, max=config.API_RETRY_WAIT_MAX),
    before_sleep=_before_sleep,
)
async def _chat(model: str, prompt: str, role: str = "", context: str = "", temperature: float | None = 0.7, log_extra: str = "") -> str:
    if os.environ.get("AI_ENTREP_MOCK_LLM", "").lower() in {"1", "true", "yes"}:
        _track_chat(role)
        text = _mock_chat_response(role, prompt)
        logger.info(
            "  %s (%s): %s -> MOCK (%d chars%s)",
            role,
            _short_model(model),
            context,
            len(text),
            log_extra,
        )
        return text
    cache_key = _cache_key(model, prompt, role, context, temperature)
    cached = _cached_response(cache_key)
    if cached is not None:
        _cache_hits[role] = _cache_hits.get(role, 0) + 1
        logger.info(
            "  %s (%s): %s -> CACHE (%d chars%s)",
            role,
            _short_model(model),
            context,
            len(cached),
            log_extra,
        )
        return cached

    _track_chat(role)
    client.api_key = config.require_openrouter_api_key()
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config.MAX_RESPONSE_TOKENS,
        extra_body={"reasoning": {"effort": config.REASONING_EFFORT}},
    )
    if temperature is not None:
        kwargs["temperature"] = temperature
    async with _get_semaphore():
        response = await asyncio.wait_for(
            client.chat.completions.create(**kwargs),
            timeout=config.API_CALL_TIMEOUT,
        )
    text = response.choices[0].message.content
    input_tokens, output_tokens, reported_cost_usd = _response_usage(response)
    _usage_stats["input_tokens"] += input_tokens
    _usage_stats["output_tokens"] += output_tokens
    _usage_stats["reported_cost_usd"] += reported_cost_usd
    _store_cached_response(
        cache_key,
        model,
        prompt,
        text,
        role,
        context,
        temperature,
        input_tokens,
        output_tokens,
        reported_cost_usd,
    )
    logger.info("  %s (%s): %s -> OK (%d chars%s)", role, _short_model(model), context, len(text), log_extra)
    return text


class FidelityAuditor:
    """Reject generated candidates that add unsupported real-world facts."""

    ROLE = "Fidelity Auditor"

    _PROSPECTIVE_MARKERS = re.compile(
        r"\b(?:hypothetical|prospective|proposed|planned|future)\b|"
        r"\b(?:would|could|will)\b|"
        r"\b(?:we|the team|the project|the company|the platform)\s+"
        r"(?:plan|plans|propose|proposes|intend|intends|aim|aims)\b",
        re.IGNORECASE,
    )
    _EXISTING_ACHIEVEMENT_MARKERS = re.compile(
        r"\b(?:already|currently|to date|so far|at present)\b|"
        r"\b(?:current|existing)\s+(?:customers?|users?|partners?|revenue|sales|pilots?|tests?)\b|"
        r"\b(?:we|the team|the project|the company|the platform|the product)\s+"
        r"(?:has|have|had|serves?|partners?|works?)\b|"
        r"\b(?:has|have|had)\s+(?:already\s+)?"
        r"(?:served|signed|raised|generated|launched|built|tested|enrolled|secured|partnered|won|earned)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _claim_context(cls, candidate: str, claim: str) -> str:
        """Return the candidate sentence containing a flagged claim when possible."""
        candidate_folded = candidate.casefold()
        claim_folded = claim.casefold().strip()
        start = candidate_folded.find(claim_folded)
        if start < 0:
            return claim
        left_breaks = [candidate.rfind(mark, 0, start) for mark in (".", "!", "?", "\n")]
        left = max(left_breaks) + 1
        end_start = start + len(claim)
        right_breaks = [
            pos for mark in (".", "!", "?", "\n")
            if (pos := candidate.find(mark, end_start)) >= 0
        ]
        right = min(right_breaks) + 1 if right_breaks else len(candidate)
        return candidate[left:right].strip()

    @classmethod
    def _is_clearly_prospective_claim(cls, candidate: str, claim: str) -> bool:
        """Recognize explicit proposals without relaxing mixed/current fact claims."""
        context = cls._claim_context(candidate, claim)
        return bool(cls._PROSPECTIVE_MARKERS.search(context)) and not bool(
            cls._EXISTING_ACHIEVEMENT_MARKERS.search(context)
        )

    async def audit(
        self,
        sources: list[tuple[str, str]],
        candidate: str,
        context: str,
        log_extra: str = "",
        prompt_template: str | None = None,
    ) -> dict:
        source_text = "\n\n".join(
            f"SOURCE {label}:\n{text}" for label, text in sources
        )
        prompt = (prompt_template or FIDELITY_AUDITOR_PROMPT).format(
            sources=source_text,
            candidate=candidate,
            fidelity_rules=FACTUAL_FIDELITY_RULES,
        )
        _track_planned(self.ROLE)
        last_error = ""
        for attempt in range(1, config.MAX_PARSE_ATTEMPTS + 1):
            try:
                raw = await _chat(
                    config.FIDELITY_AUDITOR_MODEL,
                    prompt,
                    role=self.ROLE,
                    context=f"{context} (parse attempt {attempt})",
                    temperature=config.FIDELITY_AUDITOR_TEMPERATURE,
                    log_extra=log_extra,
                )
                data = json.loads(_strip_fences(raw))
                passed = data.get("passed")
                claims = data.get("unsupported_claims", [])
                analysis = data.get("analysis", "")
                if isinstance(passed, bool) and isinstance(claims, list):
                    normalized_claims = [str(claim) for claim in claims]
                    if (
                        not passed
                        and normalized_claims
                        and all(
                            self._is_clearly_prospective_claim(candidate, claim)
                            for claim in normalized_claims
                        )
                    ):
                        return {
                            "passed": True,
                            "unsupported_claims": [],
                            "analysis": (
                                "Prospective-framing safeguard overrode the model's "
                                "rejection because every flagged claim is explicitly "
                                "hypothetical, proposed, planned, or future-tense and none "
                                "asserts an existing accomplishment. Initial model analysis: "
                                + str(analysis)
                            ),
                            "model_passed": False,
                            "model_unsupported_claims": normalized_claims,
                            "decision_basis": "prospective_framing_safeguard",
                            "unavailable": False,
                        }
                    return {
                        "passed": passed,
                        "unsupported_claims": normalized_claims,
                        "analysis": str(analysis),
                        "model_passed": passed,
                        "model_unsupported_claims": normalized_claims,
                        "decision_basis": "model_audit",
                        "unavailable": False,
                    }
                last_error = "invalid audit fields"
            except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
                last_error = str(exc)
            except Exception as exc:
                last_error = _describe_exception(exc)
                break
            logger.warning(
                "Fidelity audit parse failed for %s (attempt %d/%d): %s",
                context,
                attempt,
                config.MAX_PARSE_ATTEMPTS,
                last_error,
            )

        _inc_errors()
        return {
            "passed": False,
            "unsupported_claims": ["Audit unavailable; conservative rejection."],
            "analysis": last_error or "Fidelity audit failed.",
            "model_passed": "",
            "model_unsupported_claims": [],
            "decision_basis": "audit_unavailable",
            "unavailable": True,
        }


class Evaluator:
    """Runs a double round-robin tournament among plans."""

    ROLE = "Evaluator"

    @staticmethod
    def _plan_label(idx: int) -> str:
        return "incumbent" if idx == 0 else f"alt {idx}"

    async def _match(self, plan_a: str, plan_b: str, label_a: str = "", label_b: str = "", log_extra: str = "") -> tuple[str, str]:
        if " ".join(plan_a.split()) == " ".join(plan_b.split()):
            return "TIE", "Exact same normalized text; skipped LLM call."
        prompt = EVALUATOR_PROMPT.format(
            criteria=EVALUATION_CRITERIA, plan_a=plan_a, plan_b=plan_b
        )
        _track_planned(self.ROLE)
        context = f"{label_a} vs {label_b}" if label_a else ""
        call_failed = False

        for attempt in range(1, config.MAX_PARSE_ATTEMPTS + 1):
            try:
                raw = await _chat(config.EVALUATOR_MODEL, prompt,
                                  role=self.ROLE, context=f"{context} (parse attempt {attempt})",
                                  temperature=config.EVALUATOR_TEMPERATURE,
                                  log_extra=log_extra)
            except Exception as exc:
                call_failed = True
                logger.warning(
                    "LLM call failed for %s (attempt %d/%d): %s",
                    context, attempt, config.MAX_PARSE_ATTEMPTS, _describe_exception(exc)
                )
                break
            try:
                data = json.loads(_strip_fences(raw))
                winner = data["winner"].upper()
                if winner in ("A", "B"):
                    logger.debug("Match result: %s", winner)
                    return winner, data.get("analysis", "")
                logger.warning(
                    "JSON parsed but winner='%s' invalid (attempt %d/%d)",
                    winner, attempt, config.MAX_PARSE_ATTEMPTS,
                )
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
                logger.warning(
                    "JSON parse failed (attempt %d/%d): %s", attempt, config.MAX_PARSE_ATTEMPTS, exc
                )

        _inc_errors()
        if call_failed:
            logger.error("ERROR: Evaluator call failed for match, defaulting to tie")
        else:
            logger.error("ERROR: All JSON parse attempts failed for match, defaulting to tie")
        return "TIE", ""

    async def evaluate(
        self,
        plans: list[str],
        match_log_extras: dict | None = None,
        tie_break_rng: random.Random | None = None,
    ) -> tuple[list[int], list[float], dict]:
        """Run double round-robin and return (ranks, win_rates, match_details)."""
        n = len(plans)
        wins = [0.0] * n
        extras = match_log_extras or {}
        match_details: dict[tuple[int, int], tuple[str, str]] = {}

        # Build all match pairs (double round-robin: each pair plays twice,
        # swapping positions to control for ordering bias).
        match_specs = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                match_specs.append((
                    i,
                    j,
                    self._plan_label(i),
                    self._plan_label(j),
                    extras.get((i, j), ""),
                ))

        total_matches = len(match_specs)
        logger.debug("Running %d evaluation matches", total_matches)

        for batch_start in range(0, total_matches, config.MATCH_BATCH_SIZE):
            batch_specs = match_specs[batch_start:batch_start + config.MATCH_BATCH_SIZE]
            batch_tasks = [
                self._match(
                    plans[i], plans[j],
                    label_a=label_a, label_b=label_b,
                    log_extra=log_extra,
                )
                for i, j, label_a, label_b, log_extra in batch_specs
            ]
            batch_results = await asyncio.gather(*batch_tasks)

            for (i, j, _label_a, _label_b, _log_extra), (winner_letter, raw) in zip(batch_specs, batch_results):
                if winner_letter == "A":
                    wins[i] += 1.0
                    winner_idx = i
                    winner_key = f"plan_{winner_idx}"
                else:
                    if winner_letter == "B":
                        wins[j] += 1.0
                        winner_idx = j
                        winner_key = f"plan_{winner_idx}"
                    else:
                        wins[i] += 0.5
                        wins[j] += 0.5
                        winner_key = "tie"
                match_details[(i, j)] = (winner_key, raw)

            if total_matches > config.MATCH_BATCH_SIZE:
                logger.info(
                    "  %s batch %d-%d / %d done",
                    self.ROLE,
                    batch_start + 1,
                    batch_start + len(batch_specs),
                    total_matches,
                )

        # Each plan plays 2*(n-1) matches
        matches_per_plan = 2 * (n - 1)
        win_rates = [w / matches_per_plan for w in wins]

        # Convert wins to ranks (1=best = most wins)
        ranks = rank_from_wins(wins, tie_break_rng=tie_break_rng)

        logger.debug("Wins: %s  Ranks: %s", wins, ranks)
        return ranks, win_rates, match_details


class Narrator:
    """Generates LLM summaries of differences between plan versions."""

    ROLE = "Narrator"

    async def summarize_diff(self, old_plan: str, new_plan: str) -> str:
        prompt = NARRATIVE_DIFF_PROMPT.format(old_plan=old_plan, new_plan=new_plan)
        _track_planned(self.ROLE)
        return await _chat(config.NARRATOR_MODEL, prompt,
                           role=self.ROLE, context="narrating diff",
                           temperature=config.NARRATOR_TEMPERATURE)

    async def narrate(self, prompt: str, context: str = "") -> str:
        _track_planned(self.ROLE)
        return await _chat(config.NARRATOR_MODEL, prompt,
                           role=self.ROLE, context=context,
                           temperature=config.NARRATOR_TEMPERATURE)
