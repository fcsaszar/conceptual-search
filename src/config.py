"""Locked research design, model settings, and environment configuration."""

import os


def _env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be at least 1, got {parsed}")
    return parsed


# ═══════════════════════════════════════════════════════════════════════════════
# LLM
# ═══════════════════════════════════════════════════════════════════════════════

# --- Provider (OpenRouter) ---------------------------------------------------

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEEPSEEK_V32_MODEL = "deepseek/deepseek-v3.2"
DEFAULT_OPENROUTER_MODEL = DEEPSEEK_V32_MODEL

# Every live response is cached under the active run directory.  Search stages
# can therefore be restarted with the same run id without paying for completed
# calls again.
CALL_CACHE_PATH = os.environ.get("AI_ENTREP_CALL_CACHE", "")
MOCK_LLM = os.environ.get("AI_ENTREP_MOCK_LLM", "").lower() in {"1", "true", "yes"}


def require_openrouter_api_key() -> str:
    """Return the OpenRouter key, failing only when a live LLM call is made."""
    global OPENROUTER_API_KEY
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is required for live LLM calls."
        )
    return OPENROUTER_API_KEY

# --- Shared agents (used by both local and global search) --------------------

BASE_MODEL = os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)  # Shared default model for all agents
BASE_TEMPERATURE = 0.5                 # Shared default sampling temperature

EVALUATOR_MODEL = BASE_MODEL               # Pairwise comparison judge
EVALUATOR_TEMPERATURE = BASE_TEMPERATURE   # Sampling temperature for evaluation

NARRATOR_MODEL = BASE_MODEL                # Post-run narrative summarizer
NARRATOR_TEMPERATURE = BASE_TEMPERATURE    # Sampling temperature for narrative summaries

FIDELITY_AUDITOR_MODEL = BASE_MODEL
FIDELITY_AUDITOR_TEMPERATURE = 0.0

# --- Robustness & concurrency ------------------------------------------------

MAX_CONCURRENT_CALLS = _env_int("AI_ENTREP_MAX_CONCURRENT_CALLS", 50)
MATCH_BATCH_SIZE = 500       # Matches dispatched per asyncio.gather batch
MAX_PARSE_ATTEMPTS = 5       # Retries when LLM output fails JSON parsing
API_RETRY_ATTEMPTS = 5       # Retries for transient API errors (network, rate-limit)
API_RETRY_WAIT_MIN = 2       # Minimum exponential backoff (seconds)
API_RETRY_WAIT_MAX = 30      # Maximum exponential backoff (seconds)
MAX_RESPONSE_TOKENS = 2000   # Hard ceiling on LLM output (~8K chars)
API_CALL_TIMEOUT = 120       # Seconds before an individual API call is cancelled
# These tasks require short, auditable structured answers.  Some OpenRouter
# models otherwise spend the entire response budget printing chain-of-thought
# text and never reach the requested JSON object.
REASONING_EFFORT = "none"
PROMPT_SCHEMA_VERSION = "analysis-06-recombination-v1"


# ═══════════════════════════════════════════════════════════════════════════════
# FILES
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_FILE = "in/projects.csv"                          # Seed project descriptions

OUTPUT_FILE    = "local-search.csv"                     # Local search: per-step results
LOG_FILE       = "local-search.log"                     # Local search: run log
NARRATIVE_FILE = "local-search-narrative.txt"           # Local search: evolution narrative

GS_OUTPUT_FILE    = "global-search.csv"                 # Global search: per-step results
GS_LOG_FILE       = "global-search.log"                 # Global search: run log
GS_NARRATIVE_FILE = "global-search-narrative.txt"       # Global search: evolution narrative


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL SEARCH — hill-climbing from a single seed plan
# ═══════════════════════════════════════════════════════════════════════════════

# --- Agents -------------------------------------------------------------------

GENERATOR_MODEL = BASE_MODEL                        # Proposes improvement ideas
GENERATOR_TEMPERATURE = BASE_TEMPERATURE            # Sampling temperature

SPECIALIZED_GENERATOR_MODEL = BASE_MODEL            # Rewrites plan to incorporate an idea
SPECIALIZED_GENERATOR_TEMPERATURE = BASE_TEMPERATURE  # Sampling temperature

# Evaluation and narration reuse the shared Evaluator/Narrator settings above.

# --- Process ------------------------------------------------------------------

# Local-search seed: the highest-ranked seed under the evaluator's own ranking.
# Overridable through the environment (the local-only comparison runs one
# local search from each seed).
SEED_PROJECT_ID = os.environ.get("AI_ENTREP_SEED_PROJECT_ID", "clockchain")
NUM_IDEAS = 5                # Improvement ideas (and plan variants) per step
NUM_STEPS = 12               # Shared search horizon for both local and global search
STUCK_THRESHOLD = None       # Disable early stopping; set an integer to stop after that many incumbent wins


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL SEARCH — evolving a population via selection, mutation, and crossover
# ═══════════════════════════════════════════════════════════════════════════════

# --- Agents -------------------------------------------------------------------

MUTATOR_MODEL = BASE_MODEL               # Perturbs a plan along one dimension
MUTATOR_TEMPERATURE = BASE_TEMPERATURE   # Sampling temperature

CROSSOVER_MODEL = BASE_MODEL             # Combines two parents into a child plan
CROSSOVER_TEMPERATURE = BASE_TEMPERATURE  # Sampling temperature

# Evaluation and narration reuse the shared Evaluator/Narrator settings above.

# --- Population ---------------------------------------------------------------

# The seed set is locked through a versioned firm-set file (hashed into every
# run manifest) rather than an in-code list.  The set must contain exactly
# EXPECTED_FIRM_COUNT unique ids, all present in the input CSV, and must
# include SEED_PROJECT_ID for the local-search stage.
EXPECTED_FIRM_COUNT = _env_int("AI_ENTREP_EXPECTED_FIRM_COUNT", 15)
LOCKED_FIRM_SET_PATH = "in/firm-set-15-evalrank.txt"


def validate_firm_set(firm_ids: list[str] | tuple[str, ...]) -> None:
    """Fail fast on a structurally invalid firm set."""
    if len(firm_ids) != EXPECTED_FIRM_COUNT:
        raise ValueError(
            f"the firm set must contain exactly {EXPECTED_FIRM_COUNT} firm ids, "
            f"got {len(firm_ids)}"
        )
    if len(set(firm_ids)) != len(firm_ids):
        raise ValueError("the firm set contains duplicate ids")
    if SEED_PROJECT_ID not in firm_ids:
        raise ValueError(
            f"the firm set must include the local-search seed "
            f"{SEED_PROJECT_ID!r}"
        )

POP_SIZE = 15           # Locked ex ante starting population
POP_ELITE = 5           # Plans kept unchanged each step (elitism)
POP_MUTANT = 5          # Plans created by mutation each step
POP_OFFSPRING = 5       # Plans created by crossover each step

# --- Evolution ----------------------------------------------------------------

SELECTION_POOL_FRAC = 2 / 3    # Fraction of population eligible as parents
RANDOM_SEED = 42               # RNG seed for reproducible selection and pairing

# Diversity-preserving elitism: at most this many elite slots may be held by
# plans sharing any one founding-seed lineage; remaining slots top up by
# selection rank when the cap cannot be met.
ELITE_LINEAGE_CAP = _env_int("AI_ENTREP_ELITE_LINEAGE_CAP", 2)

# Commit-by-verification: after the search steps, every plan that ever held an
# elite slot (plus step winners) is scored against the seed archive, and the
# champion is the benchmark-best plan.  A short polish round then mutates the
# champion this many times and keeps the best verified result.
COMMIT_POLISH_VARIANTS = 5

# Primary global-search novelty measure.  The model revision is pinned so a
# future replication does not silently inherit a changed model repository.
NOVELTY_EMBEDDING_MODEL = "BAAI/bge-m3"
NOVELTY_EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
NOVELTY_EMBEDDING_DEVICE = os.environ.get("AI_ENTREP_EMBEDDING_DEVICE", "cpu")
# The selection weights and the elite lineage cap are design levers,
# overridable through the environment so that alternative retention rules
# (for example quality-only selection without a lineage cap) can be run from
# the same code base.  The reported run used weights 0.5/0.5 and a cap of 2.
QUALITY_SELECTION_WEIGHT = float(os.environ.get("AI_ENTREP_QUALITY_WEIGHT", "0.5"))
NOVELTY_SELECTION_WEIGHT = float(os.environ.get("AI_ENTREP_NOVELTY_WEIGHT", "0.5"))


def sync_step_aliases() -> None:
    """Keep legacy names aligned with the shared step-count setting."""
    global MAX_ITERATIONS, NUM_GENERATIONS
    MAX_ITERATIONS = NUM_STEPS
    NUM_GENERATIONS = NUM_STEPS


sync_step_aliases()


def validate_local_search_config() -> None:
    """Fail fast on invalid local-search settings."""
    sync_step_aliases()
    if MATCH_BATCH_SIZE < 1:
        raise ValueError(f"MATCH_BATCH_SIZE must be at least 1, got {MATCH_BATCH_SIZE}")
    if API_CALL_TIMEOUT < 1:
        raise ValueError(f"API_CALL_TIMEOUT must be at least 1, got {API_CALL_TIMEOUT}")
    if NUM_IDEAS < 1:
        raise ValueError(f"NUM_IDEAS must be at least 1, got {NUM_IDEAS}")
    if NUM_STEPS < 1:
        raise ValueError(
            f"NUM_STEPS must be at least 1, got {NUM_STEPS}. "
            "Zero-step local runs are not supported."
        )
    if STUCK_THRESHOLD is not None and STUCK_THRESHOLD < 1:
        raise ValueError(
            f"STUCK_THRESHOLD must be None or at least 1, got {STUCK_THRESHOLD}"
        )


def get_selection_pool_size() -> int:
    """Return the configured global-search parent-pool size."""
    selection_pool_size = round(POP_SIZE * SELECTION_POOL_FRAC)
    if selection_pool_size < 2:
        raise ValueError(
            "Global search requires at least 2 plans in the selection pool for crossover; "
            f"got {selection_pool_size} from POP_SIZE={POP_SIZE} and "
            f"SELECTION_POOL_FRAC={SELECTION_POOL_FRAC}"
        )
    if selection_pool_size > POP_SIZE:
        raise ValueError(
            f"Selection pool size {selection_pool_size} cannot exceed POP_SIZE={POP_SIZE}"
        )
    return selection_pool_size


def validate_global_search_config() -> None:
    """Fail fast on invalid global-search settings."""
    sync_step_aliases()
    if MATCH_BATCH_SIZE < 1:
        raise ValueError(f"MATCH_BATCH_SIZE must be at least 1, got {MATCH_BATCH_SIZE}")
    if API_CALL_TIMEOUT < 1:
        raise ValueError(f"API_CALL_TIMEOUT must be at least 1, got {API_CALL_TIMEOUT}")
    if POP_SIZE < 2:
        raise ValueError(f"POP_SIZE must be at least 2, got {POP_SIZE}")
    if POP_ELITE < 0 or POP_MUTANT < 0 or POP_OFFSPRING < 0:
        raise ValueError(
            "POP_ELITE, POP_MUTANT, and POP_OFFSPRING must all be non-negative"
        )
    if POP_ELITE + POP_MUTANT + POP_OFFSPRING != POP_SIZE:
        raise ValueError(
            f"POP_ELITE({POP_ELITE}) + POP_MUTANT({POP_MUTANT}) + "
            f"POP_OFFSPRING({POP_OFFSPRING}) must equal POP_SIZE({POP_SIZE})"
        )
    if (POP_SIZE, POP_ELITE, POP_MUTANT, POP_OFFSPRING) != (15, 5, 5, 5):
        raise ValueError(
            "this pipeline locks global search to 15 plans: five elites, "
            "five mutants, and five crossover children"
        )
    if ELITE_LINEAGE_CAP < 1 or ELITE_LINEAGE_CAP > POP_ELITE:
        raise ValueError(
            f"ELITE_LINEAGE_CAP must be between 1 and POP_ELITE({POP_ELITE}), "
            f"got {ELITE_LINEAGE_CAP}"
        )
    if COMMIT_POLISH_VARIANTS < 0:
        raise ValueError(
            f"COMMIT_POLISH_VARIANTS must be non-negative, got {COMMIT_POLISH_VARIANTS}"
        )
    if NUM_STEPS < 1:
        raise ValueError(f"NUM_STEPS must be at least 1, got {NUM_STEPS}")
    if not 0 < SELECTION_POOL_FRAC <= 1:
        raise ValueError(
            f"SELECTION_POOL_FRAC must be in the interval (0, 1], got {SELECTION_POOL_FRAC}"
        )
    if QUALITY_SELECTION_WEIGHT < 0 or NOVELTY_SELECTION_WEIGHT < 0 or abs(
        QUALITY_SELECTION_WEIGHT + NOVELTY_SELECTION_WEIGHT - 1.0
    ) > 1e-9:
        raise ValueError(
            "selection weights must be non-negative and sum to 1"
        )
    get_selection_pool_size()


def apply_global_preset(preset: str) -> None:
    """Apply the locked 15-plan population design."""
    global POP_SIZE, POP_ELITE, POP_MUTANT, POP_OFFSPRING
    if preset == "15":
        POP_SIZE = 15
        POP_ELITE = 5
        POP_MUTANT = 5
        POP_OFFSPRING = 5
    else:
        raise ValueError("only the locked 15-firm preset is supported")


def set_all_agent_models(model: str) -> None:
    """Use one model for every agent in a run."""
    global BASE_MODEL
    global EVALUATOR_MODEL, NARRATOR_MODEL
    global GENERATOR_MODEL, SPECIALIZED_GENERATOR_MODEL
    global MUTATOR_MODEL, CROSSOVER_MODEL, FIDELITY_AUDITOR_MODEL
    BASE_MODEL = model
    EVALUATOR_MODEL = model
    NARRATOR_MODEL = model
    GENERATOR_MODEL = model
    SPECIALIZED_GENERATOR_MODEL = model
    MUTATOR_MODEL = model
    CROSSOVER_MODEL = model
    FIDELITY_AUDITOR_MODEL = model
