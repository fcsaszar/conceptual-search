"""Component-level semantic novelty for the global search.

The primary construct is a plan's mean cosine distance across six
venture-concept components.  Candidate novelty is its distance from the
nearest member of the fixed, ex ante seed archive.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Protocol, Sequence


COMPONENTS = (
    "problem",
    "customer",
    "solution",
    "delivery_model",
    "revenue_logic",
    "distinctiveness",
)

COMPONENT_LABELS = {
    "problem": "Problem",
    "customer": "Customer",
    "solution": "Solution",
    "delivery_model": "Delivery model",
    "revenue_logic": "Revenue logic",
    "distinctiveness": "Distinctiveness",
}

# The source campaigns use a stable set of Markdown headings.  Multiple
# sections may contribute to one theoretical component.  A generated plan may
# instead use the six component names directly; those aliases come first.
COMPONENT_SECTIONS = {
    "problem": ("problem",),
    "customer": ("customer", "target users & use cases", "target users and use cases"),
    "solution": (
        "solution",
        "solution / product overview",
        "solution product overview",
        "key features & capabilities",
        "key features and capabilities",
        "technical details",
    ),
    "delivery_model": (
        "delivery model",
        "roadmap & delivery plan",
        "roadmap and delivery plan",
        "what's included",
        "support & post-launch plans",
        "support and post-launch plans",
    ),
    "revenue_logic": (
        "revenue logic",
        "revenue model & sustainability",
        "revenue model and sustainability",
        "why crowdfunding / use of funds",
        "why crowdfunding use of funds",
        "what's included",
        "support & post-launch plans",
        "support and post-launch plans",
    ),
    "distinctiveness": (
        "distinctiveness",
        "one-sentence summary",
        "one sentence summary",
        "solution / product overview",
        "solution product overview",
        "key features & capabilities",
        "key features and capabilities",
    ),
}


class TextEncoder(Protocol):
    """Minimal interface shared by SentenceTransformer and the mock encoder."""

    def encode(self, texts: Sequence[str], **kwargs) -> Sequence[Sequence[float]]:
        """Return one vector per input text."""


@dataclass(frozen=True)
class SelectionMetrics:
    """Quality, novelty, and their equal-weight selection score."""

    quality_percentiles: list[float]
    novelty_ranks: list[float]
    novelty_percentiles: list[float]
    combined_scores: list[float]
    selection_ranks: list[int]


def _normalize_heading(heading: str) -> str:
    text = heading.strip().lower().replace("_", " ")
    text = re.sub(r"[*:`]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -")


def parse_markdown_sections(plan: str) -> dict[str, str]:
    """Parse Markdown headings into normalized section-name/text pairs."""
    sections: dict[str, list[str]] = {}
    current = "preamble"
    sections[current] = []
    for line in plan.splitlines():
        match = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
        if match:
            current = _normalize_heading(match.group(1))
            sections.setdefault(current, [])
        else:
            sections[current].append(line.strip())
    return {
        name: " ".join(part for part in lines if part).strip()
        for name, lines in sections.items()
        if any(part for part in lines)
    }


def component_texts(plan: str) -> dict[str, str]:
    """Project a campaign description onto the six theoretical components.

    Structured source plans are mapped using ``COMPONENT_SECTIONS``.  For an
    unstructured plan, the full text is used for every component rather than
    silently returning empty vectors.  A missing component in an otherwise
    structured plan receives the same explicit marker for every campaign.
    """
    sections = parse_markdown_sections(plan)
    has_headings = any(name != "preamble" for name in sections)
    if not has_headings:
        fallback = " ".join(plan.split()) or "Not specified."
        return {component: fallback for component in COMPONENTS}

    projected: dict[str, str] = {}
    for component in COMPONENTS:
        values = [
            sections[section]
            for section in COMPONENT_SECTIONS[component]
            if sections.get(section)
        ]
        projected[component] = " ".join(values).strip() or "Not specified."
    return projected


def percentile_from_rank(rank: float, n: int) -> float:
    """Map rank 1 to 1 and rank n to 0 (all to 1 when n is one)."""
    if n < 1:
        raise ValueError("n must be positive")
    if n == 1:
        return 1.0
    return (n - rank) / (n - 1)


def descending_fractional_ranks(values: Sequence[float]) -> list[float]:
    """Return descending ranks with the average rank assigned to exact ties."""
    n = len(values)
    ordered = sorted(range(n), key=lambda index: (-values[index], index))
    ranks = [0.0] * n
    position = 0
    while position < n:
        end = position + 1
        value = values[ordered[position]]
        while end < n and math.isclose(values[ordered[end]], value, rel_tol=0.0, abs_tol=1e-12):
            end += 1
        average_rank = ((position + 1) + end) / 2
        for ordered_position in range(position, end):
            ranks[ordered[ordered_position]] = average_rank
        position = end
    return ranks


def combine_quality_and_novelty(
    quality_ranks: Sequence[int],
    novelty_scores: Sequence[float],
    quality_weight: float = 0.5,
    novelty_weight: float = 0.5,
) -> SelectionMetrics:
    """Combine quality and novelty percentile ranks with the given weights."""
    if len(quality_ranks) != len(novelty_scores):
        raise ValueError("quality_ranks and novelty_scores must have equal length")
    n = len(quality_ranks)
    if n == 0:
        raise ValueError("at least one candidate is required")

    quality_percentiles = [percentile_from_rank(rank, n) for rank in quality_ranks]
    novelty_ranks = descending_fractional_ranks(novelty_scores)
    novelty_percentiles = [percentile_from_rank(rank, n) for rank in novelty_ranks]
    combined_scores = [
        quality_weight * quality + novelty_weight * novelty
        for quality, novelty in zip(quality_percentiles, novelty_percentiles)
    ]

    # The combined rank is unique and deterministic.  Quality, novelty, and
    # original slot order break the rare equal-score tie, in that order.
    ordered = sorted(
        range(n),
        key=lambda index: (
            -combined_scores[index],
            quality_ranks[index],
            -novelty_scores[index],
            index,
        ),
    )
    selection_ranks = [0] * n
    for rank, index in enumerate(ordered, 1):
        selection_ranks[index] = rank

    return SelectionMetrics(
        quality_percentiles=quality_percentiles,
        novelty_ranks=novelty_ranks,
        novelty_percentiles=novelty_percentiles,
        combined_scores=combined_scores,
        selection_ranks=selection_ranks,
    )


class HashingMockEncoder:
    """Small deterministic encoder used only by the offline integration test."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str], **kwargs) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append(
                [value / norm for value in vector]
                if norm
                else vector
            )
        return vectors


class SemanticNovelty:
    """Embed plans by component and compare them with a fixed seed archive."""

    def __init__(
        self,
        seed_ids: Sequence[str],
        seed_plans: Sequence[str],
        *,
        model_name: str,
        model_revision: str,
        device: str = "cpu",
        encoder: TextEncoder | None = None,
        mock: bool = False,
    ) -> None:
        if len(seed_ids) != len(seed_plans) or not seed_plans:
            raise ValueError("seed ids and plans must be non-empty and aligned")
        self.seed_ids = list(seed_ids)
        self.seed_plans = list(seed_plans)
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = device
        self.backend = "mock-hash" if mock and encoder is None else "sentence-transformers"
        self._encoder = encoder or (
            HashingMockEncoder()
            if mock
            else self._load_sentence_transformer()
        )
        if encoder is not None:
            self.backend = type(encoder).__name__
        self._cache: dict[str, list[list[float]]] = {}
        self._seed_vectors = self.embed_plans(self.seed_plans)

    def _load_sentence_transformer(self) -> TextEncoder:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic novelty requires sentence-transformers. "
                "Install the packages listed in src/requirements.txt."
            ) from exc
        return SentenceTransformer(
            self.model_name,
            revision=self.model_revision,
            device=self.device,
        )

    def embed_plans(self, plans: Sequence[str]) -> list[list[list[float]]]:
        """Return component vectors in ``COMPONENTS`` order, with caching."""
        missing_plans = [plan for plan in plans if plan not in self._cache]
        if missing_plans:
            texts = []
            for plan in missing_plans:
                projected = component_texts(plan)
                texts.extend(projected[component] for component in COMPONENTS)
            encoded = self._encoder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=False,
            )
            encoded_lists = [list(map(float, vector)) for vector in encoded]
            width = len(COMPONENTS)
            for offset, plan in enumerate(missing_plans):
                start = offset * width
                self._cache[plan] = encoded_lists[start:start + width]
        return [self._cache[plan] for plan in plans]

    @staticmethod
    def component_distances(
        vectors_a: Sequence[Sequence[float]],
        vectors_b: Sequence[Sequence[float]],
    ) -> list[float]:
        """Return cosine distance for each component."""
        if len(vectors_a) != len(COMPONENTS) or len(vectors_b) != len(COMPONENTS):
            raise ValueError(f"each plan must have {len(COMPONENTS)} component vectors")
        distances = []
        for vector_a, vector_b in zip(vectors_a, vectors_b):
            dot = sum(a * b for a, b in zip(vector_a, vector_b))
            norm_a = math.sqrt(sum(a * a for a in vector_a))
            norm_b = math.sqrt(sum(b * b for b in vector_b))
            similarity = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
            distances.append(1.0 - max(-1.0, min(1.0, similarity)))
        return distances

    @classmethod
    def component_distance(
        cls,
        vectors_a: Sequence[Sequence[float]],
        vectors_b: Sequence[Sequence[float]],
    ) -> float:
        """Mean component-wise cosine distance for two projected plans."""
        distances = cls.component_distances(vectors_a, vectors_b)
        return sum(distances) / len(distances)

    def distance(self, plan_a: str, plan_b: str) -> float:
        vectors_a, vectors_b = self.embed_plans([plan_a, plan_b])
        return self.component_distance(vectors_a, vectors_b)

    def novelty_scores(self, plans: Sequence[str]) -> list[float]:
        """Distance from each plan to its nearest original seed."""
        return [float(detail["novelty"]) for detail in self.novelty_details(plans)]

    def novelty_details(self, plans: Sequence[str]) -> list[dict[str, Any]]:
        """Return aggregate and component-level distances to the seed archive.

        ``nearest_seed_id`` minimizes the mean distance over all six components,
        matching the novelty construct used for global selection.  Each
        component also records its own closest seed.  Those per-component
        labels are descriptive semantic matches, not claims of genealogical
        provenance.
        """
        plan_vectors = self.embed_plans(plans)
        details: list[dict[str, Any]] = []
        for candidate_vectors in plan_vectors:
            seed_component_distances = [
                self.component_distances(candidate_vectors, seed_vectors)
                for seed_vectors in self._seed_vectors
            ]
            seed_mean_distances = [
                sum(distances) / len(distances)
                for distances in seed_component_distances
            ]
            nearest_seed_index = min(
                range(len(self.seed_ids)),
                key=lambda index: (seed_mean_distances[index], index),
            )
            component_nearest_indices = [
                min(
                    range(len(self.seed_ids)),
                    key=lambda seed_index: (
                        seed_component_distances[seed_index][component_index],
                        seed_index,
                    ),
                )
                for component_index in range(len(COMPONENTS))
            ]
            details.append(
                {
                    "novelty": seed_mean_distances[nearest_seed_index],
                    "nearest_seed_id": self.seed_ids[nearest_seed_index],
                    "component_distances": {
                        component: seed_component_distances[nearest_seed_index][component_index]
                        for component_index, component in enumerate(COMPONENTS)
                    },
                    "component_nearest_seed_ids": {
                        component: self.seed_ids[component_nearest_indices[component_index]]
                        for component_index, component in enumerate(COMPONENTS)
                    },
                    "component_nearest_distances": {
                        component: seed_component_distances[
                            component_nearest_indices[component_index]
                        ][component_index]
                        for component_index, component in enumerate(COMPONENTS)
                    },
                }
            )
        return details

