"""Prompt templates specific to global mutation and crossover search."""


MUTATION_COMPONENT_GUIDANCE = {
    "problem": "the unmet need, pain point, or job to be done",
    "customer": "the focal user or buyer and the use context",
    "solution": "the product, service, or mechanism that addresses the problem",
    "delivery_model": "how the venture develops, provides, and supports the offering",
    "revenue_logic": "who pays, what they pay for, and how the venture can be sustained",
    "distinctiveness": "why this approach differs meaningfully from available alternatives",
}

MUTATOR_PROMPT = """\
You are an expert startup advisor. Here is a crowdfunding project description:

--- BEGIN PROJECT ---
{plan}
--- END PROJECT ---

The search algorithm selected this component for mutation:
{component}: {component_guidance}

Make one substantive change to that component only. Preserve the other five \
components and all source-grounded facts as closely as possible. Keep the \
existing Markdown headings so that the components remain machine-readable.

Rewrite the project description incorporating only that one improvement. \
The updated description should be approximately 500 words and must obey these rules:
{fidelity_rules}

Respond with a JSON object: {{"component": "{component}", "plan": "rewritten project description"}}
Output only valid JSON, nothing else."""

CROSSOVER_PROMPT = """\
You are an expert startup strategist. Here are two crowdfunding project descriptions:

--- PARENT A ---
{plan_a}
--- END PARENT A ---

--- PARENT B ---
{plan_b}
--- END PARENT B ---

Create a NEW crowdfunding project by RECOMBINING the two parents at the level of \
their six components: problem, customer, solution, delivery_model, \
revenue_logic, distinctiveness.

Recombination rules:
1. For each of the six components, choose exactly ONE parent as its source. Take at \
least two components from each parent. Use these criteria to decide which parent \
supplies the stronger version of each component:
{criteria}
2. Carry each chosen component's substantive DESIGN over faithfully: its approach, \
mechanism, target, framing, and reasoning. But the child is a brand-new venture \
proposal with no history of its own: do NOT carry claims of existing traction, \
customers, users, prototypes, development or launch status, production timelines, \
testimonials, partners, awards, or any other accomplished evidence from either \
parent. Where a source component's strength rests on such evidence, re-express the \
underlying idea as an explicitly proposed plan or target for the new venture (for \
example, "the venture plans a modular, tool-free-replacement architecture" rather \
than "each module can be replaced at home without tools").
3. Rewrite only the connective tissue -- the one-sentence summary and minimal \
transitional phrasing -- so the recombined description reads as ONE coherent new \
venture built from those components, approximately 500 words, not a patchwork.
4. Keep the source plans' Markdown headings so the six components remain \
machine-readable. The child must not be a restatement of either parent.

Treat the parents as the complete factual record and obey these rules:
{fidelity_rules}

Every statement about evidence, validation, partnerships, or performance in the \
child must be framed as proposed, planned, or to-be-tested -- never as an \
accomplishment inherited from a parent.

Respond with a JSON object: {{"component_map": {{"problem": "A" or "B", "customer": "A" or "B", "solution": "A" or "B", "delivery_model": "A" or "B", "revenue_logic": "A" or "B", "distinctiveness": "A" or "B"}}, "plan": "the new project description"}}
Output only valid JSON, nothing else."""

# Crossover children are NEW venture concepts assembled from two source
# projects, so the generic fidelity question ("is this supported by the
# sources?") misfires: an integrated concept is never "in" either source.
# This variant asks the narrow question the evidence standard actually poses:
# does the child assert any ALREADY-EXISTING accomplishment the sources do
# not support?
CROSSOVER_FIDELITY_AUDITOR_PROMPT = """\
You are auditing a proposed crowdfunding venture concept that was deliberately \
assembled by RECOMBINING components of the two source projects below. The \
recombined concept itself -- what the proposed product or service is, does, or \
would offer, its features, architecture, target users, pricing, positioning, \
and plans -- is ALLOWED to be new relative to the sources, even when written \
in present tense. Do not flag concept, feature, positioning, or plan \
statements for being absent from the sources.

--- AUTHORIZED SOURCE MATERIAL ---
{sources}
--- END AUTHORIZED SOURCE MATERIAL ---

--- GENERATED CANDIDATE ---
{candidate}
--- END GENERATED CANDIDATE ---

Your ONLY task: list every statement in the candidate that asserts an \
ALREADY-EXISTING real-world accomplishment or status of this new venture -- \
for example: built or working prototypes, completed development or \
production, units shipped or in transit, met dates, current users, customers, \
sales, revenue, growth, pilots or tests already run, measured results, \
testimonials, reviews, endorsements, certifications, awards, signed partners, \
or team credentials. The candidate is a brand-new venture proposal with no \
operating history, so ANY such assertion about this venture is unsupported \
and must be listed -- including accomplishments copied from a source project, \
because those belong to the source venture, not to this new one. Statements \
framed as proposed, planned, targeted, hypothetical, or to-be-tested are \
allowed and must not be listed. Apply this standard:
{fidelity_rules}

If the candidate asserts no already-existing accomplishments of its own, it \
passes.

Respond with a JSON object: {{"passed": true or false, "unsupported_claims": \
["exact or concise claim", ...], "analysis": "brief explanation"}}
Output only valid JSON, nothing else."""


NARRATIVE_MUTATION_PROMPT = """\
Here is a parent crowdfunding project plan:

--- PARENT ---
{parent_plan}
--- END PARENT ---

Here is the mutated child plan, which was modified along the "{dimension}" dimension:

--- CHILD ---
{child_plan}
--- END CHILD ---

In 2-3 sentences, describe how the mutation changed the plan along the \
{dimension} dimension. Focus on what specifically was added, removed, or altered."""

NARRATIVE_CROSSOVER_PROMPT = """\
Here are two parent crowdfunding project plans:

--- PARENT A ({a_id}) ---
{a_plan}
--- END PARENT A ---

--- PARENT B ({b_id}) ---
{b_plan}
--- END PARENT B ---

Here is the child plan produced by crossing the two parents:

--- CHILD ---
{child_plan}
--- END CHILD ---

In 2-3 sentences, describe what the child inherited from each parent. \
Identify which elements came from {a_id} and which from {b_id}."""
