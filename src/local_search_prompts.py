"""Shared evaluation, generation, fidelity, and narration prompts."""

EVALUATION_CRITERIA = """\
1. Clarity of value proposition
2. Market size and demand
3. Feasibility and development plan
4. Revenue model and sustainability
5. Differentiation
6. Impact and evidence plan
7. Risk awareness and mitigation"""

FACTUAL_FIDELITY_RULES = """\
Treat the supplied project description(s) as the complete factual record. You may propose \
new features, markets, revenue models, tests, partnerships, targets, projections, or plans \
only when they are explicitly framed as proposed, hypothetical, planned, or to be tested. \
Do not invent or imply existing traction, customers, users, sales, revenue, growth, pilots, \
tests, testimonials, partners, endorsements, certifications, awards, prototypes, development \
status, launch status, team credentials, dates, or quantified evidence. Do not convert a \
recommendation or aspiration into an accomplished fact. Preserve any factual claims that \
are carried forward from a source without strengthening them."""

GENERATOR_PROMPT = """\
You are an expert startup advisor. Below is a crowdfunding project description:

--- BEGIN PROJECT ---
{plan}
--- END PROJECT ---

Evaluate this project along these dimensions:
{criteria}

Suggest exactly {num_ideas} concrete, actionable improvement ideas that would make this \
project more compelling to potential backers. Each idea should propose a *small, \
targeted change* (not a complete rewrite). Suggestions must obey these rules:
{fidelity_rules}

Respond with a JSON object: {{"ideas": ["idea 1", "idea 2", ...]}}
Output only valid JSON, nothing else."""

SPECIALIZED_GENERATOR_PROMPT = """\
You are an expert startup copywriter. Below is a crowdfunding project description \
followed by one specific improvement idea.

--- BEGIN PROJECT ---
{plan}
--- END PROJECT ---

--- IMPROVEMENT IDEA ---
{idea}
--- END IDEA ---

Rewrite the project description incorporating the improvement idea above. \
Keep the overall structure and tone, but integrate the improvement naturally. \
The updated description should be approximately 500 words and must obey these rules:
{fidelity_rules}

Respond with a JSON object: {{"plan": "updated project description", \
"evidence_notes": "brief note identifying any proposed or source-supported claims"}}
Output only valid JSON, nothing else."""

EVALUATOR_PROMPT = """\
You are an expert venture capital analyst evaluating crowdfunding projects. \
Compare the two project descriptions below along these criteria:
{criteria}

--- PROJECT A ---
{plan_a}
--- END PROJECT A ---

--- PROJECT B ---
{plan_b}
--- END PROJECT B ---

For each project, list its pros and cons as bullet points (use "•" as the bullet marker), \
grounded in the criteria above. Do not reward unsupported specificity: claims of traction, \
partnerships, testimonials, completed development, or quantified evidence should improve a \
project's assessment only when they are genuine source facts. Then state which project wins \
overall and why.

Respond with a JSON object: {{"analysis": "your analysis here", "winner": "A" or "B"}}
Output only valid JSON, nothing else."""

FIDELITY_AUDITOR_PROMPT = """\
You are auditing factual fidelity in a generated crowdfunding venture concept.

--- AUTHORIZED SOURCE MATERIAL ---
{sources}
--- END AUTHORIZED SOURCE MATERIAL ---

--- GENERATED CANDIDATE ---
{candidate}
--- END GENERATED CANDIDATE ---

Apply this standard:
{fidelity_rules}

Pass the candidate only if every statement about existing real-world evidence or status is \
supported by the authorized source material. New ideas are allowed when clearly framed as \
proposals, plans, targets, hypotheses, or future tests. Unsupported claims include invented \
traction, customers, users, sales, revenue, metrics, pilots, tests, testimonials, partners, \
endorsements, certifications, awards, prototypes, development or launch status, team \
credentials, dates, and quantified evidence.

Apply the tense distinction literally. Specificity does not turn a future proposal into an \
existing fact. For example, "we plan to test with 50 participants," "we propose tracking \
six-month repeat donations," and "a future pilot would target three partners" are allowed \
new ideas even when those targets are absent from the source. Reject only when the candidate \
states or implies that such evidence, activity, status, or relationship already exists. Do \
not reject a candidate merely because a proposed plan is new, detailed, or quantified. \
When every addition is explicitly prospective, the candidate must pass.

Distinguish the venture's proposed concept from claims of accomplished evidence. A \
crowdfunding description's account of what the product or service is, does, or would \
offer -- its features, architecture, target users, pricing, or positioning -- is the \
proposal itself, even when written in present tense, and may legitimately be new relative \
to the sources, including a concept assembled from components of multiple source projects. \
Do not reject concept descriptions for being absent from the sources. Reject only \
assertions of existing real-world accomplishments or status -- built prototypes, completed \
production, met shipping dates, current users, customers, revenue, testimonials, \
partnerships, awards, or measured results -- that the sources do not support.

Respond with a JSON object: {{"passed": true or false, \
"unsupported_claims": ["exact or concise claim", ...], \
"analysis": "brief explanation"}}
Output only valid JSON, nothing else."""

NARRATIVE_DIFF_PROMPT = """\
Below are two versions of a crowdfunding project description.

--- OLD VERSION ---
{old_plan}
--- END OLD VERSION ---

--- NEW VERSION ---
{new_plan}
--- END NEW VERSION ---

Summarize in 2-3 sentences what changed between the old and new version. \
Focus on the key differences in content, emphasis, or strategy. \
Output only the summary, nothing else."""
