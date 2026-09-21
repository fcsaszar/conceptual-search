"""Frozen pairwise-evaluation prompt used by the robustness check.

This is an exact snapshot of EVALUATION_CRITERIA and EVALUATOR_PROMPT from
../src/local_search_prompts.py.  Keeping the snapshot here makes the check
self-contained while prepare_inputs.py verifies that the two copies agree.
"""

EVALUATION_CRITERIA = """\
1. Clarity of value proposition
2. Market size and demand
3. Feasibility and development plan
4. Revenue model and sustainability
5. Differentiation
6. Impact and evidence plan
7. Risk awareness and mitigation"""

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


def render_prompt(plan_a: str, plan_b: str) -> str:
    """Render the frozen evaluator prompt for one ordered pair."""
    return EVALUATOR_PROMPT.format(
        criteria=EVALUATION_CRITERIA,
        plan_a=plan_a,
        plan_b=plan_b,
    )
