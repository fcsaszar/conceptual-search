#!/usr/bin/env python3
"""Run the resumable Qwen cross-evaluator tournament."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any

from openai import AsyncOpenAI

from prompt import EVALUATION_CRITERIA, EVALUATOR_PROMPT, render_prompt


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "qwen/qwen-2.5-72b-instruct"
DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"
DEFAULT_PROVIDER = "DeepInfra"
MODEL_WEIGHTS_REVISION = "495f39366efef23836d0cfae4fbe635880d2be31"
TASK_FIELDS = [
    "task_id",
    "bucket",
    "category",
    "direction",
    "plan_a_slot",
    "plan_b_slot",
    "plan_a_text_sha256",
    "plan_b_text_sha256",
    "generated_text_sha256",
    "reference_id",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_concepts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 75:
        raise ValueError(f"Expected 75 concept slots, found {len(rows)}")
    return rows


def task_id(payload: dict[str, str], model: str, temperature: float, seed: int) -> str:
    identity = {
        "schema": 1,
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "prompt_template": EVALUATOR_PROMPT,
        "criteria": EVALUATION_CRITERIA,
        **payload,
    }
    packed = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()[:24]


def build_tasks(
    concepts: list[dict[str, str]], model: str, temperature: float, seed: int
) -> tuple[list[dict[str, str]], dict[str, str]]:
    originals = [row for row in concepts if row["group"] == "original"]
    generated_by_hash: dict[str, dict[str, str]] = {}
    group_by_hash: dict[str, set[str]] = defaultdict(set)
    slots_by_hash: dict[str, list[str]] = defaultdict(list)
    texts: dict[str, str] = {}
    for row in concepts:
        texts[row["text_sha256"]] = row["plan_text"]
        if row["group"] != "original":
            generated_by_hash.setdefault(row["text_sha256"], row)
            group_by_hash[row["text_sha256"]].add(row["group"])
            slots_by_hash[row["text_sha256"]].append(row["slot_id"])
    if len(originals) != 30 or len(generated_by_hash) != 44:
        raise ValueError("Input sample must contain 30 originals and 44 distinct generated texts")

    tasks: list[dict[str, str]] = []

    def add_task(
        *,
        bucket: str,
        category: str,
        direction: str,
        plan_a: dict[str, str],
        plan_b: dict[str, str],
        generated_hash: str = "",
        reference_id: str = "",
    ) -> None:
        base = {
            "bucket": bucket,
            "category": category,
            "direction": direction,
            "plan_a_slot": plan_a["slot_id"],
            "plan_b_slot": plan_b["slot_id"],
            "plan_a_text_sha256": plan_a["text_sha256"],
            "plan_b_text_sha256": plan_b["text_sha256"],
            "generated_text_sha256": generated_hash,
            "reference_id": reference_id,
        }
        tasks.append({"task_id": task_id(base, model, temperature, seed), **base})

    for plan_a in originals:
        for plan_b in originals:
            if plan_a["slot_id"] == plan_b["slot_id"]:
                continue
            add_task(
                bucket="original_original",
                category="original_original",
                direction="ordered",
                plan_a=plan_a,
                plan_b=plan_b,
            )

    for text_hash, representative in sorted(generated_by_hash.items()):
        groups = sorted(group_by_hash[text_hash])
        bucket = "generated_" + "_and_".join(groups)
        generated = dict(representative)
        generated["slot_id"] = "|".join(sorted(slots_by_hash[text_hash]))
        for original in originals:
            reference_id = original["lineage_id"]
            add_task(
                bucket=bucket,
                category="generated_reference",
                direction="generated_first",
                plan_a=generated,
                plan_b=original,
                generated_hash=text_hash,
                reference_id=reference_id,
            )
            add_task(
                bucket=bucket,
                category="generated_reference",
                direction="original_first",
                plan_a=original,
                plan_b=generated,
                generated_hash=text_hash,
                reference_id=reference_id,
            )

    if len(tasks) != 3510 or len({row["task_id"] for row in tasks}) != 3510:
        raise ValueError(f"Expected 3,510 distinct tasks, found {len(tasks)}")
    return tasks, texts


def select_pilot(tasks: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0 or limit >= len(tasks):
        return tasks
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in tasks:
        buckets[row["bucket"]].append(row)
    selected: list[dict[str, str]] = []
    positions = {key: 0 for key in buckets}
    keys = sorted(buckets)
    while len(selected) < limit:
        moved = False
        for key in keys:
            position = positions[key]
            if position < len(buckets[key]) and len(selected) < limit:
                selected.append(buckets[key][position])
                positions[key] += 1
                moved = True
        if not moved:
            break
    return selected


def write_tasks(path: Path, tasks: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TASK_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(tasks)


def strip_fences(text: str) -> tuple[str, str]:
    fenced = re.match(r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$", text, re.DOTALL)
    if fenced:
        return fenced.group(1), "markdown_fence"
    return text, "exact"


def parse_content(content: str) -> tuple[str, str, str]:
    candidate, parse_mode = strip_fences(content)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as first_error:
        start, end = candidate.find("{"), candidate.rfind("}")
        try:
            if start < 0 or end <= start:
                raise first_error
            payload = json.loads(candidate[start : end + 1])
            parse_mode = "extracted_json_object"
        except json.JSONDecodeError:
            # Some models supply an unambiguous winner field inside a
            # malformed analysis object (for example, it writes list items as
            # object members). Recover only the explicit winner value; never
            # infer a winner from the surrounding prose.
            matches = re.findall(
                r'["“”\']winner["“”\']\s*:\s*["“”\'](?:project\s+)?([ab])["“”\']',
                candidate,
                flags=re.IGNORECASE,
            )
            unique = {match.upper() for match in matches}
            if len(unique) != 1:
                raise first_error
            return unique.pop(), content, "explicit_winner_recovered_from_malformed_json"
    if not isinstance(payload, dict):
        raise ValueError("Response JSON is not an object")
    winner = str(payload.get("winner", "")).strip().upper()
    project_match = re.fullmatch(r"PROJECT\s+([AB])", winner)
    if project_match:
        winner = project_match.group(1)
        parse_mode = "normalized_project_winner"
    if winner not in {"A", "B"}:
        raise ValueError(f"Invalid or missing winner: {winner!r}")
    analysis = str(payload.get("analysis", ""))
    return winner, analysis, parse_mode


def response_usage(response_payload: dict[str, Any]) -> tuple[int, int, float]:
    usage = response_payload.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    cost = float(usage.get("cost") or 0.0)
    return prompt_tokens, completion_tokens, cost


def response_content(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices") or []
    if not choices:
        raise ValueError("Provider response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Provider response has no textual content")
    return content


def existing_attempts(raw_task_dir: Path) -> int:
    if not raw_task_dir.exists():
        return 0
    return len(list(raw_task_dir.glob("attempt-*.json")))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


@dataclass
class Counters:
    completed: int = 0
    completed_this_run: int = 0
    failed: int = 0
    api_failures: int = 0
    parse_failures: int = 0
    attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    consecutive_failures: int = 0


async def main(args: argparse.Namespace) -> None:
    if not args.dry_run and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required for live calls")

    concepts_path = HERE / "inputs" / "concepts.csv"
    sample_manifest_path = HERE / "inputs" / "sample-manifest.json"
    if not concepts_path.exists() or not sample_manifest_path.exists():
        raise SystemExit("Run prepare_inputs.py before this command")
    concepts = read_concepts(concepts_path)
    all_tasks, texts = build_tasks(concepts, args.model, args.temperature, args.seed)
    tasks = select_pilot(all_tasks, args.limit)

    run_dir = HERE / "outputs" / args.run_id
    raw_dir = run_dir / "raw"
    parsed_dir = run_dir / "parsed"
    locks_dir = run_dir / "locks"
    logs_dir = run_dir / "logs"
    for directory in (raw_dir, parsed_dir, locks_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    tasks_path = run_dir / "tasks.csv"
    manifest_path = run_dir / "manifest.json"
    log_path = logs_dir / "run.log"
    write_tasks(tasks_path, tasks)

    prompt_payload = json.dumps(
        {"criteria": EVALUATION_CRITERIA, "template": EVALUATOR_PROMPT},
        ensure_ascii=False,
        sort_keys=True,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "running",
        "run_id": args.run_id,
        "started_at_utc": utc_now(),
        "command": " ".join(sys.argv),
        "working_directory": str(Path.cwd()),
        "git_commit_at_start": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "sample_manifest_sha256": sha256_file(sample_manifest_path),
        "concepts_csv_sha256": sha256_file(concepts_path),
        "tasks_csv_sha256": sha256_file(tasks_path),
        "prompt_sha256": sha256_text(prompt_payload),
        "model": {
            "openrouter_id": args.model,
            "hugging_face_id": "Qwen/Qwen2.5-72B-Instruct",
            "weights_revision": MODEL_WEIGHTS_REVISION,
            "knowledge_cutoff": "2024-06-30",
            "provider_endpoint": args.endpoint,
            "provider_route": args.provider,
        },
        "inference": {
            "temperature": args.temperature,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "timeout_seconds": args.timeout,
            "max_concurrency": args.max_concurrency,
            "max_attempts_per_task": args.max_attempts,
            "reasoning": "not requested",
            "response_format": "prompt-requested JSON; no API response_format constraint",
        },
        "failure_budgets": {
            "maximum_cost_usd": args.max_cost,
            "maximum_final_task_failure_rate": args.max_failure_rate,
            "maximum_parse_failure_rate_after_100_attempts": args.max_parse_failure_rate,
            "maximum_consecutive_final_task_failures": args.max_consecutive_failures,
        },
        "task_count": len(tasks),
        "full_design_task_count": len(all_tasks),
        "pilot_limit": args.limit,
        "environment_variables_required": ["OPENROUTER_API_KEY"],
        "artifacts": {
            "raw": "raw/<task-id>/attempt-NNN.json",
            "parsed": "parsed/<task-id>.json",
            "log": "logs/run.log",
        },
    }
    atomic_json(manifest_path, manifest)

    def log(message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(
        f"Prepared {len(tasks):,} tasks ({len(all_tasks):,} in the full design); "
        f"model={args.model}, concurrency={args.max_concurrency}."
    )
    if args.dry_run:
        log("Dry run complete; no API calls were made.")
        return

    client = AsyncOpenAI(
        base_url=args.endpoint,
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=args.timeout,
    )
    counters = Counters()
    for task in tasks:
        if (parsed_dir / f"{task['task_id']}.json").exists():
            counters.completed += 1
    initial_completed = counters.completed
    queue: asyncio.Queue[dict[str, str]] = asyncio.Queue()
    for task in tasks:
        if not (parsed_dir / f"{task['task_id']}.json").exists():
            queue.put_nowait(task)
    stop_event = asyncio.Event()
    counter_lock = asyncio.Lock()
    t0 = time.monotonic()

    async def acquire_lock(task_key: str) -> Path | None:
        lock_path = locks_dir / f"{task_key}.lock"
        if lock_path.exists() and time.time() - lock_path.stat().st_mtime > args.stale_lock_seconds:
            lock_path.unlink()
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()} created={utc_now()}\n")
        return lock_path

    async def run_task(task: dict[str, str]) -> tuple[bool, dict[str, int | float]]:
        parsed_path = parsed_dir / f"{task['task_id']}.json"
        if parsed_path.exists():
            return True, {"attempts": 0, "api_failures": 0, "parse_failures": 0, "cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
        lock_path = await acquire_lock(task["task_id"])
        if lock_path is None:
            return False, {"attempts": 0, "api_failures": 0, "parse_failures": 0, "cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
        stats: dict[str, int | float] = {
            "attempts": 0,
            "api_failures": 0,
            "parse_failures": 0,
            "cost": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        raw_task_dir = raw_dir / task["task_id"]
        raw_task_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Re-parse saved responses before making another paid call. This
            # supports parser improvements and makes raw responses a genuine
            # resumability layer rather than an audit-only archive.
            for saved_attempt in sorted(raw_task_dir.glob("attempt-*.json"), reverse=True):
                saved_payload = json.loads(saved_attempt.read_text(encoding="utf-8"))
                response_payload = saved_payload.get("response")
                if not isinstance(response_payload, dict):
                    continue
                try:
                    content = response_content(response_payload)
                    winner, analysis, parse_mode = parse_content(content)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                prompt_tokens, completion_tokens, cost = response_usage(response_payload)
                parsed_payload = {
                    "schema_version": 1,
                    "task_id": task["task_id"],
                    "attempt": int(saved_payload.get("attempt", 0)),
                    "winner": winner,
                    "analysis": analysis,
                    "parse_mode": parse_mode,
                    "model_requested": args.model,
                    "model_returned": response_payload.get("model", ""),
                    "provider": response_payload.get("provider", ""),
                    "response_id": response_payload.get("id", ""),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "reported_cost_usd": cost,
                    "raw_response": str(saved_attempt.relative_to(run_dir)),
                    "completed_at_utc": utc_now(),
                    "reparsed_saved_response": True,
                }
                atomic_json(parsed_path, parsed_payload)
                return True, stats
            start_attempt = existing_attempts(raw_task_dir) + 1
            for attempt in range(start_attempt, args.max_attempts + 1):
                if stop_event.is_set():
                    break
                prompt = render_prompt(
                    texts[task["plan_a_text_sha256"]],
                    texts[task["plan_b_text_sha256"]],
                )
                request = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "seed": args.seed,
                }
                if args.provider:
                    request["extra_body"] = {
                        "provider": {
                            "order": [args.provider],
                            "allow_fallbacks": False,
                        }
                    }
                attempt_path = raw_task_dir / f"attempt-{attempt:03d}.json"
                call_started = time.monotonic()
                stats["attempts"] = int(stats["attempts"]) + 1
                try:
                    response = await client.chat.completions.create(**request)
                    response_payload = response.model_dump(mode="json")
                    raw_payload = {
                        "schema_version": 1,
                        "task_id": task["task_id"],
                        "attempt": attempt,
                        "requested_at_utc": utc_now(),
                        "elapsed_seconds": time.monotonic() - call_started,
                        "request": request,
                        "response": response_payload,
                    }
                    atomic_json(attempt_path, raw_payload)
                    prompt_tokens, completion_tokens, cost = response_usage(response_payload)
                    stats["prompt_tokens"] = int(stats["prompt_tokens"]) + prompt_tokens
                    stats["completion_tokens"] = int(stats["completion_tokens"]) + completion_tokens
                    stats["cost"] = float(stats["cost"]) + cost
                    try:
                        content = response_content(response_payload)
                        winner, analysis, parse_mode = parse_content(content)
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                        stats["parse_failures"] = int(stats["parse_failures"]) + 1
                        raw_payload["parse_error"] = f"{type(exc).__name__}: {exc}"
                        atomic_json(attempt_path, raw_payload)
                    else:
                        parsed_payload = {
                            "schema_version": 1,
                            "task_id": task["task_id"],
                            "attempt": attempt,
                            "winner": winner,
                            "analysis": analysis,
                            "parse_mode": parse_mode,
                            "model_requested": args.model,
                            "model_returned": response_payload.get("model", ""),
                            "provider": response_payload.get("provider", ""),
                            "response_id": response_payload.get("id", ""),
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "reported_cost_usd": cost,
                            "raw_response": str(attempt_path.relative_to(run_dir)),
                            "completed_at_utc": utc_now(),
                        }
                        atomic_json(parsed_path, parsed_payload)
                        return True, stats
                except Exception as exc:  # provider and transport failures are preserved
                    stats["api_failures"] = int(stats["api_failures"]) + 1
                    atomic_json(
                        attempt_path,
                        {
                            "schema_version": 1,
                            "task_id": task["task_id"],
                            "attempt": attempt,
                            "requested_at_utc": utc_now(),
                            "elapsed_seconds": time.monotonic() - call_started,
                            "request": request,
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                        },
                    )
                if attempt < args.max_attempts:
                    await asyncio.sleep(min(2 ** (attempt - 1), 8))
            return False, stats
        finally:
            if lock_path.exists():
                lock_path.unlink()

    async def worker(worker_id: int) -> None:
        while not stop_event.is_set():
            try:
                task = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            success, stats = await run_task(task)
            async with counter_lock:
                counters.attempts += int(stats["attempts"])
                counters.api_failures += int(stats["api_failures"])
                counters.parse_failures += int(stats["parse_failures"])
                counters.prompt_tokens += int(stats["prompt_tokens"])
                counters.completion_tokens += int(stats["completion_tokens"])
                counters.cost_usd += float(stats["cost"])
                if success:
                    counters.completed += 1
                    counters.completed_this_run += 1
                    counters.consecutive_failures = 0
                else:
                    counters.failed += 1
                    counters.consecutive_failures += 1
                processed = counters.completed + counters.failed
                elapsed = max(time.monotonic() - t0, 0.001)
                done_now = counters.completed - initial_completed + counters.failed
                rate = done_now / elapsed * 60
                remaining = max(len(tasks) - processed, 0)
                eta_minutes = remaining / rate if rate > 0 else float("inf")
                if done_now == 1 or done_now % args.progress_every == 0 or processed == len(tasks):
                    log(
                        f"progress completed={counters.completed:,}/{len(tasks):,} "
                        f"failed={counters.failed} attempts={counters.attempts:,} "
                        f"parse_failures={counters.parse_failures} api_failures={counters.api_failures} "
                        f"rate={rate:.1f}/min eta={eta_minutes:.1f}min cost=${counters.cost_usd:.4f}"
                    )
                final_failure_rate = counters.failed / max(done_now, 1)
                parse_failure_rate = counters.parse_failures / max(counters.attempts, 1)
                reason = ""
                if counters.cost_usd > args.max_cost:
                    reason = f"cost cap exceeded (${counters.cost_usd:.4f} > ${args.max_cost:.2f})"
                elif counters.consecutive_failures >= args.max_consecutive_failures:
                    reason = f"{counters.consecutive_failures} consecutive tasks failed"
                elif done_now >= 100 and final_failure_rate > args.max_failure_rate:
                    reason = f"final task failure rate {final_failure_rate:.3f} exceeded budget"
                elif counters.attempts >= 100 and parse_failure_rate > args.max_parse_failure_rate:
                    reason = f"parse failure rate {parse_failure_rate:.3f} exceeded budget"
                if reason:
                    log(f"STOPPING: {reason}; the run remains resumable.")
                    manifest["stop_reason"] = reason
                    stop_event.set()
            queue.task_done()

    workers = [asyncio.create_task(worker(index)) for index in range(args.max_concurrency)]
    await asyncio.gather(*workers)
    elapsed = time.monotonic() - t0
    remaining_parsed = sum((parsed_dir / f"{task['task_id']}.json").exists() for task in tasks)
    manifest.update(
        {
            "status": "complete" if remaining_parsed == len(tasks) else "stopped_resumable",
            "finished_at_utc": utc_now(),
            "elapsed_seconds_this_invocation": elapsed,
            "results": {
                "completed_tasks": remaining_parsed,
                "completed_this_invocation": counters.completed_this_run,
                "failed_tasks_this_invocation": counters.failed,
                "attempts_this_invocation": counters.attempts,
                "api_failures_this_invocation": counters.api_failures,
                "parse_failures_this_invocation": counters.parse_failures,
                "prompt_tokens_this_invocation": counters.prompt_tokens,
                "completion_tokens_this_invocation": counters.completion_tokens,
                "reported_cost_usd_this_invocation": counters.cost_usd,
            },
        }
    )
    atomic_json(manifest_path, manifest)
    log(
        f"Run status={manifest['status']}; parsed={remaining_parsed:,}/{len(tasks):,}; "
        f"elapsed={elapsed / 60:.1f}min; reported cost this invocation=${counters.cost_usd:.4f}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        help="OpenRouter provider route; pass an empty string to use automatic routing.",
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-concurrency", type=int, default=40)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--max-cost", type=float, default=5.0)
    parser.add_argument("--max-failure-rate", type=float, default=0.01)
    parser.add_argument("--max-parse-failure-rate", type=float, default=0.10)
    parser.add_argument("--max-consecutive-failures", type=int, default=10)
    parser.add_argument("--stale-lock-seconds", type=int, default=21600)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="Stratified pilot size; zero runs all tasks.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_concurrency < 1 or args.max_attempts < 1 or args.max_tokens < 1:
        parser.error("Concurrency, attempts, and max tokens must be positive")
    if not 0 <= args.temperature <= 2:
        parser.error("Temperature must be between 0 and 2")
    return args


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
