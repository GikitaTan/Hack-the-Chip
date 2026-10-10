#!/usr/bin/env python3
"""Stage A prototype agent: generate, verify, repair, and measure NumPy kernels.

The verifier is kernelbench.py. This file owns only the agent loop:

    reference -> model -> candidate -> verifier -> repair instruction -> model

Examples:

    # Deterministic smoke test; no model or accelerator required.
    python stage_a_agent.py --offline --level 1

    # Local OpenAI-compatible server (for example the seat-pod Qwen server).
    python stage_a_agent.py --level 1 --base http://localhost:8000/v1 \
        --model Qwen/Qwen3-8B

    # Shared endpoint. Add the path expected by that endpoint.
    python stage_a_agent.py --level 1 --base "$GPTOSS_BASE_URL" \
        --path /agg/v1 --model gpt-oss-20b

Every attempt is appended to JSONL. The log includes the candidate, verifier result,
failure category, confidence, prompt/completion token counts, and token-budget split.
"""

import argparse
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter

import kernelbench as kb


# ---------------------------------------------------------------- configuration

DEFAULT_MODEL = os.environ.get("STAGE_A_MODEL", "Qwen/Qwen3-8B")
CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.S)
REASONING_KEYS = ("reasoning", "reasoning_content")


# ---------------------------------------------------------------- prompts

def first_prompt(level):
    """Keep the first prompt short; rules belong in the verifier."""
    spec = kb.LEVELS[level]
    return (
        "Write a NumPy function named `kernel` with the same signature and output as this "
        "reference function:\n\n"
        f"{inspect.getsource(spec['ref'])}\n"
        "Process the arrays with explicit loops over tiles of at most 128 rows and 512 columns. "
        "Correctly handle final partial tiles. Do not call the reference function. "
        "Return exactly one complete Python code block containing imports and `kernel`, with no prose."
    )


def repair_prompt(level, source, instruction, ledger):
    """Ask for one named change instead of resending the entire rules document."""
    tried = "\n".join(f"- {item}" for item in ledger[-4:])
    tried_section = f"\n\nPrevious failed fixes:\n{tried}" if tried else ""
    return (
        f"This NumPy kernel for {kb.LEVELS[level]['name']} is not verified yet:\n\n"
        f"```python\n{source}\n```\n\n"
        f"Make this specific change: {instruction}\n"
        "Keep unrelated code unchanged. Return exactly one complete Python code block and no prose."
        f"{tried_section}"
    )


def compact_regeneration_prompt(level, instruction):
    """Start fresh when carrying the previous kernel would exceed the context window."""
    return (
        first_prompt(level)
        + "\n\nA previous candidate failed. Incorporate this one correction in the new "
          f"implementation: {instruction}"
    )


def token_breakdown(level, prompt, source="", instruction="", ledger=None):
    """Estimate where input tokens went; exact totals still come from endpoint usage."""
    ledger = ledger or []
    reference = len(inspect.getsource(kb.LEVELS[level]["ref"])) // 4 if not source else 0
    code = len(source) // 4
    feedback = len(instruction) // 4
    history = sum(len(item) for item in ledger[-4:]) // 4
    total = max(1, len(prompt) // 4)
    fixed = reference + code + feedback + history
    return {
        "reference": reference,
        "candidate_code": code,
        "repair_instruction": feedback,
        "failure_ledger": history,
        "prompt_scaffolding": max(0, total - fixed),
    }


def extract_code(text):
    """Extract code without sending prose to Python's compiler."""
    text = text or ""
    blocks = CODE_BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^\s*(import |from |def )", line):
            return "\n".join(lines[index:]).strip()
    return ""


# ---------------------------------------------------------------- verifier and feedback distillation

def verify_source(source, level, timeout):
    """Run untrusted candidate code in a child process through kernelbench.py."""
    if not source.strip():
        verdict = "EMPTY ANSWER: no Python code was returned."
        return dict(ok=False, returncode=1, verdict=verdict, passed=0, total=0)

    with tempfile.TemporaryDirectory(prefix="stage-a-agent-") as tmp:
        candidate = os.path.join(tmp, "candidate.py")
        with open(candidate, "w", encoding="utf-8") as handle:
            handle.write(source)
        try:
            proc = subprocess.run(
                [sys.executable, os.path.join(os.path.dirname(__file__), "kernelbench.py"),
                 "--level", str(level), "--check", candidate],
                capture_output=True, text=True, timeout=timeout,
            )
            verdict = (proc.stdout or proc.stderr).strip()
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            verdict = f"TIMEOUT: candidate exceeded the {timeout}-second verification limit."
            returncode = 124

    match = re.search(r"(\d+)/(\d+) cases passed", verdict)
    passed, total = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
    return dict(ok=(returncode == 0 and "VERIFIED" in verdict), returncode=returncode,
                verdict=verdict, passed=passed, total=total)


def failure_category(verdict):
    """Small, stable taxonomy suitable for aggregate reporting."""
    lowered = verdict.lower()
    if "max accumulator was likely initialized to 0" in lowered:
        return "max_identity"
    if ("rmsnorm square accumulator overflowed" in lowered
            or "rmsnorm accumulation is inaccurate" in lowered):
        return "accumulator_precision"
    if "subtract each row maximum" in lowered:
        return "softmax_stability"
    if "contraction k tail" in lowered:
        return "matmul_k_tail"
    if "two-pass float64" in lowered:
        return "layernorm_cancellation"
    if "inclusive at both ends" in lowered:
        return "attention_window"
    if "effective_kernel" in lowered:
        return "convolution_geometry"
    if "partial row tile" in lowered or "partial column tile" in lowered:
        return "ragged_edge"
    checks = (
        ("verified", "VERIFIED"),
        ("empty_answer", "EMPTY ANSWER"),
        ("timeout", "TIMEOUT:"),
        ("rule_violation", "RULE VIOLATIONS"),
        ("missing_kernel", "defines no `kernel` function"),
        ("parse_or_import", "failed to import"),
        ("wrong_shape", "WRONG SHAPE"),
        ("non_finite", "NON-FINITE OUTPUT"),
        ("numerical", "NUMERICAL MISMATCH"),
        ("runtime", "RAISED:"),
    )
    for category, marker in checks:
        if marker in verdict:
            return category
    return "unknown"


def instruction_from_verdict(verdict):
    """Translate a verifier verdict into one prescriptive repair instruction."""
    lowered = verdict.lower()
    banned = re.search(r"calls banned `([^`]+)`", verdict)
    if not banned:
        banned = re.search(r"method `\.([^`(]+)\(\)` is banned", verdict)
    if banned:
        name = banned.group(1)
        return (f"Replace the banned `{name}` operation with explicit Python loops over the "
                "appropriate tile elements; preserve the existing tile boundaries and output shape.")
    if "fancy/boolean indexing is not allowed" in verdict:
        return ("Replace fancy or boolean indexing with ordinary slice boundaries and explicit "
                "loops; do not change the mathematical operation.")
    if "defines no `kernel` function" in verdict:
        return "Rename or add the entry function so it is exactly `def kernel(...):`."
    if "failed to import" in verdict:
        detail = verdict.split("failed to import:", 1)[-1].strip()
        return f"Fix the Python syntax or import error `{detail}` and return a complete module."
    if "WRONG SHAPE" in verdict:
        detail = next((line.strip() for line in verdict.splitlines() if "WRONG SHAPE" in line), "")
        return f"Correct only the output allocation and output-size/index formula. {detail}"
    if "max accumulator was likely initialized to 0" in lowered:
        return ("Initialize every row-maximum accumulator to `-np.inf`, not zero, so rows "
                "containing only negative values remain correct.")
    if ("rmsnorm square accumulator overflowed" in lowered
            or "rmsnorm accumulation is inaccurate" in lowered):
        return ("Accumulate every row's squared values in a float64 scalar across all column "
                "tiles, divide by the full column count, add eps, then normalize that row.")
    if "subtract each row maximum" in lowered:
        return ("Make softmax numerically stable: find each row maximum first, subtract it "
                "from every logit before exp, then normalize by the shifted exponential sum.")
    if "contraction k tail" in lowered:
        return ("Include the final partial K tile in matmul: iterate through all K values and "
                "bound the last chunk with min(k0 + 128, K).")
    if "two-pass float64" in lowered:
        return ("Compute LayerNorm variance in two float64 passes: first find the mean, then "
                "accumulate (x - mean)^2; do not use E[x^2] - E[x]^2.")
    if "inclusive at both ends" in lowered:
        return ("Make the attention window inclusive: for query i, include keys from max(0, i-w) "
                "through min(n-1, i+w), including the right endpoint.")
    if "effective_kernel" in lowered:
        return ("Use effective_kernel = dilation * (K - 1) + 1 and compute valid convolution "
                "length as (L - effective_kernel) // stride + 1.")
    if "NON-FINITE OUTPUT" in verdict:
        return ("Prevent overflow before exponentiation by subtracting each row's maximum, and "
                "ensure the normalization denominator cannot be zero.")
    if "partial row tile" in lowered or "partial column tile" in lowered:
        return ("Fix the final partial tile: derive each slice end with min(start + tile_size, "
                "dimension_size), and use those exact bounds for reads and writes.")
    if "most elements are wrong" in verdict:
        return ("Fix the core arithmetic inside a full tile. Re-check the reference formula, "
                "accumulation axes, and accumulator initialization; do not change tiling yet.")
    if "NUMERICAL MISMATCH" in verdict:
        detail = next((line.strip() for line in verdict.splitlines()
                       if "NUMERICAL MISMATCH" in line), "")
        return ("Correct the arithmetic at the reported location while preserving legal explicit "
                f"tile loops. {detail}")
    if "RAISED:" in verdict:
        detail = next((line.strip() for line in verdict.splitlines() if "RAISED:" in line), "")
        return f"Fix the runtime exception shown here, without bypassing tile loops: {detail}"
    if "TIMEOUT:" in verdict:
        return "Remove accidental unbounded work or non-terminating loops while preserving tiling."
    if "EMPTY ANSWER" in verdict:
        return "Return a complete Python code block containing imports and a `kernel` function."
    compact = " ".join(verdict.split())[:500]
    return f"Fix the verifier failure below with one minimal change: {compact}"


def confidence_from_result(result):
    """Confidence means verified coverage, never model self-confidence."""
    if result["ok"]:
        return 1.0
    if result["total"]:
        return result["passed"] / result["total"]
    return 0.0


# ---------------------------------------------------------------- model

def endpoint_url(base, path):
    root = base.rstrip("/") + path
    return root.rstrip("/") + "/chat/completions"


def ask_model(args, prompt):
    import httpx

    estimated_prompt_tokens = max(1, len(prompt) // 4)
    available = args.context - estimated_prompt_tokens - 64
    if available < 256:
        raise RuntimeError(
            f"prompt needs about {estimated_prompt_tokens} tokens, leaving only {available} "
            f"inside the {args.context}-token context window"
        )
    answer_budget = min(args.max_tokens, available)
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": answer_budget,
        "temperature": args.temperature,
        "top_p": 0.95,
    }
    # Thinking is off by default: both workshop endpoints were observed spending the entire
    # budget in hidden reasoning and returning no kernel. Keep --think only for controlled
    # comparisons, and record reasoning_chars so the failure remains visible.
    body["chat_template_kwargs"] = {"enable_thinking": args.thinking}

    response = httpx.post(endpoint_url(args.base, args.path), json=body,
                          timeout=args.request_timeout, verify=not args.insecure)
    if response.status_code != 200:
        raise RuntimeError(f"model endpoint returned HTTP {response.status_code}: "
                           f"{response.text[:500]}")
    payload = response.json()
    choice = payload["choices"][0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    reasoning = next((message.get(key) for key in REASONING_KEYS if message.get(key)), "")
    usage = payload.get("usage", {})
    return dict(
        content=content,
        reasoning_chars=len(reasoning),
        finish_reason=choice.get("finish_reason"),
        prompt_tokens=usage.get("prompt_tokens", estimated_prompt_tokens),
        completion_tokens=usage.get("completion_tokens", max(0, len(content) // 4)),
        estimated_prompt_tokens=estimated_prompt_tokens,
        answer_budget=answer_budget,
    )


# ---------------------------------------------------------------- deterministic offline smoke test

OFFLINE_GOOD = {
    1: """import numpy as np
def kernel(x, a, b):
    out = np.empty_like(x)
    rows, cols = x.shape
    for r0 in range(0, rows, 128):
        for c0 in range(0, cols, 512):
            r1, c1 = min(r0 + 128, rows), min(c0 + 512, cols)
            tile = x[r0:r1, c0:c1]
            y = a * tile + b
            out[r0:r1, c0:c1] = np.where(y > 0, y, 0.0)
    return out
""",
    2: """import numpy as np
def kernel(x):
    rows, cols = x.shape
    out = np.zeros(rows, dtype=x.dtype)
    for r0 in range(0, rows, 128):
        r1 = min(r0 + 128, rows)
        for r in range(r0, r1):
            acc = 0.0
            for c0 in range(0, cols, 512):
                c1 = min(c0 + 512, cols)
                for c in range(c0, c1):
                    acc += float(x[r, c])
            out[r] = acc
    return out
""",
    3: """import numpy as np
def kernel(x):
    rows, cols = x.shape
    out = np.empty(rows, dtype=x.dtype)
    for r0 in range(0, rows, 128):
        r1 = min(r0 + 128, rows)
        for r in range(r0, r1):
            best = -np.inf
            for c0 in range(0, cols, 512):
                c1 = min(c0 + 512, cols)
                for c in range(c0, c1):
                    if x[r, c] > best:
                        best = x[r, c]
            out[r] = best
    return out
""",
    4: """import numpy as np
def kernel(x, eps=1e-6):
    rows, cols = x.shape
    out = np.empty_like(x)
    for r0 in range(0, rows, 128):
        r1 = min(r0 + 128, rows)
        for r in range(r0, r1):
            square_sum = 0.0
            for c0 in range(0, cols, 512):
                c1 = min(c0 + 512, cols)
                for c in range(c0, c1):
                    value = float(x[r, c])
                    square_sum += value * value
            scale = np.sqrt(square_sum / cols + eps)
            for c0 in range(0, cols, 512):
                c1 = min(c0 + 512, cols)
                out[r, c0:c1] = x[r, c0:c1] / scale
    return out
""",
    5: """import numpy as np
def kernel(x):
    rows, cols = x.shape
    out = np.empty_like(x)
    for r0 in range(0, rows, 128):
        for r in range(r0, min(r0 + 128, rows)):
            best = -np.inf
            for c0 in range(0, cols, 512):
                for c in range(c0, min(c0 + 512, cols)):
                    if x[r, c] > best:
                        best = float(x[r, c])
            shifted = np.empty(cols, dtype=np.float64)
            denominator = 0.0
            for c0 in range(0, cols, 512):
                for c in range(c0, min(c0 + 512, cols)):
                    shifted[c] = np.exp(float(x[r, c]) - best)
                    denominator += shifted[c]
            for c0 in range(0, cols, 512):
                c1 = min(c0 + 512, cols)
                out[r, c0:c1] = shifted[c0:c1] / denominator
    return out
""",
    6: """import numpy as np
def kernel(x):
    rows, cols = x.shape
    out = np.empty((cols, rows), dtype=x.dtype)
    for r0 in range(0, rows, 128):
        r1 = min(r0 + 128, rows)
        for c0 in range(0, cols, 512):
            c1 = min(c0 + 512, cols)
            for r in range(r0, r1):
                for c in range(c0, c1):
                    out[c, r] = x[r, c]
    return out
""",
    7: """import numpy as np
def kernel(a, b):
    M, K = a.shape
    _, N = b.shape
    out = np.zeros((M, N), dtype=np.float32)
    for m0 in range(0, M, 128):
        for n0 in range(0, N, 512):
            for m in range(m0, min(m0 + 128, M)):
                for n in range(n0, min(n0 + 512, N)):
                    acc = 0.0
                    for k0 in range(0, K, 128):
                        for k in range(k0, min(k0 + 128, K)):
                            acc += float(a[m, k]) * float(b[k, n])
                    out[m, n] = acc
    return out
""",
    8: """import numpy as np
def kernel(x, eps=1e-5):
    rows, cols = x.shape
    out = np.empty_like(x)
    for r0 in range(0, rows, 128):
        for r in range(r0, min(r0 + 128, rows)):
            total = 0.0
            for c0 in range(0, cols, 512):
                for c in range(c0, min(c0 + 512, cols)):
                    total += float(x[r, c])
            mean = total / cols
            variance_total = 0.0
            for c0 in range(0, cols, 512):
                for c in range(c0, min(c0 + 512, cols)):
                    delta = float(x[r, c]) - mean
                    variance_total += delta * delta
            scale = np.sqrt(variance_total / cols + eps)
            for c0 in range(0, cols, 512):
                for c in range(c0, min(c0 + 512, cols)):
                    out[r, c] = (float(x[r, c]) - mean) / scale
    return out
""",
    9: """import numpy as np
def kernel(q, k, v, w):
    n, d = q.shape
    out = np.zeros_like(q)
    root_d = np.sqrt(float(d))
    for i in range(n):
        start, stop = max(0, i - w), min(n, i + w + 1)
        scores = np.empty(stop - start, dtype=np.float64)
        best = -np.inf
        for j in range(start, stop):
            score = 0.0
            for t in range(d):
                score += float(q[i, t]) * float(k[j, t])
            score /= root_d
            scores[j - start] = score
            if score > best:
                best = score
        denominator = 0.0
        for j in range(start, stop):
            scores[j - start] = np.exp(scores[j - start] - best)
            denominator += scores[j - start]
        for j in range(start, stop):
            weight = scores[j - start] / denominator
            for t in range(d):
                out[i, t] += weight * float(v[j, t])
    return out
""",
    10: """import numpy as np
def kernel(x, wt, stride, dilation):
    c_in, length = x.shape
    c_out, _, K = wt.shape
    effective_kernel = dilation * (K - 1) + 1
    out_length = (length - effective_kernel) // stride + 1
    out = np.zeros((c_out, out_length), dtype=np.float32)
    for o in range(out_length):
        for kk in range(K):
            pos = o * stride + kk * dilation
            for co in range(c_out):
                for ci in range(c_in):
                    out[co, o] += float(wt[co, ci, kk]) * float(x[ci, pos])
    return out
""",
}


def offline_reply(level, round_number):
    if level not in OFFLINE_GOOD:
        raise RuntimeError("offline smoke tests are available only for levels 1 through 10")
    if round_number == 0:
        return dict(content=f"```python\n{kb.PLANTED_BUGS[level]}\n```",
                    reasoning_chars=0, finish_reason="stop", prompt_tokens=0,
                    completion_tokens=max(1, len(kb.PLANTED_BUGS[level]) // 4),
                    estimated_prompt_tokens=0, answer_budget=0)
    return dict(content=f"```python\n{OFFLINE_GOOD[level]}\n```",
                reasoning_chars=0, finish_reason="stop", prompt_tokens=0,
                completion_tokens=max(1, len(OFFLINE_GOOD[level]) // 4),
                estimated_prompt_tokens=0, answer_budget=0)


# ---------------------------------------------------------------- loop and reporting

def solve_level(args, level, log_handle, run_number):
    prompt = first_prompt(level)
    prompt_parts = token_breakdown(level, prompt)
    ledger = []
    best_confidence = 0.0
    best_source = ""
    attempt_categories = []
    repeated_failures = Counter()
    token_totals = Counter()

    def outcome(ok, rounds, confidence, category, code):
        return dict(ok=ok, rounds=rounds, confidence=confidence, category=category,
                    code=code, attempt_categories=attempt_categories,
                    token_usage=dict(token_totals))

    print(f"\n=== level {level}: {kb.LEVELS[level]['name']} ===")
    for round_number in range(args.rounds):
        started = time.perf_counter()
        reply = (offline_reply(level, round_number) if args.offline
                 else ask_model(args, prompt))
        source = extract_code(reply["content"])
        result = verify_source(source, level, args.verify_timeout)
        category = failure_category(result["verdict"])
        attempt_categories.append(category)
        confidence = confidence_from_result(result)
        instruction = "" if result["ok"] else instruction_from_verdict(result["verdict"])
        elapsed = time.perf_counter() - started
        token_totals["prompt_tokens"] += int(reply["prompt_tokens"] or 0)
        token_totals["completion_tokens"] += int(reply["completion_tokens"] or 0)
        token_totals["reasoning_chars"] += int(reply["reasoning_chars"] or 0)
        token_totals["attempts"] += 1
        for part, count in prompt_parts.items():
            token_totals[f"input_{part}"] += int(count)

        if confidence > best_confidence or (result["ok"] and not best_source):
            best_confidence, best_source = confidence, source

        record = {
            "run": run_number,
            "level": level,
            "round": round_number,
            "ok": result["ok"],
            "confidence": confidence,
            "passed": result["passed"],
            "total": result["total"],
            "failure_category": category,
            "instruction": instruction,
            "verdict": result["verdict"],
            "code": source,
            "finish_reason": reply["finish_reason"],
            "reasoning_chars": reply["reasoning_chars"],
            "prompt_tokens": reply["prompt_tokens"],
            "completion_tokens": reply["completion_tokens"],
            "token_budget": {
                "prompt_estimate": reply["estimated_prompt_tokens"],
                "answer_budget": reply["answer_budget"],
                "context_limit": args.context,
                "input_breakdown_estimate": prompt_parts,
            },
            "elapsed_seconds": round(elapsed, 3),
        }
        log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_handle.flush()

        print(f"round {round_number}: category={category} confidence={confidence:.2f} "
              f"tokens={reply['prompt_tokens']}+{reply['completion_tokens']} "
              f"time={elapsed:.1f}s")
        if result["ok"]:
            print("  VERIFIED")
            return outcome(True, round_number + 1, 1.0, "verified", source)

        print(f"  next instruction: {instruction[:300]}")
        if reply["finish_reason"] == "length":
            print("  warning: answer was truncated; shorten the next prompt or increase context")

        ledger.append(f"{category}: {instruction[:180]}")
        failure_key = (category, instruction)
        repeated_failures[failure_key] += 1
        if repeated_failures[failure_key] >= args.give_up_after:
            print(f"  STOPPING: the same {category} repair failed "
                  f"{repeated_failures[failure_key]} times")
            return outcome(False, round_number + 1, best_confidence, category, best_source)
        if not source:
            prompt = first_prompt(level)
            prompt_parts = token_breakdown(level, prompt)
        else:
            prompt = repair_prompt(level, source, instruction, ledger)
            prompt_parts = token_breakdown(level, prompt, source, instruction, ledger)
            if len(prompt) // 4 > args.context - 320:
                prompt = compact_regeneration_prompt(level, instruction)
                prompt_parts = token_breakdown(level, prompt)
                print("  context budget: dropped the previous kernel and failure ledger; "
                      "requesting a fresh implementation with the distilled correction")

    print(f"  NOT VERIFIED after {args.rounds} rounds; best confidence={best_confidence:.2f}")
    category = ledger[-1].split(":", 1)[0] if ledger else "unknown"
    return outcome(False, args.rounds, best_confidence, category, best_source)


def build_summary(args, levels, history, categories):
    """Build the machine-readable experiment artifact required by the challenge."""
    token_totals = Counter()
    level_reports = []
    for level in levels:
        results = history[level]
        for result in results:
            token_totals.update(result["token_usage"])
        solved = sum(result["ok"] for result in results)
        level_reports.append({
            "level": level,
            "operation": kb.LEVELS[level]["name"],
            "trap": kb.LEVELS[level]["trap"],
            "runs": len(results),
            "verified": solved,
            "solve_rate": solved / len(results),
            "mean_confidence": sum(result["confidence"] for result in results) / len(results),
            "rounds": [result["rounds"] for result in results],
            "final_categories": [result["category"] for result in results],
        })
    failure_taxonomy = {key: value for key, value in sorted(categories.items())
                        if key != "verified"}
    return {
        "schema_version": "1.0",
        "mode": "offline_smoke_test" if args.offline else "model_evaluation",
        "model": None if args.offline else args.model,
        "context_limit": args.context,
        "tile_limit": {"rows": kb.TILE_ROWS, "columns": kb.TILE_COLS},
        "tolerance": {"rtol": kb.DEFAULT_RTOL, "atol": kb.DEFAULT_ATOL,
                      "reason": "float32 kernels with float64 reference accumulation; atol protects values near zero"},
        "all_verified": all(report["verified"] == report["runs"] for report in level_reports),
        "levels": level_reports,
        "failure_taxonomy": failure_taxonomy,
        "all_attempt_categories": dict(sorted(categories.items())),
        "token_instrumentation": dict(sorted(token_totals.items())),
        "attempt_log": args.log,
        "offline_disclaimer": ("Deterministic harness smoke test; not a model solve-rate result"
                               if args.offline else None),
    }


def parse_levels(args):
    if args.all:
        return list(sorted(kb.LEVELS))
    if args.levels:
        values = []
        for part in args.levels.split(","):
            values.append(int(part.strip()))
        invalid = [value for value in values if value not in kb.LEVELS]
        if invalid:
            raise SystemExit(f"unknown levels: {invalid}; choose from {sorted(kb.LEVELS)}")
        return values
    return [args.level]


def main():
    parser = argparse.ArgumentParser(description="Automatic Stage A NumPy kernel agent")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--level", type=int, default=1, choices=sorted(kb.LEVELS))
    choice.add_argument("--levels", help="comma-separated levels, for example 1,2,3")
    choice.add_argument("--all", action="store_true", help="run all ten levels")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base", default=(os.environ.get("STAGE_A_BASE_URL")
                                            or os.environ.get("KERNEL_AGENT_BASE_URL")
                                            or os.environ.get("GPTOSS_BASE_URL")))
    parser.add_argument("--path", default="",
                        help="append before /chat/completions, e.g. /agg/v1")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--verify-timeout", type=int, default=30)
    parser.add_argument("--give-up-after", type=int, default=3,
                        help="stop after the same failure and repair instruction recur N times")
    parser.add_argument("--log", default="stage_a_attempts.jsonl")
    parser.add_argument("--report", default="stage_a_report.json",
                        help="write aggregate solve rates, taxonomy, confidence, and token use")
    parser.add_argument("--offline", action="store_true",
                        help="deterministic smoke test for levels 1-10; numbers are not model results")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--think", dest="thinking", action="store_true",
                          help="enable hidden reasoning for a controlled comparison")
    thinking.add_argument("--no-think", dest="thinking", action="store_false",
                          help="explicitly select the default: return code without hidden thinking")
    parser.set_defaults(thinking=False)
    parser.add_argument("--insecure", action="store_true",
                        help="disable TLS verification only for a trusted workshop endpoint")
    args = parser.parse_args()

    if args.rounds < 1 or args.repeat < 1:
        parser.error("--rounds and --repeat must be positive")
    if not args.offline and not (args.base or "").strip():
        parser.error("set --base, STAGE_A_BASE_URL, KERNEL_AGENT_BASE_URL, or GPTOSS_BASE_URL")
    if args.base:
        args.base = args.base.rstrip("/")

    levels = parse_levels(args)
    if args.offline and any(level not in OFFLINE_GOOD for level in levels):
        parser.error("--offline supports only levels 1 through 10")

    history = {level: [] for level in levels}
    categories = Counter()
    with open(args.log, "a", encoding="utf-8") as log_handle:
        for run_number in range(1, args.repeat + 1):
            if args.repeat > 1:
                print(f"\n######## run {run_number}/{args.repeat} ########")
            for level in levels:
                result = solve_level(args, level, log_handle, run_number)
                history[level].append(result)
                categories.update(result["attempt_categories"])

    print("\n=== Stage A summary ===")
    for level in levels:
        results = history[level]
        solved = sum(result["ok"] for result in results)
        mean_confidence = sum(result["confidence"] for result in results) / len(results)
        print(f"level {level}: verified {solved}/{len(results)}  "
              f"mean confidence={mean_confidence:.2f}  "
              f"rounds={[result['rounds'] for result in results]}")
    print("failure taxonomy:", dict(categories))
    print(f"attempt log: {args.log}")
    summary = build_summary(args, levels, history, categories)
    with open(args.report, "w", encoding="utf-8") as report_handle:
        json.dump(summary, report_handle, ensure_ascii=False, indent=2)
        report_handle.write("\n")
    print(f"summary report: {args.report}")
    if args.offline:
        print("OFFLINE SMOKE TEST ONLY: do not report these solve rates as model results.")


if __name__ == "__main__":
    main()
