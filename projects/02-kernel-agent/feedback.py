"""Stable, machine-readable feedback for the NKI kernel checker.

The checker decides whether a candidate is correct.  This module only explains
that decision; changing a classification here must never change the reward or
the pass/fail result produced by :mod:`agent` and :mod:`nkibench`.
"""

from dataclasses import asdict, dataclass, replace
import re
from typing import Any, Optional


SCHEMA_VERSION = "1.1"


@dataclass(frozen=True)
class Feedback:
    """One stable diagnosis and the next action a generator should take."""

    category: str
    family: str
    severity: str
    stage: str
    summary: str
    suggestion: str
    redesign_required: bool
    retryable: bool = True
    raw_error: str = ""
    reward: Optional[float] = None
    reward_breakdown: Optional[dict] = None
    current: Optional[float] = None
    maximum: Optional[float] = None
    dimension: Optional[str] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    level: Optional[int] = None
    case: Optional[str] = None
    shape: Optional[Any] = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        """Return a JSON-serialisable object with a stable set of keys."""
        return asdict(self)

    def to_prompt(self) -> str:
        """Render the diagnosis as a concise repair instruction for the model."""
        evidence = f" Checker evidence: {self.raw_error}" if self.raw_error else ""
        action = f" Next action: {self.suggestion}" if self.suggestion else ""
        return f"[{self.category}] {self.summary}.{evidence}{action}".strip()

    def with_reward(self, reward: float, reward_breakdown: dict) -> "Feedback":
        """Attach scoring data without changing the diagnosis itself."""
        return replace(self, reward=reward, reward_breakdown=reward_breakdown)


def make_feedback(category: str, family: str, severity: str, stage: str,
                  summary: str, suggestion: str, redesign_required: bool,
                  raw_error: str = "", retryable: bool = True, **context) -> Feedback:
    """Construct feedback while ignoring context fields outside the public schema."""
    allowed = {"current", "maximum", "dimension", "expected", "actual",
               "level", "case", "shape"}
    return Feedback(
        category=category,
        family=family,
        severity=severity,
        stage=stage,
        summary=summary.rstrip("."),
        suggestion=suggestion.rstrip("."),
        redesign_required=redesign_required,
        retryable=retryable,
        raw_error=raw_error,
        **{key: value for key, value in context.items() if key in allowed},
    )


def classify_error(error_text: str, stage: str = "runtime", **context) -> Feedback:
    """Convert checker/runtime wording into a stable diagnosis.

    Pattern order is intentional: specific layout and numerical failures must be
    recognised before their broad runtime fallbacks.
    """
    raw = (error_text or "").strip()
    # ``enrich()`` is a compatibility API and may already have rendered a
    # Feedback object.  Recover its original evidence so classification is
    # idempotent instead of turning a structured prompt into UNKNOWN_ERROR.
    rendered = re.match(r"^\[[A-Z_]+\].*? Checker evidence: (.*?)(?: Next action:|$)",
                        raw, re.S)
    if rendered:
        raw = rendered.group(1).strip()
    lower = raw.lower()

    def result(category, family, severity, summary, suggestion,
               redesign_required=False, retryable=True, **fields):
        fields.update(context)
        return make_feedback(category, family, severity, stage, summary, suggestion,
                             redesign_required, raw, retryable, **fields)

    if not raw or "no code came back" in lower or "empty answer" in lower:
        return result("EMPTY_CODE", "FORMAT", "blocking",
                      "No candidate kernel was returned",
                      "Return one complete Python code block containing the imports and kernel")

    if "does not parse" in lower or "syntaxerror" in lower or "failed to import" in lower:
        return result("SYNTAX_ERROR", "FORMAT", "blocking",
                      "The candidate is not valid importable Python",
                      "Fix the reported syntax/import location and return a complete module")

    m = re.search(r"partition dimension (\d+) exceeds (?:the )?maximum(?: of)? (\d+)", raw, re.I)
    if not m:
        m = re.search(r"partition dimension (\d+) exceeds maximum=(\d+)", raw, re.I)
    if m:
        current, maximum = int(m.group(1)), int(m.group(2))
        return result("PARTITION_LIMIT", "TILE_LAYOUT", "structural",
                      f"Partition dimension {current} exceeds the hardware maximum {maximum}",
                      f"Tile that partition dimension in chunks of at most {maximum}; keep the final chunk partial",
                      True, current=current, maximum=maximum, dimension="partition")

    m = re.search(r"dma_copy requires src and dst to have the same number of elements,\s*"
                  r"got src=(\d+), dst=(\d+)", raw, re.I)
    if m:
        source, destination = int(m.group(1)), int(m.group(2))
        return result("DMA_SIZE_MISMATCH", "TILE_LAYOUT", "structural",
                      "DMA source and destination contain different numbers of elements",
                      "Allocate the destination with exactly the source slice shape; DMA does not reshape or broadcast",
                      True, expected=source, actual=destination)

    m = re.search(r"value array of shape (\([^)]*\)) could not be broadcast to\s*"
                  r"indexing result of shape (\([^)]*\))", raw, re.I)
    if m:
        return result("TILE_SHAPE_MISMATCH", "TILE_LAYOUT", "structural",
                      "A value is being written into a differently shaped destination slice",
                      "Make the destination slice and value have exactly the same shape",
                      True, actual=m.group(1), expected=m.group(2))

    m = re.search(r"Out-of-bound access .*? dimension (\d+):\s*index range "
                  r"\[(\d+),\s*(\d+)\] exceed dimension size of (\d+)", raw, re.I)
    if m:
        dimension, high, size = m.group(1), int(m.group(3)), int(m.group(4))
        return result("OUT_OF_BOUNDS", "TILE_LAYOUT", "structural",
                      f"A tile indexes through {high} on dimension {dimension}, whose size is {size}",
                      "Derive bounds from the tensor shape and use a partial final tile instead of padding past the edge",
                      True, current=high, maximum=size, dimension=dimension)

    m = re.search(r"Matmul contraction dimension (\d+) exceeds pmax=(\d+)", raw, re.I)
    if m:
        current, maximum = int(m.group(1)), int(m.group(2))
        return result("MATMUL_CONTRACTION_LIMIT", "TILE_LAYOUT", "structural",
                      f"Matmul contraction dimension {current} exceeds pmax {maximum}",
                      f"Split K into chunks of at most {maximum} and accumulate all chunks into one PSUM tile",
                      True, current=current, maximum=maximum, dimension="K")

    if "must have at least 2 dimensions" in lower:
        return result("TILE_RANK", "TILE_LAYOUT", "structural",
                      "An SBUF or PSUM tile has fewer than two dimensions",
                      "Represent the value as (rows, columns), using (1, N) or (N, 1) for a vector",
                      True)

    if "cannot reshape array of size" in lower:
        return result("INVALID_RESHAPE", "TILE_LAYOUT", "structural",
                      "The requested reshape changes the number of elements",
                      "Keep the supplied shape and use matching tile slices instead of reshaping",
                      True)

    if ("must be in ['sbuf', 'psum']" in lower
            or re.search(r"\w+ must be in \['(?:sbuf|psum)'\], got \w+", raw)):
        return result("BUFFER_PLACEMENT", "TILE_LAYOUT", "structural",
                      "An NKI operation received a tile from an unsupported memory region",
                      "Use SBUF for matmul operands, PSUM for its destination, and DMA only between HBM and on-chip buffers",
                      True)

    if "'memoryregion' object is not callable" in lower:
        return result("MEMORY_REGION_MISUSE", "API", "blocking",
                      "A memory region was called as though it were a function",
                      "Allocate with nl.ndarray(..., buffer=nl.sbuf/nl.psum/nl.shared_hbm); do not call the region")

    m = re.search(r"(\w+)\(\) got an unexpected keyword argument '([^']+)'", raw)
    if m:
        return result("INVALID_API_ARGUMENT", "API", "blocking",
                      f"{m.group(1)} does not accept the `{m.group(2)}` keyword",
                      f"Remove `{m.group(2)}=` and call {m.group(1)} with its documented NKI signature")

    if "got multiple values for argument" in lower:
        return result("DUPLICATE_API_ARGUMENT", "API", "blocking",
                      "An API argument was supplied more than once",
                      "Pass each NKI argument once by keyword, for example dst=, stationary=, and moving=")

    m = re.search(r"module '([\w.]+)' has no attribute '(\w+)'", raw)
    if m:
        return result("UNKNOWN_API", "API", "blocking",
                      f"The NKI module `{m.group(1)}` has no `{m.group(2)}` API",
                      "Replace the invented name with an API from nki.language or nki.isa")

    m = re.search(r"'([^']+)' object has no attribute '([^']+)'", raw)
    if m:
        return result("UNKNOWN_OBJECT_API", "API", "blocking",
                      f"A `{m.group(1)}` value has no `{m.group(2)}` operation",
                      "Use an nl/nisa operation supported for NKI tensors")

    if "unsupported operand type(s) for" in lower and "nkitensor" in lower:
        return result("TENSOR_PYTHON_OPERATOR", "API", "structural",
                      "A Python arithmetic operator was applied directly to an NKI tile",
                      "Use an NKI ISA operation, or accumulate matmul chunks into one PSUM tile",
                      True)

    if "no module named" in lower:
        return result("UNKNOWN_MODULE", "API", "blocking",
                      "The candidate imports a module that is unavailable",
                      "Use only import nki, import nki.language as nl, and import nki.isa as nisa")

    if "no function named" in lower:
        return result("MISSING_ENTRYPOINT", "RULE", "blocking",
                      "The required kernel entry point is missing",
                      "Define the exact entry-point name and signature required by the level")

    if "not decorated with `@nki.jit`" in raw:
        return result("MISSING_JIT_DECORATOR", "RULE", "blocking",
                      "The kernel entry point is not compiled with @nki.jit",
                      "Add @nki.jit immediately above the entry-point function")

    if "rule violation" in lower or "hands the whole operation to a framework" in lower:
        return result("RULE_VIOLATION", "RULE", "blocking",
                      "The candidate violates a level rule",
                      "Replace the operation named in the evidence with explicit NKI kernel operations")

    if "the kernel modified its input" in lower:
        return result("INPUT_MUTATION", "NUMERICAL", "numerical",
                      "The kernel overwrote an input owned by its caller",
                      "Allocate and return a new shared_hbm output instead of writing into an input")

    if "wrong shape" in lower:
        m = re.search(r"returned (\([^)]*\)), reference is (\([^)]*\))", raw, re.I)
        return result("WRONG_OUTPUT_SHAPE", "NUMERICAL", "numerical",
                      "The returned tensor shape differs from the reference",
                      "Correct the output-size arithmetic and destination slicing before changing values",
                      True, actual=m.group(1) if m else None, expected=m.group(2) if m else None)

    if "non-finite output" in lower:
        return result("NON_FINITE_OUTPUT", "NUMERICAL", "numerical",
                      "The output contains NaN or infinity",
                      "Ensure every SBUF/PSUM tile is initialized before use and stabilize reductions before exponentiation",
                      True)

    if "output is" in lower and "zeros" in lower:
        return result("OUTPUT_NOT_WRITTEN", "NUMERICAL", "numerical",
                      "Most computed values never reached the returned output",
                      "Copy every result PSUM to SBUF and then to the correct shared_hbm destination slice",
                      True)

    if "of the output is zero" in lower:
        return result("PARTIAL_OUTPUT", "NUMERICAL", "numerical",
                      "Only part of the output was computed or written",
                      "Check loop coverage and destination indices for every full and partial tile",
                      True)

    if "numerical mismatch" in lower:
        ragged = "ragged edge" in lower or "final partial" in lower
        return result("RAGGED_EDGE_MISMATCH" if ragged else "NUMERICAL_MISMATCH",
                      "NUMERICAL", "numerical",
                      "The kernel values differ from the reference" +
                      (" in a final partial tile" if ragged else ""),
                      ("Handle the final partial partition/free tile using shape-derived bounds"
                       if ragged else
                       "Check operand layout, accumulation, and the reported failing index"),
                      True)

    if "correct on cpu but wrong on hardware" in lower:
        return result("HARDWARE_HAZARD", "RUNTIME", "structural",
                      "The simulator found a pattern that can be incorrect on Trainium hardware",
                      "Remove the hazard named in the warning before accepting the CPU result",
                      True)

    if "too much hbm traffic" in lower:
        m = re.search(r"moving ([\d.]+)x .*? requires ([\d.]+)x", raw, re.I)
        return result("HBM_TRAFFIC_LIMIT", "PERFORMANCE", "performance",
                      "The kernel is numerically correct but exceeds the HBM traffic budget",
                      "Reuse resident tiles and accumulate into one PSUM before copying the final result out",
                      True, current=float(m.group(1)) if m else None,
                      maximum=float(m.group(2)) if m else None)

    if "cannot simulate" in lower or "neuron sdk is not installed" in lower:
        return result("SDK_UNAVAILABLE", "ENVIRONMENT", "environment",
                      "The checker environment cannot simulate NKI kernels",
                      "Run the checker in the provided Neuron SDK pod; do not ask the generator to change the kernel",
                      False, retryable=False)

    if "could not be loaded" in lower:
        return result("KERNEL_LOAD_ERROR", "RUNTIME", "blocking",
                      "The module imports but the kernel entry point cannot be loaded",
                      "Fix the reported load-time exception at the entry point")

    if "raised " in lower:
        return result("RUNTIME_EXCEPTION", "RUNTIME", "runtime",
                      "The kernel raised an exception during simulation",
                      "Fix the first reported exception, then rerun the same failing shape")

    if "correct on every shape" in lower or lower == "verified":
        return result("VERIFIED", "SUCCESS", "info",
                      "The kernel passed every required check", "", False, retryable=False)

    fallback = {
        "parse": ("SYNTAX_ERROR", "FORMAT", "blocking"),
        "rules": ("RULE_VIOLATION", "RULE", "blocking"),
        "load": ("KERNEL_LOAD_ERROR", "RUNTIME", "blocking"),
        "numerics": ("NUMERICAL_MISMATCH", "NUMERICAL", "numerical"),
        "performance": ("PERFORMANCE_FAILURE", "PERFORMANCE", "performance"),
        "success": ("VERIFIED", "SUCCESS", "info"),
    }
    category, family, severity = fallback.get(stage,
                                               ("UNKNOWN_ERROR", "RUNTIME", "runtime"))
    return result(category, family, severity,
                  "The checker reported an unclassified failure",
                  "Use the raw checker evidence to fix the first failing operation")
