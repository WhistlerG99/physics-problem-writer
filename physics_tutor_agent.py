"""
Physics Tutor Multi-Agent System — LangGraph + OpenAI
======================================================
Architecture
------------
  writer → validate → (blank? retry writer) → dispatch_students
         → [student_0 … student_N]  (parallel via Send)
         → collect_results
              └─ should_loop? ──yes──→ writer
                               └─ no ──→ summarise → END

Termination conditions (either):
  • ≤ MAX_CORRECT students answered correctly  (problem is hard enough)
  • problem_count reached MAX_PROBLEMS

Answer format
-------------
  The writer produces a problem, full worked solution, and a single
  numeric answer (integer or fraction p/q).
"""

from __future__ import annotations

import argparse
import operator
import os
import re
import sys
import textwrap
from fractions import Fraction
from pathlib import Path
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────
# Default constants (all overridable via CLI)
# ─────────────────────────────────────────────────────────────

DEFAULT_NUM_STUDENTS   = 8
DEFAULT_MAX_CORRECT    = 2      # stop when ≤ this many students are correct
DEFAULT_MAX_PROBLEMS   = 6
DEFAULT_OUTPUT_DIR     = "."
DEFAULT_WRITER_MODEL   = "gpt-4o"
DEFAULT_STUDENT_MODEL     = "gpt-4o"
DEFAULT_REVIEWER_SAMPLES      = 4     # students shown to the writer each round
DEFAULT_REASONING_MAX_CHARS  = 2000  # max chars of student reasoning sent to writer
DEFAULT_WRITER_TEMPERATURE   = 0.7   # temperature for writer/verifier/summariser
MAX_WRITER_RETRIES     = 3      # max blank-response retries before aborting


# ─────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────

class RoundResult(TypedDict):
    student_id: int
    answer:     str
    reasoning:  str   # full student response, not just the final answer
    correct:    bool


# ─────────────────────────────────────────────────────────────
# Custom reducer for round_results
# ─────────────────────────────────────────────────────────────

def _merge_round_results(
    current: list[RoundResult], incoming: list[RoundResult]
) -> list[RoundResult]:
    """
    Fan-in reducer for per-round student results.

    - incoming == []  →  RESET  (writer cleared the slate for a new round)
    - incoming != []  →  APPEND (a student worker is writing its result)

    Using operator.add would make the empty-list reset a no-op ([] + x = x),
    so results from previous rounds would bleed into the next round's counts.
    """
    if not incoming:
        return []          # explicit reset
    return current + incoming


class TutorState(TypedDict):
    # ── runtime config ────────────────────────────────────────
    num_students:   int
    max_correct:    int
    max_problems:   int
    output_dir:     str
    summary_dir:    str
    writer_model:          str
    student_model:         str
    num_reviewer_samples:    int   # how many student attempts the writer sees
    reasoning_max_chars:     int   # max chars of each student reasoning sent to writer
    writer_temperature:      float # temperature for writer/verifier/summariser

    # ── notes content ─────────────────────────────────────────
    latex_paths:    list[str]
    notes_text:     str        # full raw extracted text (may be very long)
    notes_summary:  str        # LLM-condensed concept index fed to writer

    # ── current problem (set by writer, cleared on retry) ─────
    context:          str   # problem setup and given values
    question:         str   # single-line question ending in ?
    solution:         str   # full worked solution
    reference_answer: str   # just the final number

    # ── writer blank-retry counter (resets each real round) ───
    writer_retry_count: int

    # ── solution verification ─────────────────────────────────
    solution_verified: bool
    verify_reason:     str

    # ── per-round student results ─────────────────────────────
    round_results: Annotated[list[RoundResult], _merge_round_results]

    # ── loop control ──────────────────────────────────────────
    problem_count: int
    stop_reason:   str
    all_rounds:    Annotated[list[dict], operator.add]


class StudentState(TypedDict):
    student_id:       int
    problem:          str
    reference_answer: str
    model:            str
    num_students:     int
    output_dir:       str
    problem_count:    int
    round_results:    Annotated[list[RoundResult], _merge_round_results]


# ─────────────────────────────────────────────────────────────
# LaTeX extraction
# ─────────────────────────────────────────────────────────────

# Chunk size for summarisation passes (in characters).
# gpt-4o context is ~128k tokens ≈ 400k chars; we stay well under that.
_CHUNK_CHARS = 80_000

_SUMMARISE_SYSTEM = textwrap.dedent("""
    You are a physics teaching assistant. You will be given a portion of
    physics lecture notes in LaTeX source form.

    Your job is to produce a DENSE CONCEPT INDEX of the material — a
    structured, comprehensive summary that preserves:
      • Every named law, principle, and theorem (with its equation)
      • Every defined quantity and its symbol
      • Every formula or relation (write them out explicitly)
      • Every worked example or special case mentioned
      • Section/topic names so the writer knows what areas are covered

    Format: use short bullet points and inline equations.  Do NOT write
    prose paragraphs.  Do NOT omit equations — they are the most
    important part.  Be thorough: a problem writer must be able to set
    quantitative problems from your index alone.
""").strip()


def find_latex_files(directories: list[str]) -> list[str]:
    found: list[str] = []
    for d in directories:
        dp = Path(d)
        if not dp.is_dir():
            print(f"  [WARNING] Not a directory, skipping: {d}", file=sys.stderr)
            continue
        for ext in ("*.tex", "*.latex"):
            found.extend(str(p) for p in dp.rglob(ext))
    found.sort()
    return found


def _strip_latex_boilerplate(raw: str) -> str:
    """Remove preamble and non-content commands from LaTeX source."""
    if r"\begin{document}" in raw:
        raw = raw.split(r"\begin{document}", 1)[1]
    if r"\end{document}" in raw:
        raw = raw.split(r"\end{document}", 1)[0]
    raw = re.sub(
        r"\\(label|ref|cite|bibitem|bibliography|usepackage"
        r"|documentclass|newcommand|renewcommand|setlength"
        r"|pagestyle|thispagestyle|clearpage|newpage"
        r"|maketitle|tableofcontents)\b[^\n]*", "", raw)
    return re.sub(r"\n{3,}", "\n\n", raw).strip()


def extract_text_from_latex(paths: list[str]) -> str:
    """Read and lightly strip all LaTeX files; return combined raw text."""
    all_chunks: list[str] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  [WARNING] File not found: {path}", file=sys.stderr)
            continue
        print(f"  [LaTeX] Reading: {p.name}")
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
            raw = _strip_latex_boilerplate(raw)
            all_chunks.append(f"=== {p.name} ===\n{raw}")
        except Exception as e:
            print(f"  [ERROR] Could not read {path}: {e}", file=sys.stderr)
    return "\n\n".join(all_chunks)


def _cache_key(latex_paths: list[str]) -> str:
    """
    Stable cache key based on the sorted file paths and their last-modified
    times.  Any change to the notes files invalidates the cache.
    """
    import hashlib
    parts = []
    for p in sorted(latex_paths):
        mtime = Path(p).stat().st_mtime if Path(p).exists() else 0
        parts.append(f"{p}:{mtime}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _cache_path(summary_dir: str, latex_paths: list[str]) -> str:
    key = _cache_key(latex_paths)
    return os.path.join(summary_dir, f"notes_summary_{key}.txt")


def summarise_notes(notes_text: str, model: str,
                    summary_dir: str = "", latex_paths: list[str] | None = None,
                    temperature: float | None = None) -> str:
    """
    Condense arbitrarily long notes into a dense concept index.

    Results are cached to  <output_dir>/notes_summary_<hash>.txt
    based on the file paths and modification times of the LaTeX sources.
    Re-runs skip the LLM calls entirely if the notes have not changed.
    """
    # ── Try cache first ───────────────────────────────────────
    if summary_dir and latex_paths:
        cache_file = _cache_path(summary_dir, latex_paths)
        if Path(cache_file).exists():
            cached = Path(cache_file).read_text(encoding="utf-8")
            print(f"  [NOTES] Loaded summary from cache: {cache_file}")
            print(f"  [NOTES] Cached summary: {len(cached):,} chars")
            return cached
    else:
        cache_file = ""

    total = len(notes_text)
    print(f"  [NOTES] Total extracted text: {total:,} chars")

    if total == 0:
        return "(No notes content found.)"

    # ── Split into chunks ─────────────────────────────────────
    chunks: list[str] = []
    pos = 0
    while pos < total:
        end = min(pos + _CHUNK_CHARS, total)
        if end < total:
            boundary = notes_text.rfind("\n\n", pos, end)
            if boundary > pos:
                end = boundary
        chunks.append(notes_text[pos:end].strip())
        pos = end

    print(f"  [NOTES] Split into {len(chunks)} chunk(s) for summarisation")

    # ── Map: summarise each chunk ─────────────────────────────
    partial: list[str] = []
    for i, chunk in enumerate(chunks):
        print(f"  [NOTES] Summarising chunk {i+1}/{len(chunks)} "
              f"({len(chunk):,} chars)…")
        summary = invoke_llm(
            model, _SUMMARISE_SYSTEM,
            f"Summarise the following physics notes into a dense concept index:\n\n{chunk}",
            temperature=temperature,
        )
        partial.append(summary)

    combined = "\n\n---\n\n".join(partial)

    # ── Reduce: merge partial summaries if still large ────────
    if len(chunks) > 1:
        print(f"  [NOTES] Merging {len(partial)} partial summaries "
              f"({len(combined):,} chars)…")
        combined = invoke_llm(
            model, _SUMMARISE_SYSTEM,
            "Merge these partial concept indices into one unified, "
            "non-redundant concept index:\n\n" + combined,
            temperature=temperature,
        )

    print(f"  [NOTES] Final concept index: {len(combined):,} chars")

    # ── Save to cache ─────────────────────────────────────────
    if cache_file:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_file).write_text(combined, encoding="utf-8")
        print(f"  [NOTES] Summary cached → {cache_file}")

    return combined

# ─────────────────────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────────────────────

def is_reasoning_model(model: str) -> bool:
    """
    Returns True for models that use an internal reasoning/thinking step
    and therefore:
      • need max_completion_tokens (not max_tokens) set high enough to
        cover reasoning tokens + visible output tokens
      • should not receive a temperature parameter
      • should not receive a SystemMessage (fold into user turn)

    Covers:
      o-series  : o1, o3, o3-pro, o4-mini, …
      gpt-5.x   : gpt-5, gpt-5.5, gpt-5.4, gpt-5.3, gpt-5.1, gpt-5-mini,
                  gpt-5-nano, gpt-5.4-mini, gpt-5.4-nano, …
    """
    model_lower = model.lower()
    if re.match(r"^o\d", model_lower):          # o1, o3, o4-mini …
        return True
    if re.match(r"^gpt-5", model_lower):          # gpt-5, gpt-5.5, gpt-5.4 …
        return True
    return False


# Token budget for reasoning models.
# gpt-5.5 uses up to ~32k reasoning tokens internally; we give 32k for
# reasoning + 8k for visible output = 40k total.  Adjust down for cheaper
# models if latency / cost matters.
_REASONING_MAX_TOKENS = 40_000
_STANDARD_MAX_TOKENS  =  4_096


def make_llm(model: str, temperature: float | None = None) -> ChatOpenAI:
    reasoning = is_reasoning_model(model)
    kwargs: dict = dict(
        model=model,
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    if reasoning:
        # max_completion_tokens covers reasoning tokens + output tokens combined
        kwargs["max_completion_tokens"] = _REASONING_MAX_TOKENS
    else:
        kwargs["max_tokens"] = _STANDARD_MAX_TOKENS
        if temperature is not None:
            kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)


def invoke_llm(model: str, system: str, user: str,
               temperature: float | None = None) -> str:
    llm = make_llm(model, temperature=temperature)
    if is_reasoning_model(model):
        # Reasoning models ignore / reject SystemMessage; fold into user turn
        messages = [HumanMessage(content=f"{system}\n\n{user}")]
    else:
        messages = [SystemMessage(content=system), HumanMessage(content=user)]

    response = llm.invoke(messages)
    raw = response.content if response.content else ""

    print(f"  [LLM RAW] ({model}) first 400 chars:\n"
          f"  {raw[:400].replace(chr(10), chr(10)+'  ')}")

    if not raw.strip():
        raise RuntimeError(
            f"Model '{model}' returned empty content. "
            f"Full response: {response}"
        )
    return raw


# ─────────────────────────────────────────────────────────────
# Answer normalisation
# ─────────────────────────────────────────────────────────────

def _normalise(s: str) -> str | None:
    s = s.strip().replace(" ", "").replace(",", "").rstrip(".")
    for parser in (Fraction, lambda x: Fraction(float(x))):
        try:
            return str(parser(s).limit_denominator(10_000))
        except (ValueError, ZeroDivisionError):
            pass
    return None


def answers_match(student_raw: str, reference_raw: str) -> bool:
    def first_num(text: str) -> str:
        m = re.search(r"-?\d+(?:[./]\d+)?", text)
        return m.group() if m else text.strip()
    s = _normalise(first_num(student_raw))
    r = _normalise(first_num(reference_raw))
    return s is not None and r is not None and s == r


# ─────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────

def _write(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def _append(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


# ─────────────────────────────────────────────────────────────
# Writer prompt & parser
# ─────────────────────────────────────────────────────────────

_WRITER_SYSTEM = textwrap.dedent("""
    You are an expert physics professor writing graduate-level exam problems.

    STRICT RULES:
    1. Create ONE graduate-level physics problem whose answer is a SINGLE
       number — either an integer (e.g. 4) or a simplified fraction (e.g. 3/4).
       No units, no vectors, no multi-part answers.
    2. GRADUATE LEVEL means: the problem must require graduate-level reasoning,
       multi-step derivation, or non-trivial application of advanced theory
       (e.g. quantum mechanics, general relativity, statistical mechanics,
       QFT, condensed matter, classical field theory). It should NOT be
       solvable by a strong undergraduate using only introductory physics.
    3. The answer MUST NOT be 0, 1, or -1. These are trivial and forbidden.
       If your working leads to 0, 1, or -1, choose a different problem.
    4. ANSWER PRECISION: The final answer must have at most 4 significant
       figures. If the exact result has more significant figures, round it
       to 4 significant figures before recording it as the REFERENCE_ANSWER.
    5. DECIMAL ANSWERS: If the natural answer is a non-integer decimal and
       cannot be expressed as a clean fraction, you MAY scale it to an integer
       by multiplying by a convenient factor (e.g. multiply by 100 to get a
       percentage, or by $10^n$ to clear a small decimal). When you do this:
       — You MUST instruct the student to compute the same scaled quantity
         in both the Problem Context and the Question (e.g. "Express your
         answer as a percentage rounded to the nearest integer" or "Give
         your answer multiplied by $10^3$, rounded to the nearest integer").
       — The REFERENCE_ANSWER must be that rounded integer.
       — Never silently scale without telling the student.
    6. The problem MUST be completely self-contained. A student with no
       access to any notes must be able to solve it from the problem text alone.
       — Define every quantity, symbol, and constant you use.
       — State all given numerical values explicitly.
       — Never say "as given in the notes", "from the lecture", "as defined
         above", "using the formula from class", or any similar phrase.
       — Never reference the notes, course, textbook, or any external source.

    HARDENING TECHNIQUES (use these when refining a problem that was too easy):
    When you are shown student attempts and told to make the problem harder,
    diagnose which failure mode the students fell into and apply the
    corresponding technique to the SAME or a closely related problem:

    A) WRONG APPROACH (most common ~57% of errors):
       The student picked the wrong physical model or solution method.
       Technique: keep the same surface appearance but require a
       non-obvious model or modification that invalidates the standard
       textbook approach. Make the problem LOOK like a familiar type
       while secretly requiring a different framework.

    B) MISSED KEY INSIGHT (~21% of errors):
       The student brute-forced instead of spotting a symmetry, conservation
       law, or elegant cancellation.
       Technique: design around a hidden simplification the student will
       overlook — e.g. a high-dimensional integral that collapses with
       the right substitution, a group-theory argument, or a dimensional
       analysis shortcut. The brute-force path should be intractable.

    C) WRONG FORMULA (~14% of errors):
       The student used the right approach but the wrong formula.
       Technique: draw from less-canonical sources or use formulas that
       closely resemble a popular one but differ by a key sign or factor.
       Fertile ground: QFT propagators, nuclear cross-sections, condensed-
       matter Green functions, relativistic corrections.

    D) GAVE UP / GUESSED (~7% of errors):
       The student abandoned the derivation mid-way.
       Technique: require long careful symbolic manipulation that cannot
       be shortcut numerically. Multiple chained steps where each builds
       on the last.

    E) INFORMATION WITHHOLDING (use when the problem was solved too directly):
       The students were given intermediate results or equations they should
       have derived themselves, making the problem too mechanical.
       Technique: remove the "gift" and replace it with the more fundamental
       object the student must start from.  Examples:
         — Instead of giving the equations of motion, give the Lagrangian
           and require the student to derive the equations of motion first.
         — Instead of giving the dispersion relation, give the Hamiltonian.
         — Instead of giving a propagator, give the action and require the
           path-integral to be evaluated.
         — Instead of stating a conserved quantity, give the symmetry and
           require Noether's theorem to be applied.
       CRITICAL: When you remove a result and replace it with the underlying
       object, you MUST include in the Problem Context all assumptions that
       were used to derive that result (e.g. boundary conditions, the form
       of the metric, which fields are background vs. dynamical, any
       approximations such as weak-field or non-relativistic limit). The
       problem must remain fully self-contained.

    When refining, explicitly choose the technique most likely to exploit the
    weakness shown in the student attempts you are given.

    MATHEMATICAL NOTATION:
    All mathematical expressions MUST use LaTeX math delimiters:
    — Inline math  (variables, short expressions): $...$
      e.g. "A particle of mass $m$ moves with momentum $p = mv$."
    — Display math (equations, derivations):       $$...$$
      e.g. "$$E_n = \\frac{n^2 \\pi^2 \\hbar^2}{2mL^2}$$"
    NEVER use \\begin{equation}, \\begin{align}, or any \\begin{...}...\\end{...}
    environment — ONLY $...$ and $$...$$.
    Never write bare expressions like "E = hf" or "p^2/2m" — always wrap them.

    You MUST respond in EXACTLY this format — no preamble, no commentary:

    Problem Context:
    <All setup information using $...$ and $$...$$ for all math. Include the
     physical scenario, given quantities with values and units, definitions of
     all symbols used, any diagrams described in words.>

    Question:
    <A single sentence ending in ? that asks for exactly one numeric quantity.
     Use $...$ for any symbols mentioned. Do NOT include given values here.>

    SOLUTION:
    <Complete step-by-step working using $$...$$ for all equations and $...$
     for inline expressions. Identify principles, write equations, substitute
     values, show all algebra, state the final numeric result.
     IMPORTANT: The solution must contain ONLY the mathematical derivation.
     Do NOT include any commentary about difficulty, hardening techniques,
     what was changed from a previous version, or why the problem was
     designed a certain way. Pure physics working only.>

    REFERENCE_ANSWER:
    <single integer or simplified fraction only, e.g. 7 or 2/3;
     at most 4 significant figures; if you scaled a decimal answer,
     give the rounded integer here>
""").strip()


def _parse_writer_response(raw: str) -> tuple[str, str, str, str]:
    """
    Extract (context, question, solution, reference_answer) from writer output.

    The problem block is now split into:
      Problem Context:  — setup, given values, symbol definitions
      Question:         — single line ending in ?

    Returns ("", "", "", "") to signal a retry when nothing usable is found.
    """
    if not raw.strip():
        return "", "", "", ""

    context  = ""
    question = ""
    solution = ""
    reference_answer = ""

    # Normalise line endings
    raw = raw.replace("\r\n", "\n")

    # ── Strategy 1: new-format markers ───────────────────────
    has_ctx  = bool(re.search(r"Problem Context\s*:", raw, re.IGNORECASE))
    has_q    = bool(re.search(r"Question\s*:",        raw, re.IGNORECASE))
    has_sol  = bool(re.search(r"SOLUTION\s*:",        raw, re.IGNORECASE))
    has_ans  = bool(re.search(r"REFERENCE_ANSWER\s*:",raw, re.IGNORECASE))

    if has_ctx and has_q and has_ans:
        # Work backwards from the last marker
        _sp_ans = re.split(r"REFERENCE_ANSWER\s*:", raw, maxsplit=1, flags=re.IGNORECASE)
        before_ans, ans_block = _sp_ans[0], _sp_ans[1]
        if has_sol:
            _sp_sol = re.split(r"SOLUTION\s*:", before_ans, maxsplit=1, flags=re.IGNORECASE)
            before_sol, sol_block = _sp_sol[0], _sp_sol[1]
            solution = sol_block.strip()
        else:
            before_sol = before_ans
        # Split context and question
        ctx_q = re.split(r"Question\s*:", before_sol, maxsplit=1, flags=re.IGNORECASE)
        ctx_raw = re.split(r"Problem Context\s*:", ctx_q[0], maxsplit=1, flags=re.IGNORECASE)
        context  = ctx_raw[-1].strip()
        question = ctx_q[1].strip() if len(ctx_q) > 1 else ""
        reference_answer = ans_block.strip()

    # ── Strategy 2: old PROBLEM: marker (graceful fallback) ──
    elif bool(re.search(r"PROBLEM\s*:", raw, re.IGNORECASE)) and has_ans:
        print("  [WRITER WARNING] Got old PROBLEM: format; accepting as context.",
              file=sys.stderr)
        _sp_ans = re.split(r"REFERENCE_ANSWER\s*:", raw, maxsplit=1, flags=re.IGNORECASE)
        before_ans, ans_block = _sp_ans[0], _sp_ans[1]
        if has_sol:
            _sp_sol = re.split(r"SOLUTION\s*:", before_ans, maxsplit=1, flags=re.IGNORECASE)
            before_sol, sol_block = _sp_sol[0], _sp_sol[1]
            solution = sol_block.strip()
        else:
            before_sol = before_ans
        context  = re.split(r"PROBLEM\s*:", before_sol, maxsplit=1, flags=re.IGNORECASE)[-1].strip()
        question = ""
        reference_answer = ans_block.strip()

    # ── Strategy 3: nothing recognisable — signal retry ───────
    else:
        print("  [WRITER WARNING] Response has no recognisable markers; "
              "will retry.", file=sys.stderr)
        return "", "", "", ""

    # ── Validate / clean question ─────────────────────────────
    # Keep only the first non-empty line of the question field; if it doesn't
    # end with "?" strip trailing punctuation and add one.
    if question:
        first_line = next(
            (ln.strip() for ln in question.splitlines() if ln.strip()), ""
        )
        if not first_line.endswith("?"):
            first_line = first_line.rstrip(".!") + "?"
        question = first_line

    # Clean reference_answer to its first number
    if reference_answer:
        num_m = re.search(r"-?\d+(?:/\d+)?", reference_answer)
        reference_answer = num_m.group() if num_m else reference_answer.splitlines()[0].strip()

    return context, question, solution, reference_answer


def _format_problem(context: str, question: str) -> str:
    """Render context + question into the canonical display string."""
    parts = []
    if context:
        parts.append(f"Problem Context:\n{context}")
    if question:
        parts.append(f"Question:\n{question}")
    return "\n\n".join(parts) if parts else context


# ─────────────────────────────────────────────────────────────
# History hint builder
# ─────────────────────────────────────────────────────────────

def _build_history_hint(last_round: dict, num_students: int,
                        num_reviewer_samples: int | None = None,
                        reasoning_max_chars: int = 2000) -> str:
    """
    Build a rich prompt section describing the previous round so the writer
    can diagnose student failure modes and apply targeted hardening techniques.

    num_reviewer_samples controls how many student attempts are included.
    When set, a stratified random sample is taken: roughly half from correct
    students and half from wrong ones, so the writer sees both sides.
    If None (or >= total students), all attempts are included.
    """
    import random

    prev_context  = last_round.get("context",  "")
    prev_question = last_round.get("question", "")
    prev_solution = last_round.get("solution", "")
    prev_answer   = last_round.get("reference_answer", "")
    n_correct     = last_round.get("n_correct", 0)
    results       = last_round.get("results",  [])

    # ── Stratified sample ────────────────────────────────────────────────────
    k = num_reviewer_samples
    if k is None or k >= len(results):
        sampled = results
    else:
        correct_pool = [r for r in results if     r["correct"]]
        wrong_pool   = [r for r in results if not r["correct"]]
        # Take at least 1 from each side when both pools are non-empty
        n_correct_want = max(1, k // 2) if correct_pool and wrong_pool else k
        n_wrong_want   = k - n_correct_want
        sampled = (
            random.sample(correct_pool, min(n_correct_want, len(correct_pool)))
            + random.sample(wrong_pool,  min(n_wrong_want,  len(wrong_pool)))
        )
        # If one pool was smaller, top up from the other
        shortfall = k - len(sampled)
        if shortfall > 0:
            already = {id(r) for r in sampled}
            remainder = [r for r in results if id(r) not in already]
            sampled += random.sample(remainder, min(shortfall, len(remainder)))

    sampled_ids = {r["student_id"] for r in sampled}
    print(f"  [WRITER]  Reviewing {len(sampled)}/{len(results)} student attempts "
          f"(students: {sorted(sampled_ids)})")

    # ── Format each sampled attempt ─────────────────────────────────────────
    student_lines: list[str] = []
    for r in sorted(sampled, key=lambda x: x["student_id"]):
        mark      = "CORRECT" if r["correct"] else "WRONG"
        reasoning = r.get("reasoning", "").strip()
        snippet   = reasoning[:reasoning_max_chars] + "…" if len(reasoning) > reasoning_max_chars else reasoning
        student_lines.append(
            f"  Student {r['student_id']} [{mark}]:\n"
            + "\n".join(f"    {ln}" for ln in snippet.splitlines())
        )

    reviewed_note = (
        f"(Showing {len(sampled)} of {len(results)} student attempts — "
        f"a stratified random sample)"
        if len(sampled) < len(results) else
        f"(All {len(results)} student attempts shown)"
    )
    students_block = reviewed_note + "\n\n" + "\n\n".join(student_lines)

    hint = f"""

════════════════════════════════════════════════════════════════
PREVIOUS ROUND — TOO EASY ({n_correct}/{num_students} students correct)
════════════════════════════════════════════════════════════════

Problem Context:
{prev_context}

Question:
{prev_question}

Reference Answer: {prev_answer}

Reference Solution:
{prev_solution}

Student Attempts:
{students_block}

════════════════════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════════════════════
The problem above was too easy. Study the student attempts above carefully:

1. Diagnose which failure mode the WRONG students fell into:
   A) Wrong approach      — picked the wrong physical model entirely
   B) Missed key insight  — brute-forced instead of spotting a symmetry/trick
   C) Wrong formula       — right method, wrong formula (sign/factor error)
   D) Gave up/guessed     — abandoned the derivation mid-way
   E) Too direct          — students were handed intermediate results they
                            should have derived (problem was too mechanical)

2. MODIFY or REPLACE the problem to exploit that weakness:
   — Prefer modifying the same problem (change a parameter, add a twist,
     remove a shortcut, or withhold an intermediate result) over writing a
     completely new one — this keeps the hardening targeted at the specific
     gap the students showed.
   — Apply the corresponding hardening technique from the STRICT RULES above.
   — For technique E: replace the given result with the underlying object
     (Lagrangian, Hamiltonian, action, symmetry, etc.) and make sure all
     assumptions needed to derive the removed result are stated in the
     Problem Context.
   — The correct students should now find it significantly harder.

Write the new (or modified) problem in the required format below.
"""
    return hint


# ─────────────────────────────────────────────────────────────
# Node: writer
# ─────────────────────────────────────────────────────────────

def writer_node(state: TutorState) -> dict:
    # problem_count only increments when we actually send a problem to students,
    # so here we just read it (validate_node increments it on success).
    retry = state.get("writer_retry_count", 0)
    count = state.get("problem_count", 0)
    model = state["writer_model"]

    label = f"Round {count + 1}" if retry == 0 else f"Round {count + 1} retry {retry}"
    print(f"\n{'═'*62}")
    print(f"  [WRITER]  {label} / {state['max_problems']}  (model: {model})")
    print(f"{'═'*62}")

    # Extract + summarise notes on first call only
    notes_text    = state.get("notes_text", "")
    notes_summary = state.get("notes_summary", "")
    if not notes_text:
        notes_text = extract_text_from_latex(state["latex_paths"])
        if not notes_text.strip():
            notes_text = "(No text extracted from the supplied LaTeX files.)"
    if not notes_summary:
        notes_summary = summarise_notes(
            notes_text, model,
            summary_dir=state["summary_dir"],
            latex_paths=state["latex_paths"],
        )

    # Build a rich hardening brief from the previous round's student attempts
    history_hint = ""
    prev_rounds = state.get("all_rounds", [])
    if prev_rounds:
        history_hint = _build_history_hint(
            prev_rounds[-1],
            state["num_students"],
            num_reviewer_samples=state.get("num_reviewer_samples"),
            reasoning_max_chars=state.get("reasoning_max_chars", 2000),
        )

    user_content = f"Physics notes (concept index):\n\n{notes_summary}{history_hint}"
    raw = invoke_llm(model, _WRITER_SYSTEM, user_content,
                     temperature=state.get("writer_temperature"))
    context, question, solution, reference_answer = _parse_writer_response(raw)

    if context or question:
        print(f"  Context  : {context[:80]}{'…' if len(context) > 80 else ''}")
        print(f"  Question : {question}")
        print(f"  Answer   : {reference_answer}")

    return {
        "notes_text":         notes_text,
        "notes_summary":      notes_summary,
        "context":            context,
        "question":           question,
        "solution":           solution,
        "reference_answer":   reference_answer,
        # don't touch problem_count here — validate_node does that
    }


# ─────────────────────────────────────────────────────────────
# Node: validate_writer
# Checks whether the writer produced a usable problem.
# Routes back to writer on blank, or forward to students.
# ─────────────────────────────────────────────────────────────

# Answers that are too trivial to be useful
_FORBIDDEN_ANSWERS = {"0", "1", "-1"}


def validate_writer(state: TutorState) -> dict:
    """
    Inspects the writer's output and either increments the retry counter
    (blank or forbidden answer) or increments problem_count and resets the
    retry counter (good).  The routing decision is made by route_after_validate.
    """
    context  = state.get("context",  "").strip()
    question = state.get("question", "").strip()
    ref_ans  = state.get("reference_answer", "").strip()
    retry    = state.get("writer_retry_count", 0)

    if not context or not ref_ans:
        new_retry = retry + 1
        print(f"  [VALIDATE] Blank context or answer detected "
              f"(retry {new_retry}/{MAX_WRITER_RETRIES}). Re-asking writer.")
        return {"writer_retry_count": new_retry}

    # Normalise to check for forbidden values (handles "0/1", "-1/1", etc.)
    normed = _normalise(ref_ans) or ref_ans
    if normed in _FORBIDDEN_ANSWERS:
        new_retry = retry + 1
        print(f"  [VALIDATE] Forbidden answer {ref_ans!r} (normalised: {normed!r}). "
              f"Re-asking writer (retry {new_retry}/{MAX_WRITER_RETRIES}).")
        return {"writer_retry_count": new_retry,
                "context": "", "question": "", "reference_answer": ""}

    # Good problem — advance the round counter and reset retry count
    return {
        "problem_count":      state.get("problem_count", 0) + 1,
        "writer_retry_count": 0,
        "solution_verified":  False,
        "verify_reason":      "",
    }


def route_after_validate(state: TutorState) -> str:
    """
    After validate_writer:
      - blank & under retry limit  → "writer"
      - blank & over retry limit   → "summarise"
      - good problem               → "verify_solution"
    """
    context = state.get("context", "").strip()
    ref_ans = state.get("reference_answer", "").strip()
    retry   = state.get("writer_retry_count", 0)

    if not context or not ref_ans:
        if retry >= MAX_WRITER_RETRIES:
            print(f"  [VALIDATE] Exceeded {MAX_WRITER_RETRIES} retries. Giving up.")
            return "summarise"
        return "writer"

    return "verify_solution"


# ─────────────────────────────────────────────────────────────
# Node: verify_solution
# An independent LLM re-solves the problem and checks the answer.
# ─────────────────────────────────────────────────────────────

_VERIFY_SYSTEM = textwrap.dedent("""
    You are a rigorous physics professor checking an exam problem for errors.

    You will be given:
      • A problem statement (Context + Question)
      • A proposed solution with a final numeric answer

    Your job:
    1. Re-derive the answer INDEPENDENTLY from first principles — do NOT
       just follow the proposed solution step-by-step.
    2. Check whether your answer agrees with the proposed REFERENCE_ANSWER.
    3. If they agree, respond with exactly:
         VERDICT: CORRECT
    4. If they disagree, respond with exactly:
         VERDICT: INCORRECT
         REASON: <one concise sentence describing the error>

    Do not add any other text before or after.
""").strip()


def verify_solution(state: TutorState) -> dict:
    """
    Independently re-solves the problem and checks it against the writer's
    reference answer.  Stores the verdict in state so route_after_verify
    can decide whether to accept the problem or send the writer back.
    """
    model = state["writer_model"]
    context  = state.get("context",  "").strip()
    question = state.get("question", "").strip()
    solution = state.get("solution", "").strip()
    ref_ans  = state.get("reference_answer", "").strip()

    problem_text = _format_problem(context, question)

    user_content = (
        f"PROBLEM:\n{problem_text}\n\n"
        f"PROPOSED SOLUTION:\n{solution}\n\n"
        f"REFERENCE_ANSWER: {ref_ans}"
    )

    print(f"\n  [VERIFY]  Re-solving with {model}…")
    raw = invoke_llm(model, _VERIFY_SYSTEM, user_content,
                     temperature=state.get("writer_temperature"))

    verdict_line = next(
        (ln.strip() for ln in raw.splitlines() if ln.strip().upper().startswith("VERDICT:")),
        ""
    )
    correct = "CORRECT" in verdict_line.upper() and "INCORRECT" not in verdict_line.upper()

    if correct:
        print(f"  [VERIFY]  ✓ Solution verified correct")
    else:
        reason_line = next(
            (ln.strip() for ln in raw.splitlines() if ln.strip().upper().startswith("REASON:")),
            raw.strip()
        )
        print(f"  [VERIFY]  ✗ Solution failed verification: {reason_line}")

    return {"solution_verified": correct, "verify_reason": raw.strip()}


def route_after_verify(state: TutorState) -> str | list[Send]:
    """
    After verify_solution:
      - verified correct  → fan out to students (Send)
      - failed            → back to writer (if retries remain) or summarise
    """
    retry = state.get("writer_retry_count", 0)

    if state.get("solution_verified", False):
        # Build the full problem text students will receive
        problem_text = _format_problem(
            state.get("context", "").strip(),
            state.get("question", "").strip(),
        )
        return [
            Send("student", {
                "student_id":       i,
                "problem":          problem_text,
                "reference_answer": state["reference_answer"],
                "model":            state["student_model"],
                "num_students":     state["num_students"],
                "output_dir":       state["output_dir"],
                "problem_count":    state["problem_count"],
                "round_results":    [],
            })
            for i in range(state["num_students"])
        ]

    # Verification failed — treat like a writer retry
    new_retry = retry + 1
    print(f"  [VERIFY]  Sending writer back (retry {new_retry}/{MAX_WRITER_RETRIES})")
    if new_retry >= MAX_WRITER_RETRIES:
        print(f"  [VERIFY]  Exceeded retry limit. Giving up.")
        return "summarise"

    return "writer"


# ─────────────────────────────────────────────────────────────
# Node: student
# ─────────────────────────────────────────────────────────────

_STUDENT_SYSTEM = textwrap.dedent("""
    You are a physics undergraduate student sitting an exam.
    Solve the problem step-by-step, then give your final answer.

    IMPORTANT: The very last line of your response must be ONLY the
    numeric answer — a single integer or fraction (e.g. 4 or 3/4).
    No units, no words, just the number on its own line.
""").strip()


def student_node(state: StudentState) -> dict:
    sid = state["student_id"]
    n   = state["num_students"]
    temp = 0.3 + sid * (0.6 / max(n - 1, 1))

    raw = invoke_llm(state["model"], _STUDENT_SYSTEM,
                     state["problem"], temperature=temp)

    last_line = next(
        (ln.strip() for ln in reversed(raw.splitlines()) if ln.strip()), raw
    )

    correct = answers_match(last_line, state["reference_answer"])
    mark = "✓" if correct else "✗"
    print(f"  [Student {sid:>2}]  answered={last_line!r:>10}  "
          f"ref={state['reference_answer']!r}  {mark}")

    # Write this student's full reasoning to the student responses file
    out_dir   = state.get("output_dir", ".")
    rnum      = state.get("problem_count", 0)
    resp_file = os.path.join(out_dir, "student_responses.txt")
    sep       = "─" * 58
    block = (
        f"\n{sep}\n"
        f"Round {rnum}  |  Student {sid}  |  "
        f"Answer: {last_line}  {mark}\n"
        f"{sep}\n"
        f"{raw}\n"
    )
    _append(resp_file, block)

    return {
        "round_results": [{
            "student_id": sid,
            "answer":     last_line,
            "reasoning":  raw,
            "correct":    correct,
        }]
    }


# ─────────────────────────────────────────────────────────────
# Math text cleaner
# ─────────────────────────────────────────────────────────────

def clean_math_text(text: str) -> str:
    r"""
    Normalise LaTeX math formatting for plain-text output:

    1. Convert all \begin{...}...\end{...} environments to $$...$$ blocks.
       Handles equation, align, gather, multline, eqnarray (starred variants too).
    2. Collapse any whitespace/newlines inside a $...$ inline span onto one line.
    3. Ensure every $$...$$ block:
         - has its content on a single line (no embedded newlines)
         - is preceded and followed by exactly one blank line
    4. Remove stray leading/trailing blank lines that accumulate.
    """
    import re

    # ── step 1: \begin{env}...\end{env}  →  $$...$$ ─────────────────────────
    env_pat = re.compile(
        r'\\begin\{(equation|align|gather|multline|eqnarray)\*?\}'
        r'(.*?)'
        r'\\end\{\1\*?\}',
        re.DOTALL | re.IGNORECASE,
    )
    def _env_to_display(m):
        inner = m.group(2).strip()
        # collapse internal newlines to spaces for single-line display
        inner = re.sub(r'\s*\n\s*', ' ', inner)
        # strip alignment markers
        inner = inner.replace('\\\\', ' ').replace('&', '')
        inner = re.sub(r' {2,}', ' ', inner).strip()
        return f'$$\n{inner}\n$$'
    text = env_pat.sub(_env_to_display, text)

    # ── step 2: inline $...$ — collapse internal newlines ────────────────────
    # Match $...$ that don't contain another $ (non-greedy, no newline spanning $$)
    inline_pat = re.compile(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', re.DOTALL)
    def _fix_inline(m):
        inner = re.sub(r'\s*\n\s*', ' ', m.group(1)).strip()
        return f'${inner}$'
    text = inline_pat.sub(_fix_inline, text)

    # ── step 3: $$...$$ — put content on one line, wrap with blank lines ─────
    display_pat = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)
    def _fix_display(m):
        inner = re.sub(r'\s*\n\s*', ' ', m.group(1)).strip()
        return f'\n\n$${inner}$$\n\n'
    text = display_pat.sub(_fix_display, text)

    # ── step 4: clean up excess blank lines ──────────────────────────────────
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────
# Node: collect_results
# ─────────────────────────────────────────────────────────────

def collect_results(state: TutorState) -> dict:
    results   = state.get("round_results", [])
    n_correct = sum(1 for r in results if r["correct"])
    n_wrong   = state["num_students"] - n_correct

    print(f"\n  [COLLECT]  correct={n_correct}  wrong={n_wrong}  "
          f"(stop when correct ≤ {state['max_correct']})")

    out_file = os.path.join(state["output_dir"], "all_problems.txt")
    rnum     = state["problem_count"]

    bar = "".join(
        "✓" if r["correct"] else "✗"
        for r in sorted(results, key=lambda x: x["student_id"])
    )

    context_block  = clean_math_text(state.get("context",  ""))
    question_block = clean_math_text(state.get("question", ""))
    solution_block = clean_math_text(state.get("solution", ""))

    def _section(label: str, text: str) -> str:
        return f"{label}\n{text}\n"

    sep = "─" * 58

    block = (
        f"\n{sep}\n"
        f"Round {rnum}\n"
        f"{sep}\n\n"
        + _section("Problem Context:", context_block)
        + ("\n" + _section("Question:", question_block) if question_block else "")
        + ("\n" + _section("Solution:", solution_block) if solution_block else "")
        + f"\nAnswer   : {state['reference_answer']}\n"
        f"Students : {bar}  (correct={n_correct}  wrong={n_wrong})\n"
        f"{sep}\n"
    )
    _append(out_file, block)
    print(f"  [OUTPUT]  Appended round {rnum} → {out_file}")

    round_summary = {
        "round":            rnum,
        "context":          state.get("context", ""),
        "question":         state.get("question", ""),
        "solution":         state.get("solution", ""),
        "reference_answer": state["reference_answer"],
        "results":          results,
        "n_correct":        n_correct,
        "n_wrong":          n_wrong,
    }

    return {
        "all_rounds":    [round_summary],
        "round_results": [],
    }


# ─────────────────────────────────────────────────────────────
# Conditional edge: loop or finish
# ─────────────────────────────────────────────────────────────

def should_loop(state: TutorState) -> str:
    last      = state["all_rounds"][-1]
    n_correct = last["n_correct"]
    count     = state["problem_count"]

    if count >= state["max_problems"]:
        print(f"  [ROUTER]  Hit problem cap ({state['max_problems']}). Finishing.")
        return "summarise"

    if n_correct <= state["max_correct"]:
        print(f"  [ROUTER]  Only {n_correct} correct "
              f"(≤ {state['max_correct']}) — hard enough. Finishing.")
        return "summarise"

    print(f"  [ROUTER]  {n_correct} correct > {state['max_correct']} — "
          f"too easy. Asking writer for a harder problem.")
    return "writer"


# ─────────────────────────────────────────────────────────────
# Node: summarise
# ─────────────────────────────────────────────────────────────

def summarise_node(state: TutorState) -> dict:
    rounds    = state.get("all_rounds", [])
    last      = rounds[-1] if rounds else {}
    count     = state.get("problem_count", 0)
    mc        = state["max_correct"]
    ns        = state["num_students"]
    n_correct = last.get("n_correct", 0)

    if count >= state["max_problems"] and n_correct > mc:
        reason = f"reached the {state['max_problems']}-problem limit"
    elif not rounds:
        reason = "writer failed to produce a valid problem after retries"
    else:
        reason = (f"problem stumped enough students "
                  f"({n_correct} ≤ {mc} correct out of {ns})")

    header = (
        f"╔══════════════════════════════════════════════════════════════╗\n"
        f"║           PHYSICS TUTOR — SESSION SUMMARY                    ║\n"
        f"╚══════════════════════════════════════════════════════════════╝\n"
        f"Stopped because : {reason}\n"
        f"Total rounds    : {count}\n"
        f"Writer model    : {state['writer_model']}\n"
        f"Student model   : {state['student_model']}\n"
        f"Students        : {ns}\n"
        f"Stop threshold  : ≤ {mc} correct\n"
    )

    rows: list[str] = [header]
    for r in rounds:
        bar = "".join(
            "✓" if res["correct"] else "✗"
            for res in sorted(r["results"], key=lambda x: x["student_id"])
        )
        prob_q     = r.get("question") or r.get("context", "")
        prob_short = (prob_q[:75] + "…") if len(prob_q) > 75 else prob_q
        rows.append(
            f"  Round {r['round']:>2}  ans={r['reference_answer']:>6}  "
            f"{bar}  correct={r['n_correct']}"
        )
        rows.append(f"           {prob_short}\n")

    report = "\n".join(rows)
    print("\n" + report)

    summary_path = os.path.join(state["output_dir"], "summary.txt")
    _write(summary_path, report + "\n")
    print(f"  [OUTPUT]  Summary written → {summary_path}")

    return {"stop_reason": reason}


# ─────────────────────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(TutorState)

    g.add_node("writer",           writer_node)
    g.add_node("validate_writer",  validate_writer)
    g.add_node("verify_solution",  verify_solution)
    g.add_node("student",          student_node)
    g.add_node("collect_results",  collect_results)
    g.add_node("summarise",        summarise_node)

    # START → writer → validate
    g.add_edge(START, "writer")
    g.add_edge("writer", "validate_writer")

    # validate → retry writer | verify solution | give up
    g.add_conditional_edges(
        "validate_writer",
        route_after_validate,
        {"writer": "writer", "verify_solution": "verify_solution", "summarise": "summarise"},
    )

    # verify → fan-out to students (Send) | retry writer | give up
    g.add_conditional_edges(
        "verify_solution",
        route_after_verify,
        {"writer": "writer", "student": "student", "summarise": "summarise"},
    )

    # students → collect
    g.add_edge("student", "collect_results")

    # collect → loop or finish
    g.add_conditional_edges(
        "collect_results",
        should_loop,
        {"writer": "writer", "summarise": "summarise"},
    )

    g.add_edge("summarise", END)

    return g.compile()


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def run_tutor(
    directories:   list[str],
    num_students:  int  = DEFAULT_NUM_STUDENTS,
    max_correct:   int  = DEFAULT_MAX_CORRECT,
    max_problems:  int  = DEFAULT_MAX_PROBLEMS,
    output_dir:     str  = DEFAULT_OUTPUT_DIR,
    summary_dir:    str  = DEFAULT_OUTPUT_DIR,
    writer_model:          str  = DEFAULT_WRITER_MODEL,
    student_model:         str  = DEFAULT_STUDENT_MODEL,
    num_reviewer_samples:  int  = DEFAULT_REVIEWER_SAMPLES,
    reasoning_max_chars:   int   = DEFAULT_REASONING_MAX_CHARS,
    writer_temperature:    float = DEFAULT_WRITER_TEMPERATURE,
) -> TutorState:
    latex_paths = find_latex_files(directories)
    if not latex_paths:
        print(f"[WARNING] No .tex/.latex files found in: {directories}", file=sys.stderr)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    problems_log = os.path.join(output_dir, "all_problems.txt")
    if Path(problems_log).exists():
        raise FileExistsError(
            f"Output file already exists: {problems_log}\n"
            f"Move or delete it (or choose a different --output directory) "
            f"to avoid overwriting previous results."
        )
    _write(problems_log,
           f"Physics Tutor Session\nWriter model: {writer_model}  Student model: {student_model}\n{'═'*60}\n")

    graph = build_graph()

    initial: TutorState = {
        "num_students":       num_students,
        "max_correct":        max_correct,
        "max_problems":       max_problems,
        "output_dir":         output_dir,
        "summary_dir":        summary_dir,
        "writer_model":          writer_model,
        "student_model":         student_model,
        "num_reviewer_samples":  num_reviewer_samples,
        "reasoning_max_chars":   reasoning_max_chars,
        "writer_temperature":    writer_temperature,
        "latex_paths":        latex_paths,
        "notes_text":         "",
        "notes_summary":      "",
        "context":            "",
        "question":           "",
        "solution":           "",
        "reference_answer":   "",
        "writer_retry_count": 0,
        "solution_verified":  False,
        "verify_reason":      "",
        "round_results":      [],
        "problem_count":      0,
        "stop_reason":        "",
        "all_rounds":         [],
    }

    print("\n" + "═"*62)
    print("  PHYSICS TUTOR  —  Adaptive Difficulty Loop")
    print(f"  Writer model : {writer_model}")
    print(f"  Student model: {student_model}")
    print(f"  Students     : {num_students}")
    print(f"  Reviewer sees: {num_reviewer_samples} student(s) per round")
    print(f"  Reasoning cap: {reasoning_max_chars} chars per student")
    print(f"  Writer temp  : {writer_temperature}")
    print(f"  Stop when    : ≤ {max_correct} students correct")
    print(f"  Problem cap. : {max_problems}")
    print(f"  Output       : {output_dir}/")
    print(f"  Notes Summary: {summary_dir}/")
    print(f"  LaTeX files  : {len(latex_paths)} found")
    print("═"*62)

    return graph.invoke(initial)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adaptive physics problem generator and student evaluator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("directories", nargs="+", metavar="DIR",
                   help="One or more directories containing .tex/.latex notes files.")
    p.add_argument("--students",    "-s", type=int, default=DEFAULT_NUM_STUDENTS,
                   metavar="N", help="Number of parallel student agents per round.")
    p.add_argument("--max-correct", "-c", type=int, default=DEFAULT_MAX_CORRECT,
                   metavar="N", help="Stop when ≤ N students answer correctly.")
    p.add_argument("--rounds",      "-r", type=int, default=DEFAULT_MAX_PROBLEMS,
                   metavar="N", help="Maximum number of problems to attempt.")
    p.add_argument("--output",      "-o", type=str, default=DEFAULT_OUTPUT_DIR,
                   metavar="DIR", help="Directory for output files.")
    p.add_argument("--summary-dir", "-d", type=str, default=DEFAULT_OUTPUT_DIR,
                   metavar="DIR", help="Directory for notes summary file.")
    p.add_argument("--reviewer-samples", "-R", type=int, default=DEFAULT_REVIEWER_SAMPLES,
                   metavar="N",
                   help="How many student attempts the writer reviews each round (stratified sample).")
    p.add_argument("--reasoning-chars", "-C", type=int, default=DEFAULT_REASONING_MAX_CHARS,
                   metavar="N",
                   help="Max characters of each student's reasoning sent to the writer.")
    p.add_argument("--writer-temp", "-t", type=float, default=DEFAULT_WRITER_TEMPERATURE,
                   metavar="T",
                   help="Temperature for the writer/verifier/summariser (0.0–2.0; ignored for reasoning models).")
    p.add_argument("--writer-model",  "-m", type=str, default=DEFAULT_WRITER_MODEL,
                   help="OpenAI model for the writer/verifier/summariser (e.g. gpt-5.5, o3).")
    p.add_argument("--student-model", "-M", type=str, default=DEFAULT_STUDENT_MODEL,
                   help="OpenAI model for student agents (e.g. gpt-4o, gpt-4.1).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠  OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    run_tutor(
        directories  = args.directories,
        num_students = args.students,
        max_correct  = args.max_correct,
        max_problems = args.rounds,
        output_dir   = args.output,
        summary_dir  = args.summary_dir,
        writer_model          = args.writer_model,
        student_model         = args.student_model,
        num_reviewer_samples  = args.reviewer_samples,
        reasoning_max_chars   = args.reasoning_chars,
        writer_temperature    = args.writer_temp,
    )
