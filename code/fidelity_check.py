"""
fidelity_check.py — Grounding-fidelity evaluation for the ENAMED audit LLM layer.

This is the DP3 numerical-matching evaluation
engine. For each course it takes the *audited payload* (the pre-computed numbers
the ML pipeline produced) and a *generated explanation text*, extracts every
number that appears in the text, and checks whether each number traces back to
the payload. It aggregates over all courses and contrasts a GROUNDED arm against
an UNGROUNDED baseline.

Two arms:
  - grounded   : text produced by `local_explanation` (deterministic, offline)
                 or by `explain` (live Gemini, grounded prompt). Expectation:
                 fidelity = 1.0 by construction for the deterministic generator.
  - ungrounded : text produced by an open-prompt generator that is *allowed* to
                 add context/benchmarks. `explain_ungrounded` (live Gemini) is the
                 live open-prompt baseline; `naive_explanation` is a
                 deterministic offline stand-in for demos with no API key.

Nothing here needs the network unless you explicitly select a live-Gemini arm.

Usage
-----
    # self-test on synthetic, schema-correct payloads (no data, no key):
    python fidelity_check.py --selftest

    # over real payloads once the pipeline has produced them:
    python fidelity_check.py --payloads ../data/processed/course_payloads.json \
                             --grounded local --baseline naive --out fidelity_report.json
"""

from __future__ import annotations

import re
import json
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

# The audit LLM layer under test (repo module).
import sys, os

from llm_explainer import AuditContext, local_explanation, explain  # noqa: E402

# ---------------------------------------------------------------------------
# Number extraction and matching
# ---------------------------------------------------------------------------

# Matches integers and decimals with either '.' or ',' as the decimal mark,
# optionally followed by '%'. Grouping separators are not expected in this text.
_NUM_RE = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)(\s*%)?")


def extract_numbers(text: str) -> List[float]:
    """Every numeric literal in `text`, as floats. '65,18' -> 65.18, '51.2%' -> 51.2."""
    out: List[float] = []
    for m in _NUM_RE.finditer(text):
        raw = m.group(1).replace(",", ".")
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def grounded_number_set(ctx: AuditContext) -> List[float]:
    """Every number the explanation is *permitted* to state, drawn from the payload.

    Includes the raw payload fields plus the roundings the generators legitimately
    print (avg .2f, value_pct .1f, impact .2f, confidence .1f, |delta| .2f).
    """
    vals: List[float] = []

    def add(x: Optional[float]) -> None:
        if x is None:
            return
        vals.append(float(x))

    add(ctx.course_code)
    add(ctx.avg_score)
    add(ctx.national_median)
    delta = round(ctx.avg_score - ctx.national_median, 2)
    add(delta)
    add(abs(delta))
    add(ctx.confidence_pct)
    for c in ctx.contributions:
        add(c.get("value_pct"))
        add(c.get("impact"))
    # printed roundings
    rounded: List[float] = []
    for v in vals:
        rounded.append(round(v, 2))
        rounded.append(round(v, 1))
        rounded.append(float(round(v)))  # integer form (e.g. course_code, "1 point")
    return sorted(set(vals) | set(rounded))


def is_grounded(num: float, allowed: List[float], tol: float = 0.05) -> bool:
    """A number is grounded if it matches any allowed payload value within tol."""
    return any(abs(num - a) <= tol for a in allowed)


@dataclass
class TextCheck:
    n_numbers: int
    n_grounded: int
    ungrounded: List[float]

    @property
    def fidelity(self) -> float:
        return 1.0 if self.n_numbers == 0 else self.n_grounded / self.n_numbers


def check_text(ctx: AuditContext, text: str, tol: float = 0.05) -> TextCheck:
    allowed = grounded_number_set(ctx)
    nums = extract_numbers(text)
    ung = [n for n in nums if not is_grounded(n, allowed, tol)]
    return TextCheck(n_numbers=len(nums), n_grounded=len(nums) - len(ung), ungrounded=ung)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def naive_explanation(ctx: AuditContext) -> str:
    """Deterministic OFFLINE stand-in for an ungrounded LLM.

    Illustrates the failure mode: it phrases the audited numbers but *also* adds
    plausible-sounding statistics that are NOT in the payload (a fabricated
    national pass rate, a percentile rank, a cohort size, a YoY change). These
    are exactly the numbers a grounded layer must never emit. Use only for demos
    without an API key; for the paper's real baseline use `explain_ungrounded`.
    """
    pt = ctx.language.startswith("Port")
    # numbers below are INVENTED on purpose and absent from the payload:
    fake_national = round(ctx.national_median + 7.3, 1)   # bogus "national average"
    fake_pct = 82                                          # bogus percentile
    fake_cohort = 128                                      # bogus cohort size
    fake_yoy = 4.6                                         # bogus year-over-year gain
    if pt:
        return (
            f"O curso {ctx.course_code} teve média {ctx.avg_score:.2f}, contra uma "
            f"média nacional de cerca de {fake_national}. Isso o coloca aproximadamente "
            f"no percentil {fake_pct} entre os {fake_cohort} programas avaliados, uma "
            f"melhora de {fake_yoy}% em relação ao ano anterior."
        )
    return (
        f"Course {ctx.course_code} scored {ctx.avg_score:.2f}, versus a national average "
        f"of about {fake_national}. That places it around the {fake_pct}th percentile among "
        f"the {fake_cohort} programs assessed, a {fake_yoy}% gain over last year."
    )


def explain_ungrounded(ctx: AuditContext):
    """Live-Gemini baseline with a permissive (ungrounded) prompt. Needs a key.

    Returns the text or None. Kept minimal: reuses the repo's client indirectly by
    monkeypatching the system instruction would be brittle, so this simply asks the
    same model with an open prompt via the public `explain` path is NOT possible
    (its prompt is fixed). For the paper's live baseline, call the model here with a
    system instruction that *invites* external context. Implemented lazily to avoid
    importing google-genai unless used.
    """
    import os as _os
    key = _os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from google import genai
        from google.genai import types
    except Exception:
        return None
    payload = json.dumps(ctx.to_payload(), ensure_ascii=False, sort_keys=True)
    open_prompt_sys = (
        "You explain a medical course's audit results to its coordinator. "
        "Feel free to add helpful context, national benchmarks, percentiles, and "
        "typical figures so the coordinator can situate the result."
    )
    try:
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=30000))
        cfg = types.GenerateContentConfig(
            system_instruction=open_prompt_sys, max_output_tokens=1200,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        )
        r = client.models.generate_content(
            model="gemini-3.5-flash",
            contents="Here are the audited numbers as JSON:\n\n" + payload,
            config=cfg,
        )
        return (getattr(r, "text", "") or "").strip() or None
    except Exception:
        return None


GENERATORS: Dict[str, Callable[[AuditContext], Optional[str]]] = {
    "local": local_explanation,        # grounded, offline, deterministic
    "gemini": explain,                 # grounded, live (needs key)
    "naive": naive_explanation,        # ungrounded, offline, deterministic
    "gemini_open": explain_ungrounded, # ungrounded, live (needs key)
}


# ---------------------------------------------------------------------------
# Aggregation over courses
# ---------------------------------------------------------------------------

def payload_to_ctx(p: Dict[str, Any]) -> AuditContext:
    return AuditContext(
        course_code=int(p["course_code"]),
        avg_score=float(p["avg_score"]),
        national_median=float(p["national_median"]),
        predicted_tier=str(p["predicted_tier"]),
        contributions=list(p.get("contributions", [])),
        language=p.get("language", "English"),
        confidence_pct=p.get("confidence_pct"),
    )


def run_arm(payloads: List[Dict[str, Any]], gen_key: str, tol: float = 0.05) -> Dict[str, Any]:
    gen = GENERATORS[gen_key]
    per_course = []
    total_nums = total_grounded = 0
    n_clean = 0            # courses with zero ungrounded numbers
    skipped = 0
    for p in payloads:
        ctx = payload_to_ctx(p)
        text = gen(ctx)
        if text is None:   # live arm with no key / failure
            skipped += 1
            continue
        chk = check_text(ctx, text, tol)
        total_nums += chk.n_numbers
        total_grounded += chk.n_grounded
        if not chk.ungrounded:
            n_clean += 1
        per_course.append({
            "course_code": ctx.course_code,
            "n_numbers": chk.n_numbers,
            "n_grounded": chk.n_grounded,
            "ungrounded": chk.ungrounded,
            "fidelity": round(chk.fidelity, 4),
        })
    n = len(per_course)
    return {
        "arm": gen_key,
        "n_courses": n,
        "n_courses_skipped": skipped,
        "n_courses_fully_grounded": n_clean,
        "pct_courses_fully_grounded": round(n_clean / n, 4) if n else None,
        "numbers_total": total_nums,
        "numbers_grounded": total_grounded,
        "numbers_ungrounded": total_nums - total_grounded,
        "micro_fidelity": round(total_grounded / total_nums, 4) if total_nums else None,
        "per_course": per_course,
    }


def compare(payloads: List[Dict[str, Any]], grounded: str, baseline: str,
            tol: float = 0.05) -> Dict[str, Any]:
    return {
        "n_payloads": len(payloads),
        "tolerance": tol,
        "grounded_arm": run_arm(payloads, grounded, tol),
        "baseline_arm": run_arm(payloads, baseline, tol),
    }


# ---------------------------------------------------------------------------
# Self-test on synthetic, schema-correct payloads
# ---------------------------------------------------------------------------

def _synthetic_payloads(n: int = 350, seed: int = 42) -> List[Dict[str, Any]]:
    import random
    rng = random.Random(seed)
    labels = [
        ("I7_D", "Learned many contents", "strength"),
        ("I9_A", "Practical activities helped", "strength"),
        ("I4_B", "Clear question stems", "strength"),
        ("I1_C", "Difficulty: Medium", "neutral"),
        ("I6_A", "Lack of content knowledge", "friction"),
        ("I1_D", "Exam considered hard", "friction"),
    ]
    median = 65.18
    out = []
    for i in range(n):
        avg = round(rng.uniform(52, 80), 2)
        contribs = []
        for code, lab, sen in labels:
            contribs.append({
                "feature": code, "label": lab,
                "value_pct": round(rng.uniform(5, 70), 1),
                "impact": round(rng.uniform(1.5, 8.0), 2),
                "sentiment": sen,
            })
        out.append({
            "course_code": 10000 + i,
            "avg_score": avg,
            "national_median": median,
            "predicted_tier": "High Performance" if avg >= median else "Low Performance",
            "confidence_pct": round(rng.uniform(51, 99), 1),
            "contributions": contribs,
            "language": "Português" if i % 2 else "English",
        })
    return out


def _selftest() -> int:
    payloads = _synthetic_payloads()
    res = compare(payloads, grounded="local", baseline="naive")
    g, b = res["grounded_arm"], res["baseline_arm"]
    print("=== Grounding-fidelity self-test (synthetic payloads) ===")
    print(f"payloads: {res['n_payloads']}  tolerance: ±{res['tolerance']}\n")
    print(f"GROUNDED  (local_explanation, deterministic):")
    print(f"  numbers checked        : {g['numbers_total']}")
    print(f"  ungrounded numbers     : {g['numbers_ungrounded']}")
    print(f"  micro-fidelity         : {g['micro_fidelity']}")
    print(f"  courses fully grounded : {g['n_courses_fully_grounded']}/{g['n_courses']} "
          f"({g['pct_courses_fully_grounded']})\n")
    print(f"UNGROUNDED baseline (naive_explanation):")
    print(f"  numbers checked        : {b['numbers_total']}")
    print(f"  ungrounded numbers     : {b['numbers_ungrounded']}")
    print(f"  micro-fidelity         : {b['micro_fidelity']}")
    print(f"  courses fully grounded : {b['n_courses_fully_grounded']}/{b['n_courses']} "
          f"({b['pct_courses_fully_grounded']})")
    ok = (g["numbers_ungrounded"] == 0 and b["numbers_ungrounded"] > 0)
    print("\nRESULT:", "PASS — grounded arm emits no ungrounded number; baseline does."
          if ok else "FAIL — unexpected outcome, inspect the checker.")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Grounding-fidelity check for the audit LLM layer.")
    ap.add_argument("--selftest", action="store_true", help="run on synthetic payloads and exit")
    ap.add_argument("--payloads", help="JSON file: list of course payloads")
    ap.add_argument("--grounded", default="local", choices=list(GENERATORS), help="grounded arm generator")
    ap.add_argument("--baseline", default="naive", choices=list(GENERATORS), help="baseline arm generator")
    ap.add_argument("--tol", type=float, default=0.05, help="numeric match tolerance")
    ap.add_argument("--out", help="write full JSON report here")
    args = ap.parse_args()

    if args.selftest or not args.payloads:
        return _selftest()

    with open(args.payloads, encoding="utf-8") as fh:
        payloads = json.load(fh)
    res = compare(payloads, args.grounded, args.baseline, args.tol)
    g, b = res["grounded_arm"], res["baseline_arm"]
    print(f"GROUNDED   [{g['arm']}]  micro-fidelity={g['micro_fidelity']}  "
          f"ungrounded={g['numbers_ungrounded']}  clean_courses={g['pct_courses_fully_grounded']}")
    print(f"BASELINE   [{b['arm']}]  micro-fidelity={b['micro_fidelity']}  "
          f"ungrounded={b['numbers_ungrounded']}  clean_courses={b['pct_courses_fully_grounded']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
