"""Prototype of the field-bound checker (C1-C6) run against the existing
3,822 constructed statements, to produce before/after detection numbers."""
import json, re, collections, sys

import os, argparse
_ap = argparse.ArgumentParser(description="Field-bound verifier prototype (C1-C6).")
_ap.add_argument("--pkg", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 help="Package root containing data/ and results/")
_ap.add_argument("--out", default=None, help="Optional JSON report path")
_args = _ap.parse_args()
PKG = _args.pkg
payloads = json.load(open(f"{PKG}/data/course_payloads.json"))

# ---------- C5: spelled-out number normalisation (English) ----------
UNITS = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
         "eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,
         "fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,
         "nineteen":19}
TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,
        "eighty":80,"ninety":90}

def words_to_num(tokens):
    """Parse a token list like ['seventy','five','point','three','four'] -> 75.34"""
    if "point" in tokens:
        i = tokens.index("point")
        intpart, frac = tokens[:i], tokens[i+1:]
    else:
        intpart, frac = tokens, []
    total = 0
    for t in intpart:
        if t in TENS: total += TENS[t]
        elif t in UNITS: total += UNITS[t]
        elif t == "hundred": total *= 100
        elif t == "and": pass
        else: return None
    digits = ""
    for t in frac:
        if t in UNITS and UNITS[t] < 10: digits += str(UNITS[t])
        else: return None
    return float(f"{total}.{digits}") if digits else float(total)

WORD_RE = re.compile(
    r"\b((?:(?:" + "|".join(list(UNITS)+list(TENS)) + r"|hundred|and)[\s-]+)*"
    r"(?:" + "|".join(list(UNITS)+list(TENS)) + r"|hundred)"
    r"(?:\s+point(?:\s+(?:" + "|".join(list(UNITS)) + r"))+)?)\b", re.I)

SCOPED = True   # False reproduces the naive normalisation and its false alarms

def normalise_words(text):
    def rep(m):
        toks = re.split(r"[\s-]+", m.group(1).lower())
        v = words_to_num(toks)
        if v is None:
            return m.group(1)
        # Scope rule: only treat a spelled-out numeral as a quantity when it
        # carries a decimal ("point") or reaches two digits. Bare small numerals
        # ("one decimal", "two bullet points") are nominal, not claims.
        if SCOPED and "point" not in toks and v < 20:
            return m.group(1)
        return f"{v:g}"
    return WORD_RE.sub(rep, text)

# ---------- number extraction (sign-aware) ----------
NUM_RE = re.compile(r"(-?\d+(?:[.,]\d+)?)(\s*%)?")

# ---------- field lexicon (closed, derived from the payload schema) ----------
FIELD_PATTERNS = [
    (re.compile(r"national median", re.I), "national_median"),
    (re.compile(r"course mean|course score|mean score", re.I), "avg_score"),
    (re.compile(r"confidence", re.I), "confidence_pct"),
]
CMP_RE = re.compile(r"\b(above|below|higher than|lower than|exceeds|under)\b", re.I)
REL_UNIT_RE = re.compile(r"%")
PP_UNIT_RE = re.compile(r"\b(points?|percentage points?|p\.p\.)\b", re.I)
CAUSAL_RE = re.compile(
    r"\b(will raise|will increase|will improve|causes?|caused by|due to|leads? to|"
    r"results? in|drives?|explains?|accounts? for|increasing .{0,40}\bwill\b)\b", re.I)

DOMAIN = {"avg_score": (0, 100), "national_median": (0, 100),
          "confidence_pct": (0, 100), "value_pct": (0, 100), "impact": (0, None)}

def allowed_roundings(v):
    return {round(v, 2), round(v, 1), float(round(v)), v}

def check(text, p):
    """Return set of verdicts other than 'supported'."""
    verdicts = set()
    avg, med = p["avg_score"], p["national_median"]
    delta = round(avg - med, 2)
    rel = round(100 * (avg - med) / med, 2)

    # C6: unsupported claim type
    if CAUSAL_RE.search(text):
        verdicts.add("unsupported-type")

    t = normalise_words(text)                      # C5
    nums = [(float(m.group(1).replace(",", ".")), m.group(2) is not None)
            for m in NUM_RE.finditer(t)]
    if not nums:
        return verdicts                            # no numeric claim to bind

    is_cmp = bool(CMP_RE.search(t))

    for val, had_pct in nums:
        if is_cmp:
            # C2 direction + C3 unit, on the comparison magnitude
            marker = CMP_RE.search(t).group(1).lower()
            claims_above = marker in ("above", "higher than", "exceeds")
            if had_pct or (REL_UNIT_RE.search(t) and not PP_UNIT_RE.search(t)):
                expected = abs(rel)                # C3: relative framing
                unit_claimed = "relative"
            else:
                expected = abs(delta)
                unit_claimed = "pp"
            if abs(val - expected) > 0.05:
                # does it match the *other* framing? then it is a unit error
                other = abs(rel) if unit_claimed == "pp" else abs(delta)
                verdicts.add("unit-mismatch" if abs(val - other) <= 0.05
                             else "field-mismatch")
            if (delta > 0) != claims_above and abs(delta) > 0.1:
                verdicts.add("direction-mismatch")
            continue

        # bind to a field
        field = None
        for pat, name in FIELD_PATTERNS:
            if pat.search(t):
                field = name
                break
        if field is None:
            verdicts.add("unbound")
            continue
        truth = p.get(field)
        if truth is None:
            verdicts.add("unbound"); continue
        lo, hi = DOMAIN.get(field, (None, None))
        if (lo is not None and val < lo) or (hi is not None and val > hi):
            verdicts.add("out-of-range")           # C4
            continue
        if not any(abs(val - a) <= 0.05 for a in allowed_roundings(truth)):
            verdicts.add("field-mismatch")         # C1
    return verdicts

# ---------- run ----------
rows = [json.loads(l) for l in open(f"{PKG}/results/extra/checker_cases.jsonl")]
stat = collections.defaultdict(lambda: {"n":0, "old":0, "new":0})
for r in rows:
    p = payloads[r["program_index"]]
    s = stat[r["category"]]
    s["n"] += 1
    s["old"] += int(r["flagged"])
    s["new"] += int(bool(check(r["text"], p)))

CONTROLS = {"correct_mean","correct_median","correct_rounding","correct_comparison"}
print(f"{'category':<24}{'n':>6}{'old':>8}{'new':>8}   role")
for k in sorted(stat, key=lambda x: (x in CONTROLS, x)):
    s = stat[k]
    role = "CONTROL (0 = good)" if k in CONTROLS else "error (n = good)"
    print(f"{k:<24}{s['n']:>6}{s['old']:>8}{s['new']:>8}   {role}")

err = {k:v for k,v in stat.items() if k not in CONTROLS}
ctl = {k:v for k,v in stat.items() if k in CONTROLS}
print()
print(f"errors detected : old {sum(v['old'] for v in err.values())}/{sum(v['n'] for v in err.values())}"
      f"  ->  new {sum(v['new'] for v in err.values())}/{sum(v['n'] for v in err.values())}")
print(f"false alarms    : old {sum(v['old'] for v in ctl.values())}/{sum(v['n'] for v in ctl.values())}"
      f"  ->  new {sum(v['new'] for v in ctl.values())}/{sum(v['n'] for v in ctl.values())}")

# ---------- permissiveness of the current allowed set ----------
def allowed_set(p):
    vals = [p["course_code"], p["avg_score"], p["national_median"],
            round(p["avg_score"]-p["national_median"],2),
            abs(round(p["avg_score"]-p["national_median"],2)), p.get("confidence_pct")]
    for c in p["contributions"]:
        vals += [c.get("value_pct"), c.get("impact")]
    vals = [v for v in vals if v is not None]
    out = set(vals)
    for v in vals:
        out |= {round(v,2), round(v,1), float(round(v))}
    return out

sizes = [len(allowed_set(p)) for p in payloads]
print(f"\nallowed-set size per course: mean {sum(sizes)/len(sizes):.1f}, "
      f"min {min(sizes)}, max {max(sizes)}")
coll = sum(1 for p in payloads
           if any(abs(p["avg_score"]+7.3 - a) <= 0.05 for a in allowed_set(p)))
print(f"courses where the +7.3 mutated score collides with some allowed value: {coll}/350")

# ---------- optional JSON report ----------
if _args.out:
    report = {
        "suite": "checker_cases.jsonl",
        "n_statements": len(rows),
        "scoped_word_normalisation": SCOPED,
        "by_category": {k: {"cases": v["n"], "deployed_flagged": v["old"],
                            "field_bound_flagged": v["new"],
                            "role": "control" if k in CONTROLS else "error"}
                        for k, v in stat.items()},
        "errors": {"cases": sum(v["n"] for v in err.values()),
                   "deployed_flagged": sum(v["old"] for v in err.values()),
                   "field_bound_flagged": sum(v["new"] for v in err.values())},
        "controls": {"cases": sum(v["n"] for v in ctl.values()),
                     "deployed_false_alarms": sum(v["old"] for v in ctl.values()),
                     "field_bound_false_alarms": sum(v["new"] for v in ctl.values())},
        "permitted_set_size_per_course": {
            "mean": round(sum(sizes) / len(sizes), 2), "min": min(sizes), "max": max(sizes)},
        "mutated_score_collisions": coll,
    }
    os.makedirs(os.path.dirname(os.path.abspath(_args.out)), exist_ok=True)
    with open(_args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print("report written to", _args.out)
