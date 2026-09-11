"""Optional live-model experiment. Requires an API key and incurs API charges.
The API key is sent to the model provider for authentication.
Saved historical reports are supplied; no new live run was performed during revision.
"""
from __future__ import annotations
import os, re, json, argparse
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = "gemini-3.5-flash"

# ---------------- audit context ----------------
@dataclass
class AuditContext:
    course_code: int
    avg_score: float
    national_median: float
    predicted_tier: str
    contributions: List[Dict[str, Any]]
    language: str = "English"
    confidence_pct: Optional[float] = None
    def to_payload(self) -> Dict[str, Any]:
        d = asdict(self); d.pop("language", None)
        d["delta_vs_median"] = round(self.avg_score - self.national_median, 2)
        return d

SYS_GROUNDED = (
    "You explain PRE-COMPUTED audit results from an ML model to a medical-course "
    "coordinator. You receive a JSON object whose numbers were already calculated. "
    "Rules: (1) Use ONLY the numbers and labels present in the input. NEVER invent "
    "values, percentages, institution names, or facts not there. (2) If a value is "
    "absent, do not mention it. (3) Be concise and concrete. (4) English. (5) One "
    "short summary paragraph; then 2-3 action items derived strictly from items whose "
    "sentiment is 'friction'; then a one-line caveat that these are model-derived "
    "suggestions to be validated by the coordinator. Do not echo the JSON.")

SYS_OPEN = (
    "You explain a medical course's audit results to its coordinator. Feel free to add "
    "helpful context, national benchmarks, percentiles, and typical figures so the "
    "coordinator can situate the result.")

def _client():
    from google import genai
    from google.genai import types
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment first.")
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=30000)), types

def _gemini(ctx: AuditContext, system: str) -> Optional[str]:
    client, types = _client()
    payload = json.dumps(ctx.to_payload(), ensure_ascii=False, sort_keys=True)
    cfg = types.GenerateContentConfig(
        system_instruction=system, max_output_tokens=1200,
        thinking_config=types.ThinkingConfig(thinking_level="low"))
    r = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents="Here are the pre-computed audit numbers as JSON:\n\n" + payload,
        config=cfg)
    return (getattr(r, "text", "") or "").strip() or None

def explain(ctx):            return _gemini(ctx, SYS_GROUNDED)
def explain_ungrounded(ctx): return _gemini(ctx, SYS_OPEN)

# offline deterministic references (for comparison without spending API)
def local_explanation(ctx: AuditContext) -> str:
    delta = round(ctx.avg_score - ctx.national_median, 2); above = delta >= 0
    fr = sorted([c for c in ctx.contributions if c.get("sentiment") == "friction"],
                key=lambda c: c.get("impact", 0), reverse=True)[:3]
    stg = sorted([c for c in ctx.contributions if c.get("sentiment") == "strength"],
                 key=lambda c: c.get("impact", 0), reverse=True)[:2]
    conf = f" ({ctx.confidence_pct:.1f}%)" if ctx.confidence_pct is not None else ""
    lines = [f"Course {ctx.course_code} was classified as \"{ctx.predicted_tier}\"{conf}. "
             f"The course mean ({ctx.avg_score:.2f}) is {abs(delta):.2f} point(s) "
             f"{'above' if above else 'below'} the national median ({ctx.national_median:.2f})."]
    if stg: lines.append("Key strengths: " + ", ".join(f"{c['label']} ({c['impact']:.2f})" for c in stg) + ".")
    if fr:
        lines.append("Action items (friction points):")
        for c in fr:
            v = f"{c['value_pct']:.1f}% — " if c.get("value_pct") is not None else ""
            lines.append(f"  - {c['label']} ({c['feature']}): {v}impact {c['impact']:.2f}.")
    lines.append("Note: model-derived suggestions, to be validated by the coordinator.")
    return "\n".join(lines)

def naive_explanation(ctx: AuditContext) -> str:
    fake_nat = round(ctx.national_median + 7.3, 1)
    return (f"Course {ctx.course_code} scored {ctx.avg_score:.2f}, versus a national average of "
            f"about {fake_nat}. That places it around the 82nd percentile among the 128 programs "
            f"assessed, a 4.6% gain over last year.")

# ---------------- number checker ----------------
_NUM_RE = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)(\s*%)?")
def extract_numbers(t):
    out=[]
    for m in _NUM_RE.finditer(t):
        try: out.append(float(m.group(1).replace(",", ".")))
        except: pass
    return out
def grounded_set(ctx):
    vals=[]
    def add(x):
        if x is not None: vals.append(float(x))
    add(ctx.course_code); add(ctx.avg_score); add(ctx.national_median)
    d=round(ctx.avg_score-ctx.national_median,2); add(d); add(abs(d)); add(ctx.confidence_pct)
    for c in ctx.contributions: add(c.get("value_pct")); add(c.get("impact"))
    r=[]
    for v in vals: r+= [round(v,2),round(v,1),float(round(v))]
    return sorted(set(vals)|set(r))
def check(ctx, text, tol=0.05):
    allowed=grounded_set(ctx); nums=extract_numbers(text)
    ung=[n for n in nums if not any(abs(n-a)<=tol for a in allowed)]
    return len(nums), len(nums)-len(ung), ung

def run_arm(payloads, gen, needs_api):
    tot=g=0; clean=0; failed=0; per=[]
    for p in payloads:
        ctx=AuditContext(int(p["course_code"]),float(p["avg_score"]),float(p["national_median"]),
                         str(p["predicted_tier"]),list(p.get("contributions",[])),
                         p.get("language","English"),p.get("confidence_pct"))
        try:
            txt=gen(ctx)
        except Exception as e:
            failed+=1
            print(f"  [skip course {ctx.course_code}] {type(e).__name__}: {e}")
            # No key at all -> stop this arm immediately instead of looping.
            if isinstance(e, RuntimeError):
                return None
            continue
        if txt is None:
            failed+=1
            continue
        n,gd,ung=check(ctx,txt); tot+=n; g+=gd
        if not ung: clean+=1
        per.append({"course_code":ctx.course_code,"n_numbers":n,"n_grounded":gd,"ungrounded":ung})
    n=len(per)
    if n==0:
        return None
    return {"n_courses":n,"n_failed":failed,"numbers_total":tot,"numbers_ungrounded":tot-g,
            "micro_fidelity":round(g/tot,4) if tot else None,
            "n_courses_fully_grounded":clean,
            "pct_courses_fully_grounded":round(clean/n,4) if n else None,"per_course":per}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--payloads",default="course_payloads.json")
    ap.add_argument("--n",type=int,default=60,help="sample size (default 60)")
    ap.add_argument("--all",action="store_true",help="use all 350")
    ap.add_argument("--out",default="fidelity_report_gemini.json")
    a=ap.parse_args()
    pl=json.load(open(a.payloads,encoding="utf-8"))
    if not a.all: pl=pl[:a.n]
    print(f"Running on {len(pl)} courses. Grounded+open = {2*len(pl)} Gemini calls.")
    arms={}
    print("arm: gemini (grounded)...");     arms["gemini_grounded"]=run_arm(pl,explain,True)
    print("arm: gemini_open (ungrounded)..."); arms["gemini_open"]=run_arm(pl,explain_ungrounded,True)
    print("arm: local (offline ref)...");    arms["local_offline"]=run_arm(pl,local_explanation,False)
    print("arm: naive (offline ref)...");    arms["naive_offline"]=run_arm(pl,naive_explanation,False)
    json.dump({"model":DEFAULT_MODEL,"n":len(pl),"arms":arms},
              open(a.out,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print("\n=== SUMMARY ===")
    for k,v in arms.items():
        if v: print(f"{k:18s} fidelity={v['micro_fidelity']}  ungrounded={v['numbers_ungrounded']}  "
                    f"clean={v['n_courses_fully_grounded']}/{v['n_courses']}")
    print("\nWrote",a.out,"— send me this file.")
