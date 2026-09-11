"""
llm_explainer.py - Grounded LLM explanation layer for the ENAMED audit dashboard.

It consumes ONLY pre-computed audit numbers (course code, score, national median,
predicted tier, and the ranked local feature contributions) and asks a Gemini
model to phrase them as concise, actionable guidance for a course coordinator.

Design principles
-----------------
- Grounded: the model receives only numbers already produced by the ML pipeline.
- Course-level aggregates are sent; public course codes may remain in payloads.
- Fail-safe: on any problem, `explain` returns None and the dashboard falls back
  to its rule-based diagnostic. The *reason* is recorded in get_last_error().

Setup
-----
    pip install "google-genai>=2.0"            # gemini-3.5-flash needs the new SDK
    # key via Streamlit secrets (.streamlit/secrets.toml): GEMINI_API_KEY = "..."
    # or environment variable GEMINI_API_KEY

Run a live diagnostic (prints the REAL cause if the call fails):
    GEMINI_API_KEY=... python llm_explainer.py
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

try:
    import streamlit as st
except Exception:
    st = None

DEFAULT_MODEL = "gemini-3.5-flash"
MAX_OUTPUT_TOKENS = 1200          # must comfortably exceed thinking + answer
THINKING_LEVEL = "low"            # "minimal" | "low" | "medium" | "high"
TIMEOUT_MS = 30000                # fail fast instead of hanging the dashboard

_last_error: Optional[str] = None


def get_last_error() -> Optional[str]:
    """Reason the most recent call returned None (None if it succeeded)."""
    return _last_error


SYSTEM_INSTRUCTION: Dict[str, str] = {
    "English": (
        "You are an assistant that explains PRE-COMPUTED audit results from a "
        "machine-learning model to a medical-course coordinator. You will receive a "
        "JSON object whose numbers were already calculated by the system. Rules: "
        "(1) Use ONLY the numbers and labels present in the input. NEVER invent "
        "values, percentages, institution names, or facts that are not there. "
        "(2) If a value is absent, do not mention it. (3) Be concise and concrete. "
        "(4) Write in English. (5) Structure your answer as: one short paragraph "
        "summarising the situation; then 2-3 bullet action items derived strictly "
        "from the items whose sentiment is 'friction'; then a one-line caveat that "
        "these are model-derived suggestions to be validated by the coordinator. "
        "Do not echo the JSON back."
    ),
    "Português": (
        "Você é um assistente que explica resultados de auditoria JÁ CALCULADOS por "
        "um modelo de aprendizado de máquina para um coordenador de curso de "
        "medicina. Você receberá um objeto JSON cujos números já foram calculados "
        "pelo sistema. Regras: (1) Use APENAS os números e rótulos presentes no "
        "input. NUNCA invente valores, percentuais, nomes de instituições ou fatos "
        "que não estejam ali. (2) Se um valor não existir, não o mencione. (3) Seja "
        "conciso e concreto. (4) Escreva em português. (5) Estruture a resposta "
        "como: um parágrafo curto resumindo a situação; depois 2-3 itens de ação "
        "derivados estritamente dos itens cujo sentimento é 'friction'; e por fim "
        "uma linha de ressalva de que são sugestões derivadas do modelo, a serem "
        "validadas pelo coordenador. Não repita o JSON na saída."
    ),
}


@dataclass
class AuditContext:
    """All the pre-computed numbers the explanation may use - nothing else."""
    course_code: int
    avg_score: float
    national_median: float
    predicted_tier: str
    contributions: List[Dict[str, Any]]
    language: str = "English"
    confidence_pct: Optional[float] = None

    def to_payload(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("language", None)
        d["delta_vs_median"] = round(self.avg_score - self.national_median, 2)
        return d


def _get_api_key() -> Optional[str]:
    if st is not None:
        try:
            key = st.secrets.get("GEMINI_API_KEY")  # type: ignore[attr-defined]
            if key:
                return key
        except Exception:
            pass
    return os.environ.get("GEMINI_API_KEY")


def _call_gemini(payload_json: str, language: str, model: str) -> Optional[str]:
    """Return the explanation text, or None. On None, get_last_error() says why.

    Failures are intentionally NOT cached: caching a transient None used to keep
    the dashboard on the fallback even after the key/quota was fixed.
    """
    global _last_error
    _last_error = None

    api_key = _get_api_key()
    if not api_key:
        _last_error = "No GEMINI_API_KEY found (Streamlit secrets or environment)."
        return None
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        _last_error = f"google-genai import failed (pip install 'google-genai>=2.0'): {e}"
        return None

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=TIMEOUT_MS),
        )
        sys_inst = SYSTEM_INSTRUCTION.get(language, SYSTEM_INSTRUCTION["English"])
        prompt = (
            "Here are the pre-computed audit numbers as JSON. "
            "Explain them following the rules.\n\n" + payload_json
        )
        # Key fix: cap thinking LOW and give the answer room. Gemini 3.x models
        # spend output tokens on internal reasoning; with a low cap the visible
        # text can come back empty (finish_reason=MAX_TOKENS). temperature/top_p
        # are intentionally omitted (not recommended for 3.5 Flash).
        config = types.GenerateContentConfig(
            system_instruction=sys_inst,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        )
        resp = client.models.generate_content(model=model, contents=prompt, config=config)

        text = (getattr(resp, "text", "") or "").strip()
        if text:
            return text

        reason = None
        try:
            reason = resp.candidates[0].finish_reason
        except Exception:
            pass
        _last_error = (
            f"Empty response (finish_reason={reason}). If MAX_TOKENS, raise "
            f"MAX_OUTPUT_TOKENS or lower THINKING_LEVEL; if SAFETY/BLOCKLIST, the "
            f"prompt was filtered."
        )
        return None

    except Exception as e:
        # Surface the real cause. Common: 429 quota/rate-limit; 401/403 bad key;
        # 404 model name; DeadlineExceeded -> timeout.
        _last_error = f"{type(e).__name__}: {e}"
        return None


def local_explanation(ctx: AuditContext) -> str:
    """Deterministic, OFFLINE grounded explanation - no API key, no network.

    Produces the same kind of prose the LLM is asked to write, using ONLY the
    pre-computed numbers in `ctx`. This is what reviewers see when no Gemini key
    is configured, so the replication package runs end-to-end with no secrets.
    By construction it cannot hallucinate: every value comes from the payload.
    """
    pt = ctx.language.startswith("Port")
    delta = round(ctx.avg_score - ctx.national_median, 2)
    above = delta >= 0
    fr = [c for c in ctx.contributions if c.get("sentiment") == "friction"]
    stg = [c for c in ctx.contributions if c.get("sentiment") == "strength"]
    fr = sorted(fr, key=lambda c: c.get("impact", 0), reverse=True)[:3]
    stg = sorted(stg, key=lambda c: c.get("impact", 0), reverse=True)[:2]
    conf = f" ({ctx.confidence_pct:.1f}%)" if ctx.confidence_pct is not None else ""

    if pt:
        lines = [
            f"O curso {ctx.course_code} foi classificado como \"{ctx.predicted_tier}\"{conf}. "
            f"A média do curso ({ctx.avg_score:.2f}) está {abs(delta):.2f} ponto(s) "
            f"{'acima' if above else 'abaixo'} da mediana nacional ({ctx.national_median:.2f})."
        ]
        if stg:
            s_ = ", ".join(f"{c['label']} ({c['impact']:.2f})" for c in stg)
            lines.append(f"Principais forças: {s_}.")
        if fr:
            lines.append("Itens de ação (pontos de atrito):")
            for c in fr:
                v = f"{c['value_pct']:.1f}% — " if c.get("value_pct") is not None else ""
                lines.append(f"  - {c['label']} ({c['feature']}): {v}impacto {c['impact']:.2f}.")
        else:
            lines.append("Nenhum ponto de atrito relevante entre os principais fatores.")
        lines.append("Observação: sugestões derivadas do modelo, a serem validadas pelo coordenador.")
    else:
        lines = [
            f"Course {ctx.course_code} was classified as \"{ctx.predicted_tier}\"{conf}. "
            f"The course mean ({ctx.avg_score:.2f}) is {abs(delta):.2f} point(s) "
            f"{'above' if above else 'below'} the national median ({ctx.national_median:.2f})."
        ]
        if stg:
            s_ = ", ".join(f"{c['label']} ({c['impact']:.2f})" for c in stg)
            lines.append(f"Key strengths: {s_}.")
        if fr:
            lines.append("Action items (friction points):")
            for c in fr:
                v = f"{c['value_pct']:.1f}% — " if c.get("value_pct") is not None else ""
                lines.append(f"  - {c['label']} ({c['feature']}): {v}impact {c['impact']:.2f}.")
        else:
            lines.append("No relevant friction points among the leading factors.")
        lines.append("Note: these are model-derived suggestions, to be validated by the coordinator.")
    return "\n".join(lines)


def explain(ctx: AuditContext, model: str = DEFAULT_MODEL) -> Optional[str]:
    """Return a grounded explanation string, or None (rule-based fallback).

    If this returns None, call get_last_error() to see exactly why.
    """
    payload_json = json.dumps(ctx.to_payload(), ensure_ascii=False, sort_keys=True)
    return _call_gemini(payload_json, ctx.language, model)


if __name__ == "__main__":
    demo = AuditContext(
        course_code=12345,
        avg_score=71.17,
        national_median=65.18,
        predicted_tier="High Performance",
        confidence_pct=57.6,
        contributions=[
            {"feature": "I7_D", "label": "Learned many contents", "value_pct": 60.0,
             "impact": 5.2, "sentiment": "strength"},
            {"feature": "I6_A", "label": "Lack of content knowledge", "value_pct": 51.2,
             "impact": 2.97, "sentiment": "friction"},
            {"feature": "I1_D", "label": "Exam considered hard", "value_pct": 35.7,
             "impact": 2.77, "sentiment": "friction"},
        ],
        language="English",
    )
    out = explain(demo)
    if out:
        print("OK - Gemini returned:\n")
        print(out)
    else:
        print("Gemini not used -", get_last_error())
        print("\nOffline grounded explanation (no key needed):\n")
        print(local_explanation(demo))
