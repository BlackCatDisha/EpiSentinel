"""
episentinel_chatbot.py
======================
Role-based, grounded LLM chatbot module for the EpiSentinel epidemic
prediction system.

Usage (inside a FastAPI router):
---------------------------------
    from episentinel_chatbot import (
        SingleDistrictRequest,
        StateOfficialRequest,
        generate_chat_response,
    )

    @router.post("/chat/district")
    async def chat_district(payload: SingleDistrictRequest):
        return {"response": await generate_chat_response(payload)}

    @router.post("/chat/state")
    async def chat_state(payload: StateOfficialRequest):
        return {"response": await generate_chat_response(payload)}

Environment variables required:
---------------------------------
    GOOGLE_API_KEY   — Gemini API key
    CONTEXT_MD_PATH  — (optional) path to context.md; defaults to ./context.md
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CONTEXT_MD_PATH: str = os.getenv("CONTEXT_MD_PATH", "context.md")

# Districts with risk_score above this threshold are injected into the
# state official prompt. Others are summarised only in aggregate counts.
STATE_RISK_INJECTION_THRESHOLD: float = 0.50

# Gemini model to use. gemini-1.5-flash is fast and cost-effective for
# structured advisory tasks; swap to gemini-1.5-pro for higher reasoning depth.
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class UserRole(str, Enum):
    DISTRICT_HEALTH_OFFICER = "district_health_officer"
    HOSPITAL_MANAGER        = "hospital_manager"
    STATE_OFFICIAL          = "state_official"


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class SHAPDriver(BaseModel):
    """A single SHAP feature contribution for a district prediction."""
    feature:     str   = Field(..., description="Feature name, e.g. 'cases_roll4_mean'")
    display_name: str  = Field(..., description="Human-readable label, e.g. '4-week rolling avg cases'")
    shap_value:  float = Field(..., description="Raw SHAP value (positive = increases risk)")
    feature_value: float = Field(..., description="Actual value of the feature this week")


class SingleDistrictRequest(BaseModel):
    """
    Request payload for district_health_officer and hospital_manager roles.
    Both roles operate on a single district's current prediction.
    """
    role:             UserRole         = Field(..., description="User's role")
    district_name:    str              = Field(..., description="Name of the district")
    risk_score:       float            = Field(..., ge=0.0, le=1.0,
                                               description="Model outbreak probability (0–1)")
    predicted_cases:  float            = Field(..., ge=0.0,
                                               description="Predicted dengue cases next week")
    shap_drivers:     List[SHAPDriver] = Field(..., min_length=1,
                                               description="Top SHAP drivers for this prediction")
    user_message:     str              = Field(..., min_length=1,
                                               description="The user's question or request")

    @field_validator("role")
    @classmethod
    def role_must_be_district_level(cls, v: UserRole) -> UserRole:
        allowed = {UserRole.DISTRICT_HEALTH_OFFICER, UserRole.HOSPITAL_MANAGER}
        if v not in allowed:
            raise ValueError(
                f"SingleDistrictRequest only accepts roles: "
                f"{[r.value for r in allowed]}. "
                f"Use StateOfficialRequest for role '{v.value}'."
            )
        return v


class DistrictSummary(BaseModel):
    """Lightweight district snapshot used inside StateOfficialRequest."""
    district_name:   str              = Field(..., description="Name of the district")
    risk_score:      float            = Field(..., ge=0.0, le=1.0)
    predicted_cases: float            = Field(..., ge=0.0)
    shap_drivers:    List[SHAPDriver] = Field(..., min_length=1)


class StateAggregates(BaseModel):
    """State-level aggregate metrics for the current prediction window."""
    total_predicted_cases: float = Field(..., ge=0.0,
                                         description="Sum of predicted cases across all districts")
    average_risk_score:    float = Field(..., ge=0.0, le=1.0,
                                         description="Mean risk score across all districts")
    active_alerts:         int   = Field(..., ge=0,
                                         description="Number of districts with risk_score > 0.5")


class StateOfficialRequest(BaseModel):
    """
    Request payload for the state_official role.
    Includes aggregate state metrics plus per-district summaries.
    """
    role:         UserRole            = Field(UserRole.STATE_OFFICIAL)
    aggregates:   StateAggregates     = Field(..., description="State-level summary metrics")
    districts:    List[DistrictSummary] = Field(..., min_length=1,
                                                description="All district summaries this week")
    user_message: str                 = Field(..., min_length=1)

    @field_validator("role")
    @classmethod
    def role_must_be_state(cls, v: UserRole) -> UserRole:
        if v != UserRole.STATE_OFFICIAL:
            raise ValueError(
                f"StateOfficialRequest only accepts role 'state_official'. "
                f"Got '{v.value}'."
            )
        return v


# Union type accepted by the core function
ChatRequest = SingleDistrictRequest | StateOfficialRequest


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT.MD LOADER  (read once, cached in memory for the process lifetime)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_context_md() -> str:
    """
    Reads context.md exactly once per process and caches the result.
    Returns an empty string with a warning if the file is not found,
    so the server can still start — the system prompt will degrade gracefully
    and the LLM will be instructed that no SOPs are available.
    """
    path = Path(CONTEXT_MD_PATH)
    try:
        text = path.read_text(encoding="utf-8").strip()
        logger.info("context.md loaded from '%s' (%d chars).", path, len(text))
        return text
    except FileNotFoundError:
        logger.warning(
            "context.md not found at '%s'. "
            "The chatbot will operate without SOP grounding. "
            "Set CONTEXT_MD_PATH env var or place context.md in the working directory.",
            path,
        )
        return ""


def get_context_md() -> str:
    """Public accessor — returns cached SOP text or empty string."""
    return _load_context_md()


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

# Shared guardrail block injected into every system prompt.
_GUARDRAILS = """
STRICT OPERATING RULES — YOU MUST FOLLOW THESE WITHOUT EXCEPTION:

1. GROUNDING: You MUST base every recommendation exclusively on the Standard
   Operating Procedures (SOPs) provided in the <context> block below. Do NOT
   invent, extrapolate, or recommend any protocol, medication dosage, resource
   figure, or action that is not explicitly described in that text.

2. UNCERTAINTY: If the user asks about something that is not covered by the
   provided data or the <context> SOPs, respond with:
   "I don't have sufficient information in the current SOPs to answer that.
   Please consult your state health authority guidelines."

3. DOMAIN RESTRICTION: You are a specialist epidemic advisory assistant for
   the EpiSentinel dengue outbreak prediction system. If the user asks
   anything outside of epidemic/disease prediction, the provided district data,
   or the SOP guidelines — including but not limited to general coding questions,
   creative writing, mathematics, or unrelated health topics — politely decline
   and say:
   "I'm restricted to advising on dengue outbreak response based on EpiSentinel
   predictions and your organisation's SOPs. I can't help with that request."

4. TONE: Be concise, factual, and action-oriented. Prioritise life-safety
   actions. Avoid hedging language when the SOPs are clear.
""".strip()


def _build_system_prompt_dho(district_name: str, context_md: str) -> str:
    sop_block = (
        f"<context>\n{context_md}\n</context>"
        if context_md
        else "<context>NO SOP DOCUMENT AVAILABLE — advise the user to consult "
             "their state health authority directly.</context>"
    )
    return f"""You are EpiSentinel Advisory, an AI assistant embedded in the
district health dashboard. You are speaking with the District Health Officer
(DHO) for {district_name}.

YOUR FOCUS: Tactical, on-the-ground decisions for a SINGLE district.
Translate outbreak prediction data into immediate, prioritised field actions.
Typical concerns include: activating response protocols, deploying field teams,
coordinating with PHCs, issuing public advisories, and requisitioning supplies.

{_GUARDRAILS}

{sop_block}
""".strip()


def _build_system_prompt_hospital(district_name: str, context_md: str) -> str:
    sop_block = (
        f"<context>\n{context_md}\n</context>"
        if context_md
        else "<context>NO SOP DOCUMENT AVAILABLE — advise the user to consult "
             "their state health authority directly.</context>"
    )
    return f"""You are EpiSentinel Advisory, an AI assistant embedded in the
hospital operations dashboard. You are speaking with the Hospital Manager for
a facility in {district_name}.

YOUR FOCUS: Operational facility preparedness. Translate outbreak prediction
data into concrete hospital-level actions. Typical concerns include: surge
capacity planning, bed and ICU allocation, dengue-specific supply procurement
(IV fluids, platelets, NS1 kits), staffing rotas, and patient overflow protocols.

{_GUARDRAILS}

{sop_block}
""".strip()


def _build_system_prompt_state(context_md: str) -> str:
    sop_block = (
        f"<context>\n{context_md}\n</context>"
        if context_md
        else "<context>NO SOP DOCUMENT AVAILABLE — advise the user to consult "
             "their state health authority directly.</context>"
    )
    return f"""You are EpiSentinel Advisory, an AI assistant embedded in the
state-level epidemic command dashboard. You are speaking with a State Health
Official responsible for all districts.

YOUR FOCUS: Strategic, multi-district oversight. Translate aggregate outbreak
intelligence into state-level resource allocation, inter-district coordination,
escalation decisions, and policy actions. Identify which districts need
immediate intervention versus watchful monitoring.

{_GUARDRAILS}

{sop_block}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def _format_shap_drivers(drivers: List[SHAPDriver]) -> str:
    """Formats SHAP drivers into a readable bullet list for the prompt."""
    lines = []
    for d in sorted(drivers, key=lambda x: abs(x.shap_value), reverse=True):
        direction = "↑ increases" if d.shap_value > 0 else "↓ decreases"
        lines.append(
            f"  • {d.display_name}: {d.feature_value:.2f} "
            f"({direction} risk, SHAP={d.shap_value:+.3f})"
        )
    return "\n".join(lines)


def _build_human_message_single(payload: SingleDistrictRequest) -> str:
    """Constructs the full human-turn message for DHO / Hospital Manager."""
    risk_pct   = payload.risk_score * 100
    risk_label = (
        "HIGH"     if payload.risk_score >= 0.65 else
        "MODERATE" if payload.risk_score >= 0.35 else
        "LOW"
    )
    shap_block = _format_shap_drivers(payload.shap_drivers)

    data_block = f"""
=== CURRENT PREDICTION DATA FOR {payload.district_name.upper()} ===
  Outbreak probability (next week) : {risk_pct:.1f}%  [{risk_label} RISK]
  Predicted cases (next week)      : {payload.predicted_cases:.0f}

  Key drivers behind this prediction:
{shap_block}
=========================================================
""".strip()

    return f"{data_block}\n\nMy question: {payload.user_message}"


def _build_human_message_state(payload: StateOfficialRequest) -> str:
    """
    Constructs the full human-turn message for the State Official.
    Only districts with risk_score > STATE_RISK_INJECTION_THRESHOLD are
    injected in detail to conserve tokens. The rest are counted in aggregate.
    """
    agg = payload.aggregates

    high_risk_districts = [
        d for d in payload.districts
        if d.risk_score > STATE_RISK_INJECTION_THRESHOLD
    ]
    low_risk_count = len(payload.districts) - len(high_risk_districts)

    # Aggregate block
    state_block = f"""
=== STATE-LEVEL AGGREGATE SUMMARY ===
  Total districts monitored        : {len(payload.districts)}
  Active alerts (risk > 50%)       : {agg.active_alerts}
  Average risk score (state-wide)  : {agg.average_risk_score * 100:.1f}%
  Total predicted cases (next week): {agg.total_predicted_cases:.0f}
  Districts below alert threshold  : {low_risk_count} (details omitted — no immediate action required)
======================================
""".strip()

    # Per-district detail block — only high-risk districts
    if high_risk_districts:
        district_lines = []
        for d in sorted(high_risk_districts, key=lambda x: x.risk_score, reverse=True):
            risk_pct   = d.risk_score * 100
            risk_label = "HIGH" if d.risk_score >= 0.65 else "MODERATE"
            shap_block = _format_shap_drivers(d.shap_drivers)
            district_lines.append(
                f"\n--- {d.district_name} [{risk_label} — {risk_pct:.1f}%] ---\n"
                f"  Predicted cases: {d.predicted_cases:.0f}\n"
                f"  Key drivers:\n{shap_block}"
            )
        district_block = (
            "\n=== HIGH-RISK DISTRICT DETAILS (risk > 50%) ==="
            + "".join(district_lines)
            + "\n================================================"
        )
    else:
        district_block = (
            "\n=== HIGH-RISK DISTRICTS === "
            "No districts currently exceed the 50% risk threshold."
        )

    return (
        f"{state_block}\n{district_block}\n\n"
        f"My question: {payload.user_message}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM CLIENT  (lazy singleton — instantiated on first call)
# ─────────────────────────────────────────────────────────────────────────────

_llm_instance: Optional[ChatGoogleGenerativeAI] = None


def _get_llm() -> ChatGoogleGenerativeAI:
    """
    Returns a cached ChatGoogleGenerativeAI instance.
    Reads GOOGLE_API_KEY from the environment at first call.
    Raises EnvironmentError if the key is missing.
    """
    global _llm_instance
    if _llm_instance is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY environment variable is not set. "
                "The chatbot cannot initialise without a valid Gemini API key."
            )
        _llm_instance = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0.2,      # low temperature for factual, grounded responses
            max_output_tokens=4096,
        )
        logger.info("Gemini LLM initialised (model=%s).", GEMINI_MODEL)
    return _llm_instance


# ─────────────────────────────────────────────────────────────────────────────
# CORE ASYNC FUNCTION  — import this into your FastAPI router
# ─────────────────────────────────────────────────────────────────────────────

async def generate_chat_response(payload: ChatRequest) -> str:
    """
    Main entry point. Accepts a SingleDistrictRequest or StateOfficialRequest,
    constructs the appropriate prompt, calls Gemini, and returns the response
    string.

    Raises:
        EnvironmentError  — if GOOGLE_API_KEY is not set.
        ValueError        — if payload type is unrecognised (should not happen
                            with correct Pydantic validation).

    FastAPI usage:
        @router.post("/chat/district")
        async def chat_district(payload: SingleDistrictRequest):
            response = await generate_chat_response(payload)
            return {"response": response}
    """
    context_md = get_context_md()
    llm        = _get_llm()

    # ── Build role-specific system prompt and human message ─────────────────
    if isinstance(payload, SingleDistrictRequest):
        if payload.role == UserRole.DISTRICT_HEALTH_OFFICER:
            system_text = _build_system_prompt_dho(payload.district_name, context_md)
        elif payload.role == UserRole.HOSPITAL_MANAGER:
            system_text = _build_system_prompt_hospital(payload.district_name, context_md)
        else:
            # Validator should prevent this, but be defensive
            raise ValueError(f"Unexpected role in SingleDistrictRequest: {payload.role}")
        human_text = _build_human_message_single(payload)

    elif isinstance(payload, StateOfficialRequest):
        system_text = _build_system_prompt_state(context_md)
        human_text  = _build_human_message_state(payload)

    else:
        raise ValueError(
            f"Unrecognised payload type: {type(payload)}. "
            "Expected SingleDistrictRequest or StateOfficialRequest."
        )

    # ── Invoke LLM ───────────────────────────────────────────────────────────
    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=human_text),
    ]

    logger.debug(
        "Calling Gemini | role=%s | system_chars=%d | human_chars=%d",
        payload.role.value, len(system_text), len(human_text),
    )

    result = await llm.ainvoke(messages)
    response_text: str = result.content

    logger.info(
        "Gemini response | role=%s | response_chars=%d",
        payload.role.value, len(response_text),
    )

    return response_text
