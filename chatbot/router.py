"""
router.py  —  FastAPI router that wires episentinel_chatbot into HTTP endpoints.
Drop this file into your FastAPI app and include the router.

    # main.py
    from fastapi import FastAPI
    from router import router
    app = FastAPI()
    app.include_router(router)
"""

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from episentinel_chatbot import (
    SingleDistrictRequest,
    StateOfficialRequest,
    generate_chat_response,
    get_context_md,
    GEMINI_MODEL,
)

router = APIRouter(prefix="/chat", tags=["chatbot"])


class ChatResponse(BaseModel):
    response: str


@router.post(
    "/district",
    response_model=ChatResponse,
    summary="Chat endpoint for District Health Officer or Hospital Manager",
)
async def chat_district(payload: SingleDistrictRequest) -> ChatResponse:
    """
    Accepts a SingleDistrictRequest for roles:
      - district_health_officer
      - hospital_manager
    """
    try:
        text = await generate_chat_response(payload)
        return ChatResponse(response=text)
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")


@router.post(
    "/state",
    response_model=ChatResponse,
    summary="Chat endpoint for State Official — multi-district strategic view",
)
async def chat_state(payload: StateOfficialRequest) -> ChatResponse:
    """
    Accepts a StateOfficialRequest for role:
      - state_official
    """
    try:
        text = await generate_chat_response(payload)
        return ChatResponse(response=text)
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")


# ── General dashboard chat ────────────────────────────────────────────────────

class GeneralChatRequest(BaseModel):
    message: str
    district_context: Optional[str] = None  # e.g. "Kolar: risk 77.7%, Critical"


@router.post(
    "/general",
    response_model=ChatResponse,
    summary="General-purpose dashboard chat backed by Gemini",
)
async def chat_general(payload: GeneralChatRequest) -> ChatResponse:
    """
    Simple endpoint for the main dashboard chatbot.
    Accepts a plain user message with optional district context and calls Gemini.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="GOOGLE_API_KEY not set")

    context_md = get_context_md()
    sop_block = f"<context>\n{context_md}\n</context>" if context_md else "<context>No SOP available.</context>"

    district_block = ""
    if payload.district_context:
        district_block = f"\n\nCurrent dashboard context:\n{payload.district_context}"

    system_prompt = f"""You are Sentinel AI, an epidemic intelligence assistant embedded in the
EpiSentinel dengue outbreak prediction dashboard for Karnataka, India.

You help health officers and analysts understand district-level dengue risk predictions,
preventive measures, symptoms, treatment, and model explanations.

Keep answers concise, factual, and action-oriented. Use bullet points where helpful.
Do not invent data — only reference what is in the context or your general medical knowledge.
If asked about something unrelated to dengue/epidemic response, politely decline.

{sop_block}{district_block}"""

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0.3,
            max_output_tokens=1024,
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=payload.message),
        ]
        result = await llm.ainvoke(messages)
        return ChatResponse(response=result.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")
