"""
router.py  —  FastAPI router that wires episentinel_chatbot into HTTP endpoints.
Drop this file into your FastAPI app and include the router.

    # main.py
    from fastapi import FastAPI
    from router import router
    app = FastAPI()
    app.include_router(router)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from episentinel_chatbot import (
    SingleDistrictRequest,
    StateOfficialRequest,
    generate_chat_response,
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
