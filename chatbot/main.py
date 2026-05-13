from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
from router import router as chat_router
from predict_router import router as predict_router
from episentinel_chatbot import get_context_md

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_context_md()   # warm the cache at startup, not on first request
    yield

app = FastAPI(title="EpiSentinel Backend", lifespan=lifespan)
app.include_router(chat_router)
app.include_router(predict_router)