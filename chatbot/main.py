from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
from router import router
from episentinel_chatbot import get_context_md

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_context_md()   # warm the cache at startup, not on first request
    yield

app = FastAPI(title="EpiSentinel Chatbot", lifespan=lifespan)
app.include_router(router)