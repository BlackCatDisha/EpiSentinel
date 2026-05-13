from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
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

# Allow requests from any origin (dashboard on rawgithack, file://, localhost:8080, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(predict_router)

# Serve the dashboard as static files at /dashboard and also at /
DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")

    # Redirect so relative assets (style.css, app.js) resolve under /dashboard/
    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/dashboard/", status_code=302)
