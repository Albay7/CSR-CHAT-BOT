from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_engine import CambridgeRAGEngine

# ── Startup: initialise RAG engine once ───────────────────────────────────────
rag_engine: Optional[CambridgeRAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    rag_engine = CambridgeRAGEngine()
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cambridge Support RAG Bot API",
    description="Retrieval-Augmented Generation API for Cambridge student support.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response schemas ─────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    top_k: int = 3


class SourceItem(BaseModel):
    title: str
    category: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    mode: str


# ── API Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Health check endpoint used by Azure App Service and frontend."""
    return {"status": "ok", "engine_ready": rag_engine is not None}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Main RAG chat endpoint (synchronous def runs in threadpool to prevent blocking the event loop)."""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine is initializing. Please try again in a few moments.")

    history = [{"role": m.role, "content": m.content} for m in request.history]

    result = rag_engine.generate_response(
        query=request.message,
        chat_history=history,
        top_k=request.top_k,
    )

    sources = [
        SourceItem(
            title=s.get("title", ""),
            category=s.get("category", ""),
            url=s.get("url", "#"),
        )
        for s in result.get("sources", [])
    ]

    return ChatResponse(
        answer=result.get("answer", ""),
        sources=sources,
        mode=result.get("mode", "Unknown"),
    )


# ── Static frontend ────────────────────────────────────────────────────────────
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def serve_frontend():
    """Serve the main chat UI."""
    index_file = static_dir / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(str(index_file))
