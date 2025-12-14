from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import chat_router, health_router, ingest_router
from .middleware import RateLimitMiddleware
from ..utils.settings import settings
from ..services.qdrant_service import qdrant_service


app = FastAPI(
    title="RAG Chatbot API",
    description="API for the RAG Chatbot that answers questions about the Physical AI and Humanoid Robotics textbook",
    version="1.0.0"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.allowed_origin,  # https://q4-hackathon-eosin.vercel.app
        "http://localhost:3000",  # Local development
        "http://localhost:3001",  # Local development (alternate port)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],  # Allow specified methods
    allow_headers=["*"],  # Allow all headers
)


@app.on_event("startup")
async def startup_event():
    """
    Initialize Qdrant collection on startup
    """
    qdrant_service.create_collection()


# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Include routers
app.include_router(chat_router.router, prefix="/api/v1", tags=["chat"])
app.include_router(health_router.router, prefix="/api/v1", tags=["health"])
app.include_router(ingest_router.router, prefix="/api/v1", tags=["ingest"])


@app.get("/")
def read_root():
    return {"message": "RAG Chatbot API is running!"}