from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ...services.qdrant_service import qdrant_service
from ...services.database_service import engine
from ...utils.settings import settings
import httpx
import asyncio


router = APIRouter()


class HealthStatus(BaseModel):
    status: str
    details: dict


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Health check endpoint that verifies connectivity to Qdrant, Neon, and Gemini API
    """
    details = {
        "qdrant": {"status": "unknown", "message": ""},
        "database": {"status": "unknown", "message": ""},
        "gemini": {"status": "unknown", "message": ""}
    }

    # Check Qdrant connection
    try:
        # Try to get collection info to verify connection
        qdrant_service.client.get_collection(qdrant_service.collection_name)
        details["qdrant"]["status"] = "healthy"
        details["qdrant"]["message"] = "Connected to Qdrant successfully"
    except Exception as e:
        details["qdrant"]["status"] = "unhealthy"
        details["qdrant"]["message"] = f"Qdrant connection failed: {str(e)}"

    # Check database connection
    try:
        # Try to connect to the database
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        details["database"]["status"] = "healthy"
        details["database"]["message"] = "Connected to database successfully"
    except Exception as e:
        details["database"]["status"] = "unhealthy"
        details["database"]["message"] = f"Database connection failed: {str(e)}"

    # Check Gemini API
    try:
        # Make a simple request to verify API key and connectivity
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={settings.gemini_api_key}",
                timeout=10.0
            )
            if response.status_code == 200:
                details["gemini"]["status"] = "healthy"
                details["gemini"]["message"] = "Connected to Gemini API successfully"
            else:
                details["gemini"]["status"] = "unhealthy"
                details["gemini"]["message"] = f"Gemini API returned status code: {response.status_code}"
    except Exception as e:
        details["gemini"]["status"] = "unhealthy"
        details["gemini"]["message"] = f"Gemini API connection failed: {str(e)}"

    # Determine overall status
    overall_status = "healthy" if all(detail["status"] == "healthy" for detail in details.values()) else "unhealthy"

    return HealthStatus(status=overall_status, details=details)