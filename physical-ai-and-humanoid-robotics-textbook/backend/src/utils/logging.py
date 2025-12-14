import logging
import sys
from datetime import datetime
from typing import Any, Dict


# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")
    ]
)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


# Create a global logger for the application
app_logger = get_logger("rag_chatbot")


def log_api_call(
    endpoint: str,
    method: str,
    user_id: str = None,
    session_id: str = None,
    response_time: float = None,
    status_code: int = None
):
    """
    Log API calls with relevant information
    """
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "endpoint": endpoint,
        "method": method,
        "user_id": user_id,
        "session_id": session_id,
        "response_time_ms": response_time,
        "status_code": status_code
    }

    app_logger.info(f"API_CALL: {log_data}")


def log_gemini_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float
):
    """
    Log Gemini API usage and costs
    """
    cost_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": cost_usd
    }

    app_logger.info(f"GEMINI_COST: {cost_data}")


def log_error(error: Exception, context: str = ""):
    """
    Log errors with context
    """
    app_logger.error(
        f"ERROR in {context}: {str(error)} - Type: {type(error).__name__}",
        exc_info=True
    )