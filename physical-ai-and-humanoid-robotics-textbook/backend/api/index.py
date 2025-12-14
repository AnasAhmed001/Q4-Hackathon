# Vercel entry point
import sys
from pathlib import Path

# Add the src directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from src.api.main import app

# Export the FastAPI app for Vercel
handler = app
