# Quickstart Guide: RAG Chatbot

## Prerequisites

- Python 3.11+
- Node.js 18+
- uv package manager
- Access to Cohere API
- Access to Gemini API
- Qdrant Cloud account
- Neon Postgres account

## Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Set up backend environment:
```bash
cd backend
uv venv  # Create virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

3. Set up frontend environment:
```bash
cd frontend
npm install
```

4. Create environment files:
```bash
# backend/.env
COHERE_API_KEY=your_cohere_api_key
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
NEON_DATABASE_URL=your_neon_database_url
ALLOWED_ORIGIN=https://q4-hackathon-eosin.vercel.app
```

## Running the Application

1. Start the backend:
```bash
cd backend
uv run src/api/main.py  # or use your preferred ASGI server
```

2. Start the frontend (for development):
```bash
cd frontend
npm start
```

## Initial Setup

1. Ingest the book content:
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_url": "https://q4-hackathon-eosin.vercel.app"}'
```

2. Test the health endpoint:
```bash
curl http://localhost:8000/health
```

## API Usage

### General Chat Query:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the key concepts in humanoid robotics?", "user_session": "session-123"}'
```

### Selection-Based Query:
```bash
curl -X POST http://localhost:8000/chat/selection \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Can you explain this further?",
    "selected_text": "Humanoid robots are robots with human-like characteristics...",
    "user_session": "session-123"
  }'
```