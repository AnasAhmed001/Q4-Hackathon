---
id: 1
title: "Create RAG Chatbot"
stage: "spec"
date: "2025-12-11"
surface: "agent"
model: "Claude Sonnet 4.5"
feature: "1-rag-chatbot"
branch: "1-rag-chatbot"
user: "user"
command: "/sp.specify"
labels: ["feature", "specification", "rag", "chatbot"]
links:
  spec: "specs/1-rag-chatbot/spec.md"
  ticket: null
  adr: null
  pr: null
files:
  - "specs/1-rag-chatbot/spec.md"
  - "specs/1-rag-chatbot/checklists/requirements.md"
tests: []
---

# RAG Chatbot Specification

## Overview
Build a RAG chatbot embedded in Docusaurus book (https://q4-hackathon-eosin.vercel.app/) to answer questions about book content and user-selected text.

## Tech Stack
- **Backend**: FastAPI + uv package manager
- **Agent**: OpenAI Agents SDK with Gemini API
- **Embeddings**: Cohere API
- **Vector DB**: Qdrant Cloud (Free Tier)
- **Database**: Neon Serverless Postgres
- **Frontend**: React component in Docusaurus

## Backend (FastAPI)
**Endpoints:**
- `POST /ingest` - Ingest book content
- `POST /chat` - Handle queries with history
- `POST /chat/selection` - Handle selected text queries
- `GET /health` - Health check

**Setup:**
- CORS enabled for Vercel domain
- uv for dependency management

## RAG Pipeline

**Ingestion:**
1. Parse Docusaurus markdown files
2. Chunk by sections (~500-1000 tokens)
3. Generate Cohere embeddings (`embed-english-v3.0`, type: `search_document`)
4. Store in Qdrant with metadata (title, section, URL)
5. Store references in Postgres

**Retrieval:**
1. Generate query embeddings (Cohere, type: `search_query`)
2. Semantic search in Qdrant
3. Two modes:
   - General: Search full knowledge base
   - Selection: Use selected text + relevant chunks
4. Generate response via OpenAI Agents SDK + Gemini
5. Store conversation history in Postgres

## Database Schema (Postgres)
```sql
conversations: id, user_session, created_at
messages: id, conversation_id, role, content, timestamp
documents: id, title, url, content_hash, ingested_at
```

## Qdrant Collection
- Name: `book_knowledge_base`
- Dimensions: 1024
- Distance: Cosine
- Payload: `{text, page_title, url, section}`

## Frontend (React)
- Chat interface with history
- Text selection mode toggle
- Loading/error states
- Citation links to book sections
- Responsive design

## Deployment
- FastAPI: Railway/Render/Vercel
- Qdrant Cloud + Neon Postgres configured
- Environment variables:
  - `COHERE_API_KEY`
  - `GEMINI_API_KEY`
  - `QDRANT_URL`, `QDRANT_API_KEY`
  - `NEON_DATABASE_URL`
  - `ALLOWED_ORIGIN=https://q4-hackathon-eosin.vercel.app`

## Reference
OpenAI Agents SDK: https://openai.github.io/openai-agents-python/

## Success Criteria
- Accurate book content answers
- Working selection-based queries
- Seamless Vercel integration
- Proper citations

---

## Instructions for Implementation
**DO NOT start implementing any features yet.**
**DO NOT create any todo lists or task breakdowns.**

# Response

I've created a comprehensive specification for the RAG chatbot feature. Here's what was completed:

1. Created a new feature branch: `1-rag-chatbot`
2. Created the specification file at `specs/1-rag-chatbot/spec.md` with:
   - User scenarios and testing (3 prioritized user stories)
   - Functional requirements (10 specific requirements)
   - Key entities definition
   - Success criteria with measurable outcomes
3. Created a quality checklist at `specs/1-rag-chatbot/checklists/requirements.md`
4. All specification quality validation items have passed

The specification focuses on the user value and business needs while avoiding implementation details. It defines clear acceptance scenarios and measurable success criteria that are technology-agnostic. The specification is ready for the next phase of planning.