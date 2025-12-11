# Implementation Plan: RAG Chatbot for Docusaurus Book

**Branch**: `1-rag-chatbot` | **Date**: 2025-12-11 | **Spec**: specs/1-rag-chatbot/spec.md
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of a RAG (Retrieval Augmented Generation) chatbot that integrates with the Docusaurus book to answer questions about book content and selected text. The system will use FastAPI backend with Qdrant vector database for semantic search, Cohere API for embeddings, and OpenAI Agents SDK with Gemini for response generation. The frontend will be a React component embedded in the Docusaurus site.

## Technical Context

**Language/Version**: Python 3.13, JavaScript/TypeScript for React
**Primary Dependencies**: FastAPI, Cohere API, Qdrant, OpenAI Agents SDK, Gemini API, Neon Postgres, React
**Storage**: Neon Serverless Postgres, Qdrant Cloud vector database
**Testing**: pytest for backend, Jest for frontend
**Target Platform**: Web application (Docusaurus integration)
**Project Type**: Web (backend API + frontend component)
**Performance Goals**: Query responses delivered within 5 seconds for 95% of requests
**Constraints**: <200ms p95 for API responses, must handle concurrent users, must maintain WCAG 2.1 AA compliance for accessibility
**Scale/Scope**: Supports multiple concurrent users, handles book content of moderate size (up to 1000 pages worth of content)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Pre-design assessment:**
- **Pedagogical clarity**: The RAG chatbot will enhance learning by providing immediate answers to student questions about book content, supporting the textbook's goal of accessibility for students from different backgrounds
- **Technical accuracy**: All technical implementations will be verified against authoritative sources (API documentation, best practices)
- **Hands-on learning**: The chatbot provides an interactive way for students to engage with textbook content
- **Progressive complexity**: The feature supports the textbook's learning pathways by allowing students to ask follow-up questions
- **Open educational resource**: The implementation will be open source and reusable
- **Accessibility**: The React chat component will maintain WCAG 2.1 AA compliance
- **Content Review Process**: The implementation will undergo technical review before deployment

**Post-design assessment:**
- **Pedagogical clarity**: Confirmed - the API design allows for rich responses with citations that enhance learning
- **Technical accuracy**: Confirmed - using established APIs (Cohere, Gemini) with proper error handling
- **Hands-on learning**: Confirmed - the chat interface promotes active engagement with content
- **Progressive complexity**: Confirmed - conversation history allows for follow-up questions building on previous context
- **Open educational resource**: Confirmed - all components are open source technologies
- **Accessibility**: Confirmed - API responses include structured data for proper citation rendering in accessible UI
- **Content Review Process**: Confirmed - API contracts defined for proper testing and validation

## Project Structure

### Documentation (this feature)

```text
specs/1-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
│   └── chat-api.yaml    # OpenAPI specification
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   ├── conversation.py
│   │   ├── message.py
│   │   └── document.py
│   ├── services/
│   │   ├── ingestion_service.py
│   │   ├── rag_service.py
│   │   ├── embedding_service.py
│   │   ├── qdrant_service.py
│   │   └── database_service.py
│   ├── api/
│   │   ├── main.py
│   │   ├── routers/
│   │   │   ├── chat_router.py
│   │   │   ├── ingest_router.py
│   │   │   └── health_router.py
│   │   └── dependencies.py
│   └── utils/
│       ├── markdown_parser.py
│       ├── text_chunker.py
│       └── settings.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── contract/
└── requirements.txt

frontend/
├── src/
│   ├── components/
│   │   ├── ChatWidget.jsx
│   │   ├── MessageList.jsx
│   │   ├── MessageInput.jsx
│   │   └── CitationRenderer.jsx
│   ├── services/
│   │   ├── apiClient.js
│   │   └── chatService.js
│   └── styles/
│       └── chat.css
└── package.json
```

**Structure Decision**: Web application structure selected with separate backend (FastAPI) and frontend (React) components to maintain clear separation of concerns. The backend handles RAG processing and data management, while the frontend provides the user interface embedded in the Docusaurus site.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |