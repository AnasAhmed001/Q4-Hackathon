# Feature Specification: RAG Chatbot for Docusaurus Book

**Feature Branch**: `1-rag-chatbot`
**Created**: 2025-12-11
**Status**: Draft
**Input**: User description: "# RAG Chatbot Specification

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
**DO NOT create any todo lists or task breakdowns.""

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Ask Questions About Book Content (Priority: P1)

As a reader of the Docusaurus book, I want to ask questions about the book content so that I can quickly find relevant information without manually searching through pages.

**Why this priority**: This is the core functionality that delivers immediate value by allowing users to get answers from the book content through natural language queries.

**Independent Test**: Can be fully tested by asking questions about book content and receiving accurate answers with proper citations to relevant sections. Delivers value as a standalone Q&A system for the book.

**Acceptance Scenarios**:

1. **Given** I am viewing the Docusaurus book, **When** I type a question in the chat interface, **Then** I receive an accurate answer based on the book content with citations to relevant sections
2. **Given** I have asked a question, **When** I submit it to the chatbot, **Then** I see a loading state while the system processes my query

---

### User Story 2 - Query Based on Selected Text (Priority: P2)

As a reader who has selected specific text in the book, I want to ask questions about that selected text so that I can get more detailed information or clarifications about specific content.

**Why this priority**: Enhances the core functionality by allowing context-aware queries based on selected text, providing more targeted responses.

**Independent Test**: Can be fully tested by selecting text in the book, switching to selection mode, and asking questions about the selected content. Delivers value as a context-aware assistant.

**Acceptance Scenarios**:

1. **Given** I have selected text in the book, **When** I use the selection-based query feature, **Then** the chatbot considers both the selected text and related content to provide comprehensive answers
2. **Given** I am in selection mode, **When** I ask a question, **Then** the response is focused on the selected text with additional relevant context from the knowledge base

---

### User Story 3 - View Conversation History (Priority: P3)

As a user of the chatbot, I want to see my conversation history so that I can reference previous questions and answers during my research session.

**Why this priority**: Provides continuity and context for longer research sessions, allowing users to build on previous interactions.

**Independent Test**: Can be fully tested by having multiple conversations and viewing the history. Delivers value as a persistent conversation interface.

**Acceptance Scenarios**:

1. **Given** I have had multiple conversations, **When** I view the chat interface, **Then** I can see my previous questions and answers
2. **Given** I am in a new session, **When** I start a new conversation, **Then** my previous conversation data is preserved

---

### Edge Cases

- What happens when the query returns no relevant results from the knowledge base?
- How does the system handle very long text selections?
- What occurs when the vector database is temporarily unavailable?
- How does the system respond to inappropriate or off-topic questions?
- What happens when the book content has been updated after initial ingestion?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST allow users to ask questions about book content through a chat interface
- **FR-002**: System MUST provide accurate answers based on the book content with proper citations
- **FR-003**: System MUST support two query modes: general knowledge base search and selection-based queries
- **FR-004**: System MUST store and retrieve conversation history for user sessions
- **FR-005**: System MUST display loading states during query processing
- **FR-006**: System MUST handle text selection from the book and use it as context for queries
- **FR-007**: System MUST provide error handling for failed queries or unavailable services
- **FR-008**: System MUST ingest book content from Docusaurus markdown files into the knowledge base
- **FR-009**: System MUST preserve text formatting and structure when ingesting content
- **FR-010**: System MUST support responsive design for different device sizes

### Key Entities *(include if feature involves data)*

- **Conversation**: Represents a user's chat session, containing metadata about when it was created and user session identifier
- **Message**: Represents an individual message in a conversation, with role (user/assistant), content, and timestamp
- **Document**: Represents an ingested book section with title, URL, content hash, and ingestion timestamp
- **Knowledge Base Entry**: Represents a chunk of book content with embeddings for semantic search and metadata for citations

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users receive accurate answers to book-related questions 90% of the time based on manual validation
- **SC-002**: Query responses are delivered within 5 seconds for 95% of requests
- **SC-003**: Users can successfully ask questions and receive properly cited answers in both general and selection-based modes
- **SC-004**: 80% of users find the information they're looking for within 3 queries
- **SC-005**: The system successfully ingests all book content without data loss or corruption
- **SC-006**: Users can switch between general and selection-based query modes seamlessly