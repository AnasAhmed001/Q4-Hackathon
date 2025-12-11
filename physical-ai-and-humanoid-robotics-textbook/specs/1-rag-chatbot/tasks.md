---
description: "Task list for RAG Chatbot implementation"
---

# Tasks: RAG Chatbot for Docusaurus Book

**Input**: Design documents from `/specs/1-rag-chatbot/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Web app**: `backend/src/`, `website/src/`
- **Backend**: `backend/src/`, `backend/tests/`
- **Frontend**: `frontend/src/`, `frontend/tests/`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create backend project structure in backend/
- [ ] T002 Initialize uv project with FastAPI dependencies in backend/
- [ ] T003 [P] Install backend dependencies: fastapi, uvicorn, sqlalchemy, psycopg2-binary, qdrant-client, cohere, openai-agents, openai, python-dotenv, pydantic-settings, asyncpg
- [ ] T004 Configure environment variables and settings in backend/src/utils/settings.py
- [ ] T005 Create .env file with required API keys in backend/:
  ```
  COHERE_API_KEY=your_cohere_api_key_here  # Get from https://dashboard.cohere.com/api-keys
  GEMINI_API_KEY=AIzaSyCoq8oWwk5YUF2DDGzLZCVX378hKNH1wQ4
  QDRANT_URL=https://2e225b71-4342-4efe-a3a3-c228b6554f86.us-east4-0.gcp.cloud.qdrant.io:6333
  QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.jMlCSKvckvlLj3A379LAaY2v4MkAbyGdr77fRyhtQ7M
  DATABASE_URL=postgresql://neondb_owner:npg_1BJHKRxts5yM@ep-wandering-recipe-a1bdvjuz-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
  ALLOWED_ORIGIN=https://q4-hackathon-eosin.vercel.app
  BACKEND_API_URL=http://localhost:8000  # or production URL
  ```
- [ ] T006 Add .env file to .gitignore to prevent committing API keys
---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T007 Setup Neon Postgres database connection in backend/src/services/database_service.py
- [ ] T008 [P] Create SQLAlchemy models: Conversation, Message, Document in backend/src/models/
- [ ] T009 [P] Implement CRUD operations for each model in backend/src/services/
- [ ] T010 [P] Setup Qdrant client and create collection "textbook_embeddings" with 1024 dimensions (Cohere embed-english-v3.0) using Cosine distance in backend/src/services/qdrant_service.py with payload indexes for 'module' field
- [ ] T011 [P] Implement Qdrant upsert and search functions in backend/src/services/qdrant_service.py
- [ ] T012 Create FastAPI app with CORS for https://q4-hackathon-eosin.vercel.app in backend/src/api/main.py (allow credentials, methods: GET/POST/DELETE)
- [ ] T013 Create GET /health endpoint in backend/src/api/routers/health_router.py to verify connectivity to Qdrant, Neon, and Gemini API
- [ ] T014 Configure logging in backend/src/utils/logging.py
- [ ] T015 Test health check with all services
- [ ] T015.5 Run initial ingestion of all website/docs/ content to populate Qdrant collection
- [ ] T015.6 Verify embeddings created for all modules (intro, module-1-ros2, module-2-digital-twin, module-3-ai-robot-brain, module-4-vision-language-action, capstone) in Qdrant

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Ask Questions About Book Content (Priority: P1) 🎯 MVP

**Goal**: Allow users to ask questions about the book content and receive accurate answers with citations

**Independent Test**: Can be fully tested by asking questions about book content and receiving accurate answers with proper citations to relevant sections

### Implementation for User Story 1

- [ ] T016 [P] Create embedding service using Cohere API (embed-english-v3.0 model) in backend/src/services/embedding_service.py with input_type="search_document" for docs and "search_query" for queries
- [ ] T016.5 Implement batch embedding processing in embedding_service.py (process 96 texts per batch for Cohere API efficiency)
- [ ] T017 [P] Create comprehensive RAG service in backend/src/services/rag_service.py with general and selection-based retrieval logic (used by agent_tools.py)
- [ ] T018 [P] Create advanced markdown parser for Docusaurus files in backend/src/utils/markdown_parser.py with metadata extraction
- [ ] T019 [P] Create intelligent text chunker in backend/src/utils/text_chunker.py (500-1000 tokens, header splits, 50-100 token overlap, code block preservation)
- [ ] T020 Create POST /chat endpoint in backend/src/api/routers/chat_router.py with conversation history, semantic search, and response generation using agent
- [ ] T020.5 Implement streaming response support for agent in /chat endpoint to show incremental responses to user
- [ ] T021 [P] Implement comprehensive ingestion service in backend/src/services/ingestion_service.py with parse→chunk→embed→store workflow
- [ ] T022 Create POST /ingest endpoint in backend/src/api/routers/ingest_router.py with duplicate detection and progress tracking
- [ ] T023 [P] **Use Context7 MCP Server** for latest OpenAI Agents SDK documentation. Create OpenAI Agent service in backend/src/services/agent_service.py using OpenAI Agents SDK with Gemini:
  ```python
  from agents import Agent, OpenAIChatCompletionsModel, RunConfig
  from openai import AsyncOpenAI
  
  # Configure custom client for Gemini
  external_client = AsyncOpenAI(
      api_key=GEMINI_API_KEY,
      base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
  )
  
  # Define model
  model = OpenAIChatCompletionsModel(
      model="gemini-2.0-flash",
      openai_client=external_client
  )
  
  # Create agent with tools (defined in agent_tools.py)
  agent = Agent(
      name="Textbook Assistant",
      instructions="You are a helpful teaching assistant for the Physical AI and Humanoid Robotics textbook. Use search_textbook_content tool to find relevant information. Always cite sources by mentioning the module and section. Provide clear explanations suitable for students. Maintain Flesch-Kincaid grade level 10-12.",
      model=model,
      tools=[search_textbook_content, search_selected_text]
  )
  
  # Run config
  config = RunConfig(
      model=model,
      model_provider=external_client,
      tracing_disable=True
  )
  ```
- [ ] T024 [P] Create function tools in backend/src/services/agent_tools.py:
  ```python
  from agents import function_tool
  import cohere
  from qdrant_client import QdrantClient
  
  # Initialize clients
  co = cohere.Client(api_key=COHERE_API_KEY)
  qdrant_client = QdrantClient(
      url=QDRANT_URL,
      api_key=QDRANT_API_KEY
  )
  
  @function_tool
  def search_textbook_content(query: str, top_k: int = 5) -> list[dict]:
      """Search the textbook content for relevant information.
      
      Args:
          query: The search query
          top_k: Number of top results to return (default 5)
      
      Returns:
          List of relevant content chunks with metadata
      """
      # Generate query embedding using Cohere
      response = co.embed(
          texts=[query],
          model="embed-english-v3.0",
          input_type="search_query"
      )
      query_embedding = response.embeddings[0]
      
      # Search in Qdrant
      search_results = qdrant_client.search(
          collection_name="textbook_embeddings",
          query_vector=query_embedding,
          limit=top_k,
          score_threshold=0.7
      )
      
      # Format results
      return [
          {
              "content": hit.payload["content"],
              "module": hit.payload["module"],
              "section": hit.payload["section"],
              "url": hit.payload["url"],
              "score": hit.score
          }
          for hit in search_results
      ]
  
  @function_tool
  def search_selected_text(query: str, selected_text: str, top_k: int = 3) -> list[dict]:
      """Search for information related to user-selected text.
      
      Args:
          query: The user's question
          selected_text: The text highlighted by the user
          top_k: Number of supplementary results to return
      
      Returns:
          Primary context (selected text) plus supplementary relevant chunks
      """
      # Use selected text as primary context
      # Optionally retrieve additional relevant chunks
      combined_query = f"{query} {selected_text}"
      return search_textbook_content(combined_query, top_k)
  ```
- [ ] T025 Add rate limiting (10 req/hour anonymous, 50 req/hour authenticated) and enhanced error handling for chat endpoints
- [ ] T025.5 Implement exponential backoff retry logic for Cohere and Gemini API rate limits in backend/src/utils/retry_handler.py
- [ ] T026 Create POST /chat/selection endpoint in backend/src/api/routers/chat_router.py with selected text processing using search_selected_text tool
- [ ] T026.5 Implement session ID generation and management in backend/src/services/session_service.py (UUID-based, stored in cookies/localStorage)
- [ ] T027 Implement conversation management in backend/src/services/conversation_service.py with session creation and history retrieval from Neon
- [ ] T028 [P] Create ChatWidget React component in website/src/components/ChatWidget.tsx with message history and UI state management
- [ ] T029 [P] Create ChatMessage React component in website/src/components/ChatMessage.tsx with citation rendering and responsive styling
- [ ] T030 [P] Create ChatInput React component in website/src/components/ChatInput.tsx with loading states and error handling
- [ ] T031 [P] Create SelectionMode React component in website/src/components/SelectionMode.tsx with text selection detectionection
- [ ] T032 Create API client service in website/src/services/apiClient.js with backend base URL from env (BACKEND_API_URL) and cross-origin request handling
- [ ] T033 Implement message sending/receiving in website/src/services/chatService.js
- [ ] T034 Implement conversation state management in website/src/services/chatService.js
- [ ] T035 Add selection-based query support in website/src/services/chatService.js
- [ ] T036 Style chat components for mobile responsiveness in website/src/styles/
- [ ] T037 Implement text selection mode toggle in website/src/components/ChatWidget.jsx
- [ ] T038 Integrate frontend with backend API in website/src/services/chatService.js

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Query Based on Selected Text (Priority: P2)

**Goal**: Allow users to ask questions about selected text in the book with additional context from knowledge base

**Independent Test**: Can be fully tested by selecting text in the book, switching to selection mode, and asking questions about the selected content

### Implementation for User Story 2

- [ ] T039 Update RAG service to handle selection-based queries in backend/src/services/rag_service.py with selected text + search related chunks
- [ ] T040 Test selection-based retrieval quality and response accuracy
- [ ] T041 Test cross-origin requests for selection-based queries in website/tests/

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - View Conversation History (Priority: P3)

**Goal**: Provide users with access to their conversation history during research sessions

**Independent Test**: Can be fully tested by having multiple conversations and viewing the history

### Implementation for User Story 3

- [ ] T042 Update chat endpoints to maintain conversation context in backend/src/api/routers/chat_router.py
- [ ] T043 Enhance Conversation model to support session management in backend/src/models/conversation.py
- [ ] T044 Add conversation history retrieval in backend/src/services/database_service.py
- [ ] T045 Update frontend to display conversation history in website/src/components/ChatWidget.jsx
- [ ] T046 Add session management in website/src/services/chatService.js
- [ ] T047 Test multi-turn conversations and message storage with timestamps

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T048 [P] Documentation updates in docs/
- [ ] T049 Code cleanup and refactoring
- [ ] T050 Performance optimization across all stories
- [ ] T051 [P] Additional unit tests in backend/tests/ and website/tests/
- [ ] T051.5 Run contract tests to verify API endpoints match contracts/chat-api.yaml schema
- [ ] T051.6 Test OpenAI Agents SDK tool calling with sample queries (verify agent selects correct tools and formats responses properly)
- [ ] T052 Security hardening (input validation, API key rotation, HTTPS enforcement)
- [ ] T052.5 Implement rate limiting middleware in backend/src/api/middleware.py (10 req/hour anonymous, 50 req/hour authenticated)
- [ ] T052.6 Add comprehensive logging for API usage, errors, and Gemini API costs in backend/src/utils/logging.py
- [ ] T052.7 Set up basic monitoring for API health, response times, and usage metrics (Railway metrics or custom dashboard)
- [ ] T053 Run quickstart.md validation
- [ ] T054 Responsive design improvements in website/src/styles/
- [ ] T055 Accessibility improvements for WCAG 2.1 AA compliance (keyboard navigation, ARIA labels, screen reader support)
- [ ] T056 [P] Create Docusaurus theme wrapper in website/src/theme/Root.js to inject ChatWidget globally on all pages
- [ ] T056.5 Add ChatWidget provider to Docusaurus layout in website/docusaurus.config.js customFields and themeConfig
- [ ] T056.6 Test chat widget appears and functions correctly on all /docs/** pages (intro, modules, capstone)
- [ ] T057 Ensure no conflicts with Docusaurus features in website/docusaurus.config.js
- [ ] T058 Test chat widget on multiple pages and screen sizes
- [ ] T059 [P] Deploy FastAPI backend to Railway (recommended for Python/FastAPI) with production environment variables and health check endpoint
- [ ] T060 Configure production environment variables for deployed backend (use Railway secrets for API keys)
- [ ] T061 [P] Configure production environment to use existing Neon Postgres (ep-wandering-recipe-a1bdvjuz) and Qdrant Cloud cluster (2e225b71-4342-4efe-a3a3-c228b6554f86) - verify connectivity and test queries
- [ ] T062 Verify CORS configuration for Vercel domain (https://q4-hackathon-eosin.vercel.app) in backend/src/api/main.py
- [ ] T063 Deploy Docusaurus changes to Vercel
- [ ] T064 Run end-to-end tests on live site:
- [ ] T064.5 Test complete user flow: open chat → ask question → receive answer with citations → click citation link → verify navigation
- [ ] T064.6 Test selection mode: highlight text → click "Ask about this" → ask question → verify selected context used in response
- [ ] T064.7 Test conversation history: multi-turn conversation with follow-up questions → verify context maintained across turns
- [ ] T065 Write README with setup instructions in README.md
- [ ] T066 Document API endpoints in docs/api-reference.md
- [ ] T067 Add deployment guide in docs/deployment.md
- [ ] T068 Create troubleshooting section in docs/troubleshooting.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all models for User Story 1 together:
Task: "Create embedding service in backend/src/services/embedding_service.py"
Task: "Create RAG service in backend/src/services/rag_service.py"
Task: "Create markdown parser in backend/src/utils/markdown_parser.py"
Task: "Create text chunker in backend/src/utils/text_chunker.py"

# Launch all frontend components for User Story 1 together:
Task: "Create frontend ChatWidget component in website/src/components/ChatWidget.jsx"
Task: "Create frontend MessageList component in website/src/components/MessageList.jsx"
Task: "Create frontend MessageInput component in website/src/components/MessageInput.jsx"
Task: "Create frontend CitationRenderer component in website/src/components/CitationRenderer.jsx"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence