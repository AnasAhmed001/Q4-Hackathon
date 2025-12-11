---
id: 5
title: "Update RAG Agent Tasks"
stage: "tasks"
date: "2025-12-11"
surface: "agent"
model: "Claude Sonnet 4.5"
feature: "1-rag-chatbot"
branch: "1-rag-chatbot"
user: "user"
command: "/sp.tasks"
labels: ["feature", "tasks", "rag", "agent", "chat"]
links:
  spec: "specs/1-rag-chatbot/spec.md"
  plan: "specs/1-rag-chatbot/plan.md"
  tasks: "specs/1-rag-chatbot/tasks.md"
  ticket: null
  adr: null
  pr: null
files:
  - "specs/1-rag-chatbot/tasks.md"
tests: []
---

# Task 3: RAG Agent & Chat Endpoints

## Objective
Implement RAG retrieval logic, OpenAI Agents SDK integration, and chat endpoints.

## Deliverables
1. **OpenAI Agents SDK Setup**
   - Configure agent with Gemini API in `agent.py`
   - Create system prompt for book Q&A
   - Implement retrieval tool/function
   - Test basic agent responses

2. **Retrieval Logic**
   - General query: embed query → search Qdrant → rank results
   - Selection-based: use selected text + search related chunks
   - Format context with citations
   - Test retrieval quality

3. **Chat Endpoints**
   - `POST /chat`:
     - Accept query + conversation_id
     - Retrieve history from Postgres
     - Perform semantic search
     - Generate response via agent
     - Store message and return with citations
   - `POST /chat/selection`:
     - Accept selected text + query
     - Implement selection-based retrieval
     - Generate response
   - Add rate limiting and error handling

4. **Conversation Management**
   - Session creation and history retrieval
   - Message storage with timestamps
   - Test multi-turn conversations

**Success Criteria:** Both chat endpoints working with accurate responses and citations

# Response

I've updated the existing tasks.md file to incorporate the detailed requirements for the RAG agent and chat endpoints. The updates include:

1. Enhanced RAG service (T017): Now includes general and selection-based retrieval logic
2. New OpenAI Agent service (T023): Creates agent service with Gemini API configuration and system prompt for book Q&A
3. New retrieval tool implementation (T024): Implements retrieval tool for the agent
4. Enhanced chat endpoint (T020): Now includes conversation history, semantic search, and response generation
5. New selection chat endpoint (T026): Creates POST /chat/selection endpoint with selected text processing
6. New conversation management service (T027): Implements session creation and history retrieval
7. Enhanced RAG service for User Story 2 (T035): Updated to handle selection-based queries with selected text + search related chunks
8. Added selection-based testing (T040): Tests selection-based retrieval quality and response accuracy
9. Added multi-turn conversation testing (T046): Tests multi-turn conversations and message storage with timestamps

The task numbers were adjusted to accommodate the new requirements while maintaining the existing structure. All updates follow the required format with proper task IDs, parallel markers, and file paths. The tasks are still organized by user stories to enable independent implementation and testing.