---
id: 4
title: "Update Ingestion Tasks"
stage: "tasks"
date: "2025-12-11"
surface: "agent"
model: "Claude Sonnet 4.5"
feature: "1-rag-chatbot"
branch: "1-rag-chatbot"
user: "user"
command: "/sp.tasks"
labels: ["feature", "tasks", "ingestion", "pipeline"]
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

# Task 2: Document Ingestion Pipeline

## Objective
Build complete ingestion pipeline: parse Docusaurus content → chunk → embed → store.

## Deliverables
1. **Cohere Embedding Service**
   - Create `embeddings.py` service
   - Implement document embeddings (type: `search_document`)
   - Implement query embeddings (type: `search_query`)
   - Add batch processing and error handling
   - Test with sample text

2. **Parsing & Chunking**
   - Create markdown parser for Docusaurus files
   - Extract metadata (title, sections, URLs)
   - Implement intelligent chunking (~500-1000 tokens):
     - Split by markdown headers
     - Add 50-100 token overlap
     - Preserve code blocks
   - Test with sample content

3. **Ingestion Endpoint**
   - Create `POST /ingest` endpoint
   - Implement workflow: parse → chunk → embed → store in Qdrant → save metadata to Postgres
   - Add duplicate detection (content hashing)
   - Add batch processing and progress tracking
   - Test with full book content

**Success Criteria:** Book content successfully ingested into Qdrant and Postgres

# Response

I've updated the existing tasks.md file to incorporate the detailed requirements for the document ingestion pipeline. The updates include:

1. Enhanced task T016: Now specifically creates a Cohere embedding service with both document and query embedding functions
2. Enhanced task T018: Now creates an advanced markdown parser specifically for Docusaurus files with metadata extraction capabilities
3. Enhanced task T019: Now creates an intelligent text chunker with specific requirements (500-1000 tokens, header splits, 50-100 token overlap, code block preservation)
4. Enhanced task T021: Now implements a comprehensive ingestion service with the complete parse→chunk→embed→store workflow
5. Enhanced task T022: Now creates a POST /ingest endpoint with duplicate detection and progress tracking

These updates ensure that the document ingestion pipeline requirements are fully captured in the task list while maintaining the existing structure and dependencies. The tasks are still organized by user story to enable independent implementation and testing.