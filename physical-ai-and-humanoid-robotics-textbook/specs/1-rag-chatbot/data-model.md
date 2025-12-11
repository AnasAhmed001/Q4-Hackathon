# Data Model: RAG Chatbot

## Entity: Conversation
- **Description**: Represents a user's chat session with metadata
- **Fields**:
  - id: UUID (Primary Key)
  - user_session: String (Session identifier)
  - created_at: DateTime (Timestamp of creation)
- **Validation**:
  - id must be unique
  - created_at must be in the past or present
- **Relationships**: Contains multiple Messages

## Entity: Message
- **Description**: Represents an individual message in a conversation
- **Fields**:
  - id: UUID (Primary Key)
  - conversation_id: UUID (Foreign Key to Conversation)
  - role: String (Either "user" or "assistant")
  - content: Text (The message content)
  - timestamp: DateTime (When the message was created)
- **Validation**:
  - role must be either "user" or "assistant"
  - conversation_id must reference an existing Conversation
  - content must not be empty
- **Relationships**: Belongs to a Conversation

## Entity: Document
- **Description**: Represents an ingested book section
- **Fields**:
  - id: UUID (Primary Key)
  - title: String (Title of the document/section)
  - url: String (URL to the document in the book)
  - content_hash: String (Hash of the content for deduplication)
  - ingested_at: DateTime (When the document was ingested)
- **Validation**:
  - url must be a valid URL format
  - content_hash must be unique to prevent duplicates
- **Relationships**: Used for reference during RAG retrieval

## Entity: Knowledge Base Entry
- **Description**: Represents a chunk of book content with embeddings for semantic search
- **Fields**:
  - id: UUID (Primary Key for Qdrant)
  - text: Text (The content chunk)
  - page_title: String (Title of the page/section)
  - url: String (URL to the source)
  - section: String (Section identifier)
  - embedding: Array<float> (Cohere embedding vector, 1024 dimensions)
- **Validation**:
  - embedding must have exactly 1024 dimensions
  - text must not be empty
- **Relationships**: Used for semantic search in Qdrant