from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional, List
from pydantic import BaseModel
from uuid import uuid4
from ..utils.settings import settings

# Create database engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=300,    # Recycle connections every 5 minutes
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """
    Dependency function to get database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# Pydantic Schemas
# ============================================================================

class DocumentCreate(BaseModel):
    url: str
    title: str
    module: str
    section: Optional[str] = None
    content: Optional[str] = None
    metadata_json: Optional[str] = None


class ConversationCreate(BaseModel):
    session_id: str
    title: Optional[str] = None
    metadata_json: Optional[str] = None


class MessageCreate(BaseModel):
    conversation_id: int
    role: str
    content: str
    is_selected_text_query: bool = False
    metadata_json: Optional[str] = None


# ============================================================================
# Database Operations
# ============================================================================

class DatabaseOperations:
    """Consolidated database operations for all models"""
    
    # Document operations
    def get_document_by_url(self, db: Session, url: str):
        from ..models.document import Document
        return db.query(Document).filter(Document.url == url).first()
    
    def get_documents_by_module(self, db: Session, module: str):
        from ..models.document import Document
        return db.query(Document).filter(Document.module == module).all()
    
    def create_document(self, db: Session, obj_in: DocumentCreate):
        from ..models.document import Document
        db_obj = Document(**obj_in.dict())
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    # Conversation operations
    def get_conversation_by_session_id(self, db: Session, session_id: str):
        from ..models.conversation import Conversation
        return db.query(Conversation).filter(Conversation.session_id == session_id).first()
    
    def create_conversation(self, db: Session, obj_in: ConversationCreate):
        from ..models.conversation import Conversation
        db_obj = Conversation(
            session_id=obj_in.session_id,
            title=obj_in.title,
            metadata_json=obj_in.metadata_json
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    # Message operations
    def get_messages_by_conversation(self, db: Session, conversation_id: int):
        from ..models.message import Message
        return db.query(Message).filter(Message.conversation_id == conversation_id).all()
    
    def create_message(self, db: Session, obj_in: MessageCreate):
        from ..models.message import Message
        db_obj = Message(
            conversation_id=obj_in.conversation_id,
            role=obj_in.role,
            content=obj_in.content,
            is_selected_text_query=obj_in.is_selected_text_query,
            metadata_json=obj_in.metadata_json
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj


# Global instance
db_ops = DatabaseOperations()


# ============================================================================
# Utility Functions
# ============================================================================

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid4())