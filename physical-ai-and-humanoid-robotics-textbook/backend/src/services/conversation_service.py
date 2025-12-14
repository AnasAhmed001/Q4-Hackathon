from typing import List, Optional
from sqlalchemy.orm import Session
from uuid import uuid4
from ..models.conversation import Conversation
from ..models.message import Message
from ..services.database_service import get_db, SessionLocal, db_ops, ConversationCreate, MessageCreate, generate_session_id


class ConversationService:
    def __init__(self):
        pass

    def get_or_create_conversation(self, session_id: str) -> Conversation:
        """
        Get existing conversation by session_id or create a new one
        """
        db: Session = SessionLocal()
        try:
            # Try to get existing conversation
            existing_conversation = db_ops.get_conversation_by_session_id(db, session_id=session_id)
            if existing_conversation:
                return existing_conversation

            # Create new conversation if it doesn't exist
            if not session_id or session_id == "new":
                session_id = generate_session_id()

            conversation_in = ConversationCreate(
                session_id=session_id,
                title=f"Conversation {session_id[:8]}"  # Use first 8 chars as title
            )

            new_conversation = db_ops.create_conversation(db, obj_in=conversation_in)
            return new_conversation
        finally:
            db.close()

    def add_message(self, conversation_id: int, role: str, content: str, is_selected_text_query: bool = False):
        """
        Add a message to a conversation
        """
        db: Session = SessionLocal()
        try:
            message_in = MessageCreate(
                conversation_id=conversation_id,
                role=role,
                content=content,
                is_selected_text_query=is_selected_text_query
            )

            # Create the message
            db_ops.create_message(db, obj_in=message_in)
        finally:
            db.close()

    def get_conversation_history(self, conversation_id: int) -> List[Message]:
        """
        Get all messages for a specific conversation
        """
        db: Session = SessionLocal()
        try:
            return db_ops.get_messages_by_conversation(db, conversation_id=conversation_id)
        finally:
            db.close()

    def get_conversation_by_session_id(self, session_id: str) -> Optional[Conversation]:
        """
        Get conversation by session ID
        """
        db: Session = SessionLocal()
        try:
            return db_ops.get_conversation_by_session_id(db, session_id=session_id)
        finally:
            db.close()

    def create_new_conversation(self, session_id: Optional[str] = None) -> Conversation:
        """
        Create a new conversation with an optional session ID
        """
        if not session_id:
            session_id = generate_session_id()

        db: Session = SessionLocal()
        try:
            conversation_in = ConversationCreate(
                session_id=session_id,
                title=f"New Conversation {session_id[:8]}"
            )

            new_conversation = db_ops.create_conversation(db, obj_in=conversation_in)
            return new_conversation
        finally:
            db.close()


# Global instance
conversation_service = ConversationService()
