from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from ..services.database_service import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=True)  # Optional title based on first message
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    metadata_json = Column(Text)  # For storing additional metadata as JSON