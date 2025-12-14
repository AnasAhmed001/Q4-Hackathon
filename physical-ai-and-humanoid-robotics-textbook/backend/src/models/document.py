from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from ..services.database_service import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False)  # URL of the document in the book
    title = Column(String, nullable=False)
    module = Column(String, nullable=False)  # Module name (e.g., "module-1-ros2")
    section = Column(String, nullable=True)  # Section within the module
    content = Column(Text, nullable=False)  # Full content of the document
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    metadata_json = Column(Text)  # For storing additional metadata as JSON