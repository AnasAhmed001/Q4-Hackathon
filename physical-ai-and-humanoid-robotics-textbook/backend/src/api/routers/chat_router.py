from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
from ...services.rag_service import rag_service
from ...services.conversation_service import conversation_service
from ...services.agent_service import agent_service
from ...utils.logging import log_api_call, app_logger


router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str
    selected_text: Optional[str] = None  # For selection-based queries


@router.post("/chat")
async def streaming_chat_endpoint(chat_request: ChatRequest):
    """
    Streaming chat endpoint that returns token-by-token responses for better UX
    """
    async def event_generator():
        try:
            # Log the API call
            log_api_call(
                endpoint="/api/v1/chat",
                method="POST",
                session_id=chat_request.session_id
            )

            # Get conversation history if it exists
            conversation = conversation_service.get_or_create_conversation(chat_request.session_id)

            # If selected_text is provided, use selection-based retrieval
            if chat_request.selected_text:
                context_chunks = rag_service.retrieve_with_selection_context(
                    chat_request.message,
                    chat_request.selected_text
                )
            else:
                # General retrieval
                context_chunks = rag_service.retrieve_relevant_content(chat_request.message)

            # Prepare citations
            citations = [
                {
                    "module": chunk["module"],
                    "section": chunk["section"],
                    "url": chunk["url"],
                    "score": chunk["score"]
                }
                for chunk in context_chunks
            ]

            # Generate response using the agent with streaming
            full_response = ""
            async for token in agent_service.generate_response_stream(
                query=chat_request.message,
                context_chunks=context_chunks,
                conversation_history=conversation_service.get_conversation_history(conversation.id) if conversation_service.get_conversation_history(conversation.id) else []
            ):
                if token:  # Only yield non-empty tokens
                    full_response += token
                    yield f"data: {json.dumps({'token': token, 'done': False, 'citations': []})}\n\n"

            # Save the user message and assistant response to the conversation
            conversation_service.add_message(conversation.id, "user", chat_request.message)
            conversation_service.add_message(conversation.id, "assistant", full_response)

            # Send final message with citations
            yield f"data: {json.dumps({'token': '', 'done': True, 'citations': citations})}\n\n"

        except Exception as e:
            app_logger.error(f"Error in streaming chat endpoint: {str(e)}")
            yield f"data: {json.dumps({'token': '', 'done': True, 'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")