from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from typing import List, Dict, Any, AsyncGenerator
import cohere
from qdrant_client import QdrantClient
from ..utils.settings import settings


# ============================================================================
# Function Tools for Agent
# ============================================================================

# Initialize clients for tools
co = cohere.Client(api_key=settings.cohere_api_key)
qdrant_client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key
)


@function_tool
def search_textbook_content(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search the textbook content for relevant information.

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
def search_selected_text(query: str, selected_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Search for information related to user-selected text.

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


# ============================================================================
# Agent Service
# ============================================================================

class AgentService:
    def __init__(self):
        # Configure custom client for Gemini
        external_client = AsyncOpenAI(
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        # Wrap client in OpenAIChatCompletionsModel
        model = OpenAIChatCompletionsModel(
            model="gemini-2.0-flash",  # Official OpenAI-compatible model
            openai_client=external_client
        )
        
        # Create agent with custom model and tools
        self.agent = Agent(
            name="Textbook Assistant",
            instructions="You are a helpful teaching assistant for the Physical AI and Humanoid Robotics textbook. Use search_textbook_content tool to find relevant information. Always cite sources by mentioning the module and section. Provide clear explanations suitable for students. Maintain Flesch-Kincaid grade level 10-12.",
            model=model,
            tools=[search_textbook_content, search_selected_text]
        )

    async def generate_response_stream(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]] = None,
        conversation_history: List = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response using the agent with proper streaming
        """
        # Use the streamed runner (agent already has Gemini model configured)
        result = Runner.run_streamed(self.agent, query)

        async for event in result.stream_events():
            # Check for text delta events (streamed tokens)
            if event.type == "raw_response_event":
                if isinstance(event.data, ResponseTextDeltaEvent):
                    # Yield each token/delta as it arrives
                    yield event.data.delta


# Global instance
agent_service = AgentService()