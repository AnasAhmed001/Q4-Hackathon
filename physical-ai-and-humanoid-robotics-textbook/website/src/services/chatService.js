import { streamMessage, healthCheck } from './apiClient';

// Session management
export async function getSessionId() {
  // Try to get session ID from localStorage
  let sessionId = localStorage.getItem('chat_session_id');

  if (!sessionId) {
    // Generate a new session ID if none exists
    sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('chat_session_id', sessionId);
  }

  return sessionId;
}

// Get conversation history from local storage
export async function getConversationHistory(sessionId) {
  const history = localStorage.getItem(`chat_history_${sessionId}`);
  if (history) {
    try {
      const parsed = JSON.parse(history);
      // Convert timestamp strings back to Date objects
      return parsed.map(msg => ({
        ...msg,
        timestamp: new Date(msg.timestamp)
      }));
    } catch (e) {
      console.error('Error parsing conversation history:', e);
      return [];
    }
  }
  return [];
}

// Save message to conversation history
export async function saveMessageToHistory(sessionId, message) {
  try {
    const history = await getConversationHistory(sessionId);
    const updatedHistory = [...history, message];
    localStorage.setItem(`chat_history_${sessionId}`, JSON.stringify(updatedHistory));
  } catch (e) {
    console.error('Error saving message to history:', e);
  }
}

// Clear conversation history
export async function clearConversationHistory(sessionId) {
  localStorage.removeItem(`chat_history_${sessionId}`);
}

// Stream message to backend (streaming only now)
export async function* streamMessageFromBackend(message, sessionId, selectedText) {
  try {
    for await (const chunk of streamMessage(message, sessionId, selectedText)) {
      yield chunk;
    }
  } catch (error) {
    console.error('Error streaming message from backend:', error);
    throw error;
  }
}

// Health check
export async function checkHealth() {
  try {
    const response = await healthCheck();
    return response;
  } catch (error) {
    console.error('Health check failed:', error);
    return null;
  }
}

// Export the main service object
export const chatService = {
  getSessionId,
  getConversationHistory,
  saveMessageToHistory,
  clearConversationHistory,
  streamMessage: streamMessageFromBackend,
  checkHealth
};