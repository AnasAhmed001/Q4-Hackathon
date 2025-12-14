import React, { useState, useEffect, useRef } from 'react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { SelectionMode } from './SelectionMode';
import { chatService } from '../services/chatService';
import './ChatWidget.css';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  citations?: any[];
}

const ChatWidget: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState<string>('');
  const [isSelectionMode, setIsSelectionMode] = useState<boolean>(false);
  const [selectedText, setSelectedText] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [connectionError, setConnectionError] = useState<boolean>(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initialize session on component mount
  useEffect(() => {
    const initSession = async () => {
      try {
        const session = await chatService.getSessionId();
        setSessionId(session);

        // Load conversation history if exists
        const history = await chatService.getConversationHistory(session);
        if (history && history.length > 0) {
          setMessages(history);
        }

        // Test backend connection
        const health = await chatService.checkHealth();
        setConnectionError(!health);
      } catch (error) {
        console.error('Failed to initialize session:', error);
        setConnectionError(true);
      }
    };

    initSession();
  }, []);

  // Set up text selection detection for the entire document
  useEffect(() => {
    const handleGlobalSelection = () => {
      const selection = window.getSelection();
      if (selection && selection.toString().trim() !== '') {
        const selectedText = selection.toString().trim();
        if (selectedText && selectedText.length > 0 && selectedText.length < 1000) { // Prevent huge selections
          // Only set selected text if we're in selection mode
          if (isSelectionMode) {
            setSelectedText(selectedText);
          }
        }
      }
    };

    // Listen for selection changes
    document.addEventListener('selectionchange', handleGlobalSelection);

    return () => {
      document.removeEventListener('selectionchange', handleGlobalSelection);
    };
  }, [isSelectionMode]);

  const handleSendMessage = async (message: string) => {
    if (!message.trim() || isLoading) return;

    // Check if backend is reachable
    if (connectionError) {
      alert('Cannot connect to backend. Please check if the server is running.');
      return;
    }

    // Add user message to UI immediately
    const userMessage: Message = {
      id: Date.now().toString(),
      content: message,
      role: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    // Save user message to history
    await chatService.saveMessageToHistory(sessionId, userMessage);

    // Create assistant message placeholder for streaming
    const assistantMessageId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantMessageId,
      content: '',
      role: 'assistant',
      timestamp: new Date(),
      citations: []
    };

    setMessages(prev => [...prev, assistantMessage]);

    try {
      // Stream response from backend with timeout
      let fullResponse = '';
      let citations = [];
      let hasReceivedData = false;
      const timeoutMs = 60000; // 60 second timeout
      
      const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(() => reject(new Error('Request timeout. Please try again.')), timeoutMs);
      });

      const streamPromise = (async () => {
        for await (const chunk of chatService.streamMessage(
          message, 
          sessionId, 
          isSelectionMode && selectedText ? selectedText : undefined
        )) {
          hasReceivedData = true;

          if (chunk.error) {
            // Handle specific error types
            if (chunk.error.includes('429') || chunk.error.includes('quota')) {
              throw new Error('API quota exceeded. Please try again in a few moments.');
            } else if (chunk.error.includes('403') || chunk.error.includes('PERMISSION_DENIED')) {
              throw new Error('API key issue. Please contact support.');
            } else if (chunk.error.includes('404')) {
              throw new Error('Model not found. Please contact support.');
            } else {
              throw new Error(chunk.error);
            }
          }

          if (chunk.token) {
            fullResponse += chunk.token;
            // Update the assistant message with new token
            setMessages(prev => prev.map(msg =>
              msg.id === assistantMessageId
                ? { ...msg, content: fullResponse }
                : msg
            ));
          }

          if (chunk.done) {
            if (chunk.citations) {
              citations = chunk.citations;
            }
            // Update with final citations
            setMessages(prev => prev.map(msg =>
              msg.id === assistantMessageId
                ? { ...msg, citations, content: fullResponse }
                : msg
            ));
          }
        }
      })();

      await Promise.race([streamPromise, timeoutPromise]);

      // Save assistant message to history
      const finalAssistantMessage: Message = {
        id: assistantMessageId,
        content: fullResponse,
        role: 'assistant',
        timestamp: new Date(),
        citations
      };
      await chatService.saveMessageToHistory(sessionId, finalAssistantMessage);

      // Exit selection mode after sending
      if (isSelectionMode) {
        setIsSelectionMode(false);
        setSelectedText('');
      }
    } catch (error) {
      console.error('Error sending message:', error);

      // Provide specific error message
      let errorMessage = 'Sorry, I encountered an error processing your request.';
      
      if (error instanceof Error) {
        if (error.message.includes('timeout')) {
          errorMessage = '⏱️ Request timed out. The server took too long to respond. Please try again.';
        } else if (error.message.includes('quota')) {
          errorMessage = '⚠️ API quota exceeded. Please wait a few moments and try again.';
        } else if (error.message.includes('network') || error.message.includes('fetch')) {
          errorMessage = '🌐 Network error. Please check your internet connection.';
          setConnectionError(true);
        } else if (error.message) {
          errorMessage = `❌ ${error.message}`;
        }
      }

      // Update assistant message with error
      setMessages(prev => prev.map(msg =>
        msg.id === assistantMessageId
          ? { ...msg, content: errorMessage }
          : msg
      ));
    } finally {
      setIsLoading(false);
    }
  };

  const handleTextSelection = (selectedText: string) => {
    if (selectedText.trim()) {
      setSelectedText(selectedText);
      setIsSelectionMode(true);
    }
  };

  const toggleSelectionMode = () => {
    setIsSelectionMode(!isSelectionMode);
    if (isSelectionMode) {
      setSelectedText('');
    }
  };

  return (
    <>
      {/* Floating toggle button */}
      {!isOpen && (
        <button 
          className="chat-toggle-button" 
          onClick={() => {
            console.log('Chat button clicked!');
            setIsOpen(true);
          }} 
          title="Open Textbook Assistant"
        >
          💬
        </button>
      )}

      {/* Chat widget */}
      {isOpen && (
        <div className="chat-widget">
          <div className="chat-header">
            <div className="chat-header-left">
              <h3>📚 Textbook Assistant</h3>
            </div>
            <div className="chat-header-right">
              <button
                className={`selection-toggle ${isSelectionMode ? 'active' : ''}`}
                onClick={toggleSelectionMode}
                title={isSelectionMode ? "Exit selection mode" : "Enter selection mode"}
              >
                {isSelectionMode ? "📝" : "🔍"}
              </button>
              <button
                className="close-button"
                onClick={() => {
                  console.log('Close button clicked!');
                  setIsOpen(false);
                }}
                title="Close chat"
              >
                ✕
              </button>
            </div>
          </div>

          <div className="chat-messages" ref={chatContainerRef}>
            {connectionError && (
              <div className="connection-error">
                <span>🔴</span> Cannot connect to backend. Please check if the server is running.
              </div>
            )}

            {messages.length === 0 && !connectionError && (
              <div className="welcome-message">
                <h4>👋 Welcome!</h4>
                <p>Ask me anything about the Physical AI and Humanoid Robotics textbook.</p>
                <div className="quick-tips">
                  <span className="tip">💡 Try: "What is ROS2?"</span>
                  <span className="tip">🔍 Use selection mode to ask about specific text</span>
                </div>
              </div>
            )}
            
            {messages.map((msg) => (
              <ChatMessage
                key={msg.id}
                message={msg.content}
                role={msg.role}
                citations={msg.citations}
                timestamp={msg.timestamp}
              />
            ))}

            {isLoading && (
              <div className="loading-message">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            
            {/* Invisible div for auto-scroll */}
            <div ref={messagesEndRef} />
          </div>

          {isSelectionMode && selectedText && (
            <SelectionMode
              selectedText={selectedText}
              onCancel={() => {
                setIsSelectionMode(false);
                setSelectedText('');
              }}
            />
          )}

          <ChatInput
            onSendMessage={handleSendMessage}
            isLoading={isLoading}
            placeholder={isSelectionMode
              ? `Ask about: "${selectedText.substring(0, 40)}${selectedText.length > 40 ? '...' : ''}"`
              : "Ask about the textbook..."}
          />
        </div>
      )}
    </>
  );
};

export default ChatWidget;