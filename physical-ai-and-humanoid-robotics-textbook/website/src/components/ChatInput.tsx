import React, { useState, KeyboardEvent } from 'react';
import './ChatInput.css';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
  placeholder?: string;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  isLoading,
  placeholder = "Type your message..."
}) => {
  const [inputValue, setInputValue] = useState<string>('');

  const handleSubmit = () => {
    if (inputValue.trim() && !isLoading) {
      onSendMessage(inputValue);
      setInputValue('');
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
    // Allow Shift+Enter for new line
  };

  return (
    <div className="chat-input-container">
      <div className="chat-input-wrapper">
        <textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isLoading}
          className="chat-input-textarea"
          rows={1}
        />
        <button
          onClick={handleSubmit}
          disabled={isLoading || !inputValue.trim()}
          className={`chat-input-button ${isLoading ? 'loading' : ''}`}
        >
          {isLoading ? (
            <span className="loading-spinner">⏳</span>
          ) : (
            <span className="send-icon">➤</span>
          )}
        </button>
      </div>
    </div>
  );
};