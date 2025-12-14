import React from 'react';
import './ChatMessage.css';

interface Citation {
  module: string;
  section: string;
  url: string;
  score: number;
}

interface ChatMessageProps {
  message: string;
  role: 'user' | 'assistant';
  citations?: Citation[];
  timestamp: Date;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  message,
  role,
  citations,
  timestamp
}) => {
  const isUser = role === 'user';

  // Format timestamp
  const timeString = timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <div className={`chat-message ${isUser ? 'user-message' : 'assistant-message'}`}>
      <div className="message-content">
        <div className="message-text">
          {message}
        </div>

        {citations && citations.length > 0 && (
          <div className="citations">
            <h4>Sources:</h4>
            <ul>
              {citations.map((citation, index) => (
                <li key={index} className="citation-item">
                  <a
                    href={citation.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="citation-link"
                  >
                    {citation.module} - {citation.section}
                  </a>
                  <span className="confidence-score" title={`Relevance: ${citation.score}`}>
                    ({(citation.score * 100).toFixed(0)}%)
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
      <div className="message-timestamp">
        {timeString}
      </div>
    </div>
  );
};