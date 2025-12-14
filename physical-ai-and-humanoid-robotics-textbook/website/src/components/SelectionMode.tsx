import React from 'react';
import './SelectionMode.css';

interface SelectionModeProps {
  selectedText: string;
  onCancel: () => void;
}

export const SelectionMode: React.FC<SelectionModeProps> = ({
  selectedText,
  onCancel
}) => {
  return (
    <div className="selection-mode-banner">
      <div className="selection-content">
        <span className="selection-indicator">🔍</span>
        <span className="selection-text">
          Selected: "{selectedText.length > 100 ? selectedText.substring(0, 100) + '...' : selectedText}"
        </span>
        <button
          onClick={onCancel}
          className="selection-cancel-button"
          title="Cancel selection"
        >
          ✕
        </button>
      </div>
    </div>
  );
};