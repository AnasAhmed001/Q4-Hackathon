// API client service with backend base URL from env (BACKEND_API_URL) and cross-origin request handling
// In Docusaurus, only env vars prefixed with DOCUSAURUS_ are available in the browser
const BACKEND_API_URL = (typeof window !== 'undefined' && window.docusaurus?.siteConfig?.customFields?.backendApiUrl) 
  || 'http://localhost:8000';

// Health check
export async function healthCheck() {
  const response = await fetch(`${BACKEND_API_URL}/api/v1/health`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

// Chat endpoints - Streaming only (unified endpoint)
export async function* streamMessage(message, sessionId, selectedText) {
  const response = await fetch(`${BACKEND_API_URL}/api/v1/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      selected_text: selectedText || null
    })
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6); // Remove 'data: ' prefix
          if (data && data !== '[DONE]') {
            try {
              const parsed = JSON.parse(data);
              yield parsed;
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

// Ingestion endpoints
export async function ingestFile(filePath, checkDuplicates = true) {
  const response = await fetch(`${BACKEND_API_URL}/api/v1/ingest`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      file_path: filePath,
      check_duplicates: checkDuplicates
    })
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

export async function ingestDocsDirectory() {
  const response = await fetch(`${BACKEND_API_URL}/api/v1/ingest/docs`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    }
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}