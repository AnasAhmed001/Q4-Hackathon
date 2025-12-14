# RAG Chatbot for Physical AI and Humanoid Robotics Textbook

This project implements a Retrieval Augmented Generation (RAG) chatbot that allows users to ask questions about the Physical AI and Humanoid Robotics textbook content. The system uses semantic search to find relevant information and generates contextual responses using AI.

## Features

- **Question Answering**: Ask questions about the textbook content and receive accurate answers with citations
- **Text Selection Queries**: Ask questions about selected text in the book with additional context from the knowledge base
- **Conversation History**: View and continue previous conversations during research sessions
- **Citation Support**: All answers include citations to relevant sections in the textbook
- **Responsive Design**: Works on desktop and mobile devices

## Architecture

The system consists of:

- **Frontend**: React components embedded in Docusaurus site
- **Backend**: FastAPI application with RAG capabilities
- **Vector Database**: Qdrant for semantic search
- **Embeddings**: Cohere API for text embeddings
- **AI Model**: Gemini API for response generation
- **Storage**: Neon Postgres for conversation history

## Prerequisites

- Python 3.13+
- Node.js 18+
- Docker (optional, for local development)

## Setup Instructions

### Backend Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd physical-ai-and-humanoid-robotics-textbook
   ```

2. **Set up the backend environment**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   Copy the `.env` file and update with your API keys:
   ```bash
   cp .env .env.local
   # Edit .env.local with your API keys
   ```

4. **Run the backend**:
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```

### Frontend Setup

1. **Navigate to the website directory**:
   ```bash
   cd website
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Set environment variables**:
   Create a `.env` file in the website directory:
   ```
   BACKEND_API_URL=http://localhost:8000
   ```

4. **Run the development server**:
   ```bash
   npm run start
   ```

## Configuration

### API Keys Required

You'll need the following API keys:

- **Cohere API Key**: For text embeddings (get from [Cohere Dashboard](https://dashboard.cohere.com/api-keys))
- **Gemini API Key**: For response generation (get from Google AI Studio)
- **Qdrant Cloud**: Vector database (get from Qdrant Cloud)
- **Neon Postgres**: For conversation storage (get from Neon)

### Environment Variables

Create `.env` files in both the `backend/` and `website/` directories with the following variables:

**Backend (.env):**
```env
COHERE_API_KEY=your_cohere_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
DATABASE_URL=your_neon_postgres_url
ALLOWED_ORIGIN=https://your-frontend-domain.com
BACKEND_API_URL=http://localhost:8000
```

**Website (.env):**
```env
BACKEND_API_URL=http://localhost:8000
```

## API Endpoints

### Health Check
- `GET /api/v1/health` - Check connectivity to all services

### Chat
- `POST /api/v1/chat` - Send a message and receive a response
- `POST /api/v1/chat/stream` - Streaming chat response

### Ingestion
- `POST /api/v1/ingest` - Ingest a single file
- `POST /api/v1/ingest/directory` - Ingest all files in a directory
- `POST /api/v1/ingest/docs` - Ingest all docs directory content

## Deployment

### Backend (Recommended: Railway)

1. Deploy the FastAPI backend to Railway:
   ```bash
   # Install Railway CLI and link your project
   railway login
   railway init
   railway up
   ```

2. Configure environment variables in Railway dashboard

### Frontend (Recommended: Vercel)

1. Deploy the Docusaurus site to Vercel:
   ```bash
   # Push to GitHub and connect to Vercel
   git add .
   git commit -m "Deploy to Vercel"
   git push origin main
   ```

2. Configure environment variables in Vercel dashboard

## Usage

1. **Start the backend server**:
   ```bash
   cd backend
   uvicorn src.api.main:app --reload --port 8000
   ```

2. **Start the frontend**:
   ```bash
   cd website
   npm run start
   ```

3. **Access the chat widget** on any page of the Docusaurus site

## Development

### Running Tests

Backend tests:
```bash
cd backend
pytest
```

### Adding New Content

To add new textbook content:

1. Add markdown files to the `website/docs/` directory
2. Run the ingestion endpoint to process new content:
   ```bash
   curl -X POST http://localhost:8000/api/v1/ingest/docs
   ```

## Troubleshooting

See [Troubleshooting Guide](docs/troubleshooting.md) for common issues and solutions.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.