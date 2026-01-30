# PDF Character Graph

Extract character relationships from PDF books and visualize them as interactive graphs.

Upload a book PDF, and the application will:
1. Extract all characters using Named Entity Recognition (NER)
2. Resolve coreferences (pronouns, aliases)
3. Analyze relationships using an LLM
4. Generate an interactive network visualization

## Features

- **Character Discovery**: Automatic detection of characters using spaCy NER
- **Coreference Resolution**: Links pronouns and aliases to character names
- **Relationship Extraction**: Uses LLM to infer relationship types and directions
- **Interactive Visualization**: Explore character networks with vis.js graphs
- **Local-First**: Default uses Ollama for free, private inference

## Project Structure

```
.
├── backend/                    # FastAPI backend
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry
│   ├── config.py               # Configuration management
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # API route definitions
│   └── services/
│       ├── __init__.py
│       ├── pdf_processor.py    # Main PDF processing orchestration
│       ├── character_extractor.py
│       ├── relationship_extractor.py
│       ├── visualization.py
│       └── text_utils.py
├── frontend/                   # React + Vite frontend
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── main.tsx
│       ├── styles.css
│       ├── api/
│       │   └── index.ts        # API client
│       └── components/
│           ├── GraphViewer.tsx
│           └── UploadZone.tsx
├── cli.py                      # Command-line interface
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore
├── LICENSE
└── README.md
```

## Quick Start

### Option A: Docker (Recommended)

The easiest way to run the application is with Docker Compose.

#### Prerequisites

- Docker and Docker Compose
- Ollama running on host machine (for local LLM inference)

#### 1. Configure Environment

```bash
# Copy the environment template
cp .env.example .env

# Edit .env to configure LLM settings:
# - USE_LOCAL_LLM=true for Ollama (free, private)
# - USE_LOCAL_LLM=false and set OPENAI_API_KEY for OpenAI
```

#### 2. Start Ollama (if using local LLM)

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama service
brew services start ollama

# Pull a model
ollama pull llama3.2
```

#### 3. Build and Run

```bash
# Build and start containers in detached mode
docker compose up --build -d

# View logs
docker compose logs -f

# Stop containers
docker compose down
```

#### 4. Access the Application

- **Frontend**: http://localhost
- **API**: http://localhost/api/process
- **Health Check**: http://localhost/health

Upload a PDF and watch the character relationship graph appear!

---

### Option B: Local Development

For development with hot reloading:

#### Prerequisites

- Python 3.10+
- Node.js 18+
- Ollama (for local LLM inference)

#### 1. Backend Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Copy environment template
cp .env.example .env
```

### 2. Ollama Setup (Local LLM)

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama service
brew services start ollama

# Pull a model
ollama pull llama3.2
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Run the Application

**Start the backend:**
```bash
# From project root
uvicorn backend.main:app --reload --port 8000
```

**Start the frontend (new terminal):**
```bash
cd frontend
npm run dev
```

Visit **http://localhost:5173** and upload a PDF!

## CLI Usage

For command-line processing without the web interface:

```bash
# Local mode (default, FREE)
python cli.py path/to/book.pdf

# Specify Ollama model
python cli.py path/to/book.pdf --local_model mistral

# Cloud mode (requires OPENAI_API_KEY)
python cli.py path/to/book.pdf --cloud

# Limit pages processed
python cli.py path/to/book.pdf --max_pages 50
```

Results are saved to the `results/` directory:
- `characters.json` - Discovered characters and frequencies
- `relationships.jsonl` - Extracted relationships with evidence
- `character_graph.html` - Interactive visualization

## Configuration

All configuration is done via environment variables. See `.env.example` for all options.

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_LOCAL_LLM` | `true` | Use Ollama (true) or OpenAI (false) |
| `LOCAL_MODEL` | `llama3.2` | Ollama model to use |
| `OPENAI_API_KEY` | - | Required for cloud mode |
| `UPLOAD_MAX_SIZE_MB` | `50` | Maximum PDF upload size |
| `CORS_ORIGINS` | `localhost:*` | Allowed frontend origins |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/process` | Upload PDF and get graph HTML |
| GET | `/api/health` | Health check |
| GET | `/health` | Root health check |

### Process PDF

```bash
curl -X POST "http://localhost:8000/api/process" \
  -F "file=@book.pdf" \
  -H "Content-Type: multipart/form-data"
```

Response:
```json
{
  "graphHtml": "<!DOCTYPE html>..."
}
```

## Security Considerations

This application includes several security measures:

1. **CORS**: Restricted to configured origins (configure in production)
2. **File Size Limits**: Prevents DoS via large uploads (default: 50MB)
3. **File Type Validation**: Only accepts PDF files
4. **Temp File Cleanup**: Automatic cleanup of uploaded files
5. **No Secrets in Code**: All sensitive config via environment variables

**Production Recommendations:**
- Use Docker Compose for deployment (`docker compose up --build -d`)
- The nginx reverse proxy handles TLS termination and rate limiting
- CORS is automatically configured for same-origin requests through nginx
- Set `DEBUG=false` and `ENVIRONMENT=production` (defaults in Docker)
- For external HTTPS, place a TLS-terminating proxy (Traefik, Caddy) in front

## Development

### Backend Development

```bash
# Run with auto-reload
uvicorn backend.main:app --reload

# Run tests (if added)
pytest
```

### Frontend Development

```bash
cd frontend

# Development server with hot reload
npm run dev

# Type checking
npm run type-check

# Production build
npm run build
```

## Docker Architecture

When running with Docker Compose, the application uses a production-grade setup:

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Network                          │
│                                                             │
│   ┌─────────────────────┐      ┌─────────────────────────┐ │
│   │   Frontend (nginx)  │      │   Backend (FastAPI)     │ │
│   │   Port 80 (exposed) │──────│   Port 8000 (internal)  │ │
│   │                     │      │                         │ │
│   │   - Static assets   │      │   - gunicorn + uvicorn  │ │
│   │   - SPA routing     │      │   - 4 worker processes  │ │
│   │   - Reverse proxy   │      │   - PDF processing      │ │
│   │   - Rate limiting   │      │   - LLM integration     │ │
│   └─────────────────────┘      └───────────┬─────────────┘ │
│                                            │               │
└────────────────────────────────────────────┼───────────────┘
                                             │
                                   ┌─────────▼─────────┐
                                   │  Ollama (host)    │
                                   │  Port 11434       │
                                   └───────────────────┘
```

**Key Features:**
- **Multi-stage builds**: Minimal image sizes (nginx alpine, python slim)
- **Non-root users**: Backend runs as `appuser`, nginx workers as `nginx`
- **Health checks**: Automatic container health monitoring
- **Rate limiting**: nginx protects against API abuse
- **Resource limits**: Memory constraints prevent runaway processes
- **Restart policies**: Containers auto-restart on failure

## How It Works

### 1. PDF Extraction
Text is extracted from PDF pages using PyPDF2, then cleaned and normalized.

### 2. Character Discovery (NER)
spaCy's Named Entity Recognition identifies PERSON entities. Coreference resolution (via fastcoref) links pronouns to named characters.

### 3. Alias Resolution
Common patterns are detected (e.g., "Elizabeth" and "Lizzie" → same character).

### 4. Interaction Graph
Characters appearing in the same text windows are linked, creating a co-occurrence graph.

### 5. Relationship Extraction
For significant character pairs, the LLM analyzes evidence sentences to determine:
- Relationship type (family, romantic, friend, enemy, etc.)
- Direction (A is father OF B, vs mutual friendship)
- Supporting quotes

### 6. Visualization
Results are rendered as an interactive vis.js network graph with:
- Color-coded relationship types
- Weighted node sizes by character importance
- Hover tooltips with relationship details

## License

[GNU General Public License](LICENSE)
