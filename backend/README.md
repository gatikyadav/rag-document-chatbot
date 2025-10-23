# RAG Chatbot Backend

FastAPI-based backend for the RAG document chatbot system.

## Structure
```
backend/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   ├── services/
│   │   ├── document_processor.py  # Document processing
│   │   ├── vector_database.py     # ChromaDB integration
│   │   └── rag_engine.py          # RAG pipeline
│   ├── utils/
│   │   └── helpers.py       # Utility functions
│   └── api/
│       └── routes.py        # API endpoints
├── documents/               # Document storage
├── data/                    # Vector database storage
├── static/                  # Static file serving
└── requirements.txt
```

## Setup

1. Create virtual environment: `python -m venv venv`
2. Activate: `source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Configure `.env` file with your API keys
5. Run: `uvicorn app.main:app --reload`

## Environment Variables

Copy `.env.example` to `.env` and configure:
- OpenAI or Anthropic API keys
- Model preferences
- Storage paths

## Status: 🚧 Under Development
```

### 9. Final Backend Directory Structure
Your backend folder should now look like this:
```
backend/
├── venv/                    # Python virtual environment
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   ├── vector_database.py
│   │   └── rag_engine.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py
│   └── api/
│       ├── __init__.py
│       └── routes.py
├── documents/
│   ├── pdfs/
│   ├── excel/
│   ├── word/
│   ├── powerpoint/
│   └── text/
├── data/
├── static/
│   └── documents/
├── requirements.txt
├── .env
├── .env.example
├── Dockerfile
└── README.md