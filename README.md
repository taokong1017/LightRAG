# LightRAG

A lightweight Retrieval-Augmented Generation (RAG) system designed for efficient, user-friendly, and scalable knowledge retrieval and question answering. Supports multiple storage backends, heterogeneous knowledge fusion, entity/relation extraction, and streaming APIs.

## Features
- 🚀 **High Performance**: Async concurrency, batch processing, chunk caching, and large-scale document support.
- 🧠 **Intelligent Extraction**: Built-in entity/relation extraction and knowledge graph construction.
- 🔗 **Multi-Backend Support**: Works with NanoVectorDB, Postgres, MongoDB, OpenSearch, and more.
- 🛡️ **API Security**: Supports API-Key and JWT authentication.
- 🌐 **Streaming API**: FastAPI-based high-performance RESTful and streaming endpoints.
- ⚙️ **Extensible**: Modular design for easy customization and integration.

## Installation

### Requirements
- Python >= 3.10
- Recommended: Linux server environment

### Quick Install
```bash
curl -LsSf https://raw.githubusercontent.com/taokong1017/LightRAG/master/install.sh | bash
```

## Quick Start

### Start the API Server
```bash
python server.py
```

### API Endpoints
- Swagger: `http://localhost:9621/docs`


## Configuration
- Configuration is supported via a `.env` file or environment variables for storage, model, API key, and other settings. You can create your own `.env` by copying and editing the provided `.env.example` file.

## Dependencies
See [requirements.txt](requirements.txt) and [pyproject.toml](pyproject.toml) for details. Major dependencies include:
- fastapi, uvicorn, aiohttp, pydantic, tiktoken, nano-vectordb, pymongo, opensearch-py, asyncpg, langfuse, etc.

## Contributing
Contributions are welcome!
1. Fork this repo
2. Create a new branch for your feature/fix
3. Submit a PR with a clear description

## License
MIT License
