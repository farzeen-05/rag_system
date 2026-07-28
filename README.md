# 🔍 RAG System — Production-Ready Retrieval Augmented Generation

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?logo=docker)
![Kubernetes](https://img.shields.io/badge/Kubernetes-k3s-326CE5?logo=kubernetes)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F00)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-000000)
![AWS](https://img.shields.io/badge/AWS-EC2-FF9900?logo=amazonaws)
![License](https://img.shields.io/badge/License-MIT-yellow)

> A complete, self-hostable Retrieval Augmented Generation pipeline — ingest your own documents, ask natural-language questions, and get grounded answers. Runs entirely locally with Ollama, or swaps in OpenAI with a one-line config change. Deployed to production with Docker and Kubernetes on AWS.

🔗 **Live Demo:** [https://farz-rag.duckdns.org](https://farz-rag.duckdns.org)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Chunking Strategies](#chunking-strategies)
- [Getting Started](#getting-started)
- [Adding Your Own Documents](#adding-your-own-documents)
- [Switching to OpenAI](#switching-to-openai-optional)
- [Deployment](#deployment)

---

## Overview

This project implements a full RAG pipeline from scratch — not a LangChain wrapper, but a transparent, service-based architecture built for understanding and extending each stage of the retrieval-and-generation process.

**What it does:**
- Ingests `.pdf`, `.txt`, `.md`, and `.html` documents into a vector store
- Chunks documents using 4 selectable strategies (recursive, token-based, markdown-aware, semantic)
- Embeds and retrieves relevant chunks via ChromaDB
- Generates grounded answers using a local LLM (Ollama) or OpenAI
- Runs locally with Docker Compose, or in production via Kubernetes on AWS

---

## Architecture

```
User / Client
     │
     ▼
POST /query  ──────────────►  FastAPI (RAG Service)
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
              Retriever        Embeddings        Generator
              (ChromaDB)      (text → vector)   (Ollama / OpenAI)
                    │
                    ▼
              ChromaDB Vector Store
                    ▲
                    │
         Ingestion Service ◄── docs/sample_documents/
         (PDF / TXT / HTML loaders → chunker → embeddings)
```

Each stage — ingestion, embedding, retrieval, and generation — runs as its own service, making it straightforward to swap components (e.g. a different vector store or LLM provider) without touching the rest of the pipeline.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI, Uvicorn, Pydantic |
| Vector Store | ChromaDB |
| LLM (local) | Ollama |
| LLM (optional) | OpenAI (`gpt-4o-mini`) |
| Document Loaders | PDF, TXT, HTML |
| Chunking | Recursive, token-based, markdown-aware, semantic |
| Container | Docker, Docker Compose |
| Orchestration | Kubernetes (k3s) |
| Cloud | AWS EC2 |
| DNS | DuckDNS |
| Registry | Amazon ECR |

---

## Project Structure

```
rag-system/
├── services/
│   ├── api/                    ← FastAPI RAG service
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── app/
│   │       ├── main.py         ← FastAPI entry point
│   │       ├── config.py       ← All settings (reads from .env)
│   │       ├── rag/
│   │       │   ├── engine.py       ← Core RAG logic
│   │       │   ├── embeddings.py   ← Converts text to vectors
│   │       │   ├── retriever.py    ← ChromaDB search
│   │       │   ├── chunker.py      ← 4 chunking strategies
│   │       │   └── generator.py    ← LLM wrapper (Ollama/OpenAI)
│   │       └── models/
│   │           └── schemas.py      ← Request/response types
│   │
│   ├── ingestion/              ← Document indexing service
│   │   └── app/
│   │       ├── main.py         ← Scans and indexes documents
│   │       └── loaders/
│   │           ├── pdf_loader.py
│   │           ├── txt_loader.py
│   │           └── html_loader.py
│   │
│   └── chroma/                 ← ChromaDB vector database
│
├── kubernetes/                 ← K8s manifests for production
│   ├── namespace.yaml
│   ├── api/
│   ├── chroma/
│   └── ingestion/
│
├── docs/sample_documents/      ← Put your documents here
├── scripts/
│   ├── setup-local.sh          ← One-command local setup
│   └── deploy.sh               ← Deploy to AWS
└── docker-compose.yml          ← Local development
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check — used by Kubernetes liveness probe |
| GET | `/ready` | Readiness check |
| POST | `/ingest` | Index a document |
| POST | `/query` | Ask a question |
| GET | `/stats` | Vector store stats |
| DELETE | `/documents/{id}` | Remove a document |

### Ingest a document
```bash
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your document text here...",
    "doc_id": "my_doc_001",
    "metadata": {"source": "manual.pdf", "date": "2024-01"},
    "chunking_strategy": "recursive"
  }'
```

### Query
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the return policy?",
    "top_k": 5,
    "use_mmr": false
  }'
```

---

## Chunking Strategies

| Strategy | Best For |
|----------|----------|
| `recursive` | General text, default choice |
| `tokens` | Precise LLM context control |
| `markdown` | Markdown/structured docs |
| `semantic` | Best quality, slower indexing |

---

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Git

### Quick Start (Local)

```bash
# 1. Clone and enter the project
git clone https://github.com/farzeen-05/rag-system.git
cd rag-system

# 2. Run setup (starts everything, downloads model, indexes samples)
chmod +x scripts/setup-local.sh
./scripts/setup-local.sh

# 3. Ask a question
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'

# 4. Open interactive docs
open http://localhost:8080/docs
```

---

## Adding Your Own Documents

Drop any `.pdf`, `.txt`, `.md`, or `.html` files into `docs/sample_documents/` then run:
```bash
docker-compose --profile ingestion run --rm ingestion
```

---

## Switching to OpenAI (Optional)

By default the system runs fully offline using Ollama. To use OpenAI instead, edit `.env`:
```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-your-key-here
```
Then restart:
```bash
docker-compose restart api
```

---

## Deployment

**Infrastructure:**
- AWS EC2 (Amazon Linux 2023)
- k3s Kubernetes — namespace `rag-system`
- Amazon ECR — container registry
- DuckDNS for free domain

**Deploy to AWS:**
```bash
# 1. Launch a t3.micro EC2 instance and install k3s
# 2. Set your AWS account ID in kubernetes/api/deployment.yaml
# 3. Run the deploy script
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

---

## Author

**Farzeen Abdul Khadir**
ECE Graduate | ML & Full-Stack Developer | MLOps & Cloud

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://www.linkedin.com/in/farzeen-abdul-khadir-8921ba2a1)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github)](https://github.com/farzeen-05)

---

## License

This project is licensed under the MIT License.
