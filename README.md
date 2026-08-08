# 🔍 Verity — Production-Ready Retrieval Augmented Generation

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?logo=docker)
![Kubernetes](https://img.shields.io/badge/Kubernetes-k3s-326CE5?logo=kubernetes)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F00)
![BM25](https://img.shields.io/badge/Retrieval-Hybrid%20BM25%2BVector-6E56CF)
![Groq](https://img.shields.io/badge/Groq-Llama%203.1-F55036?logo=groq&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-EC2%20%2B%20ECR-FF9900?logo=amazonaws)
![Auth](https://img.shields.io/badge/Auth-JWT%20%2B%20Google%20OAuth-4285F4?logo=google)
![HTTPS](https://img.shields.io/badge/HTTPS-Let's%20Encrypt-003A70)
![Prometheus](https://img.shields.io/badge/Metrics-Prometheus-E6522C?logo=prometheus)
![License](https://img.shields.io/badge/License-MIT-yellow)

> A complete, evidence-grounded Retrieval Augmented Generation system — ingest your own documents, ask natural-language questions, and get answers backed by scored, cited sources. If the evidence isn't strong enough, it says so instead of guessing. Deployed to production on AWS with Docker, Kubernetes, HTTPS, authentication, and live observability.

🔗 **Live app:** [https://farz-rag.duckdns.org](https://farz-rag.duckdns.org)
📖 **API docs:** [https://farz-rag.duckdns.org/docs](https://farz-rag.duckdns.org/docs)

---

## 📋 Table of Contents

- [Overview](#overview)
- [What's implemented](#whats-implemented)
- [Architecture](#architecture)
- [The retrieval pipeline](#the-retrieval-pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pages](#pages)
- [API Endpoints](#api-endpoints)
- [Chunking Strategies](#chunking-strategies)
- [Continuous Evaluation](#continuous-evaluation)
- [Getting Started](#getting-started)
- [Adding Your Own Documents](#adding-your-own-documents)
- [Switching LLM Providers](#switching-llm-providers)
- [Deployment](#deployment)

---

## Overview

This project implements a full RAG pipeline from scratch — not a LangChain wrapper, but a transparent, service-based architecture where each stage of retrieval and generation is visible, measurable, and independently swappable.

**What it does:**
- Ingests documents, chunks them, embeds them, and stores them in a per-user isolated vector store
- Retrieves using **hybrid search** — dense vector similarity blended with BM25 keyword scoring — then **reranks** the results before they reach the model
- Scores every retrieved chunk with a confidence label (`high` / `medium` / `low`)
- Generates answers **constrained to retrieved context only**, with citations back to source chunks
- Refuses to answer when retrieval evidence falls below a calibrated threshold, instead of hallucinating
- Authenticates users via JWT or Google OAuth, with per-user document isolation
- Serves over HTTPS with an automatically renewed Let's Encrypt certificate
- Exposes Prometheus metrics and ships with a continuous evaluation suite that checks answer quality on every deploy
- Runs locally with Docker Compose, or in production via Kubernetes (k3s) on AWS

---

## What's implemented

| Capability | Status |
|---|---|
| Ingest + normalize (dedup, formats, metadata) | ✅ |
| Hybrid retrieval — BM25 + embeddings | ✅ |
| Two-stage retrieval — ANN + reranking | ✅ |
| Source confidence scoring | ✅ |
| Constrained generation (context-only, cited) | ✅ |
| Citation-backed responses | ✅ |
| Hallucination fallback (insufficient-evidence detection) | ✅ |
| Continuous evals (adversarial + benchmark cases) | ✅ |
| Query caching | ✅ |
| Observability (Prometheus `/metrics`, structured logs) | ✅ |
| Authentication — JWT + Google OAuth, per-user data isolation | ✅ |
| HTTPS via Let's Encrypt / Traefik, custom domain | ✅ |
| Voice input & output (Web Speech API) | ✅ |
| Zero-downtime rolling deploys on Kubernetes | ✅ |

---

## Architecture

```
                         ┌─────────────────────────────────────────┐
                         │              AWS EC2 (t3.micro)          │
                         │                                           │
  Browser ── HTTPS ──▶   │   Traefik Ingress (Let's Encrypt TLS)    │
 (farz-rag.duckdns.org)  │                    │                     │
                         │                    ▼                     │
                         │   ┌─────────────────────────────────┐   │
                         │   │      k3s (Kubernetes)            │   │
                         │   │                                   │   │
                         │   │  ┌───────────────┐  ┌──────────┐ │   │
                         │   │  │  FastAPI pod   │  │ ChromaDB │ │   │
                         │   │  │  (RAG engine)  │──│   pod    │ │   │
                         │   │  └───────┬────────┘  └──────────┘ │   │
                         │   └──────────┼──────────────────────┘   │
                         └──────────────┼──────────────────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          ▼                            ▼
                    Groq API                    Google OAuth
                 (Llama 3.1 8B)                 (JWT sessions)
```

Docker images are built locally, pushed to **AWS ECR**, and pulled by k3s on deploy — with rolling updates (`maxSurge: 1, maxUnavailable: 0`) so a new pod is verified healthy via readiness probes before the old one is terminated.

---

## The retrieval pipeline

Every query passes through six stages before a sentence is generated:

| Stage | What happens |
|---|---|
| **1. Ingest & chunk** | Documents are split with recursive chunking and tagged with source metadata for traceability. |
| **2. Hybrid retrieve** | Dense vector similarity (ChromaDB, cosine) is blended with normalized BM25 keyword scoring (`α = 0.8`), so exact terms surface even when the embedding match is weak. |
| **3. Rerank** | A second, query-aware scoring pass reorders the top candidates — too slow to run over the whole corpus, cheap over the twenty finalists. |
| **4. Confidence score** | Every retrieved chunk is labeled `high` / `medium` / `low` confidence based on its similarity score. |
| **5. Constrained generate** | The LLM is instructed to answer *only* from retrieved context and to cite which source backs each claim. |
| **6. Evidence check** | If the strongest match falls under a calibrated threshold (`0.35`), the system returns a fallback response instead of guessing. |

A continuous evaluation suite (`evals/run_evals.py`) runs a fixed set of Q&A cases against the live API and has already caught a real regression — a case where the LLM summarized retrieved evidence incompletely, omitting one of three techniques mentioned in the source chunk.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI, Uvicorn, Pydantic |
| Retrieval | ChromaDB (vector), `rank-bm25` (keyword), custom hybrid scorer + reranker |
| LLM | Groq API — Llama 3.1 8B Instant (Ollama supported as a local alternative) |
| Auth | JWT (`python-jose`, `passlib`/bcrypt) + Google OAuth 2.0 |
| Container | Docker (multi-stage builds), Docker Compose |
| Registry | Amazon ECR |
| Orchestration | Kubernetes (k3s) |
| Networking | Traefik ingress, Let's Encrypt (ACME HTTP challenge), Elastic IP, DuckDNS |
| Observability | Prometheus (`prometheus-client`), structured logging (`structlog`) |
| Frontend | Vanilla HTML/CSS/JS — landing page, Google-only login, chat console with drag-and-drop upload and voice I/O |

---

## Project Structure

```
rag_system/
├── services/
│   ├── api/                        ← FastAPI RAG service
│   │   ├── dockerfile
│   │   ├── requirements.txt
│   │   └── app/
│   │       ├── main.py             ← FastAPI entry point + routes
│   │       ├── auth.py             ← JWT + Google OAuth
│   │       ├── config.py           ← Settings (env-driven)
│   │       ├── monitoring.py       ← Prometheus metrics
│   │       ├── landing.html        ← Marketing / overview page ("/")
│   │       ├── login.html          ← Google-only sign-in ("/login")
│   │       ├── static_index.html   ← Chat console ("/console")
│   │       ├── rag/
│   │       │   ├── engine.py       ← Core RAG orchestration, caching, fallback
│   │       │   ├── embeddings.py   ← Text → vector
│   │       │   ├── retriever.py    ← ChromaDB search, hybrid retrieval, rerank
│   │       │   ├── chunker.py      ← Chunking strategies
│   │       │   └── generator.py    ← LLM client (Groq / Ollama / OpenAI)
│   │       └── models/
│   │           └── schemas.py
│   │
│   └── ingestion/                  ← Standalone document indexing service
│       └── app/
│           ├── main.py
│           └── loaders/
│               ├── pdf_loader.py
│               ├── txt_loader.py
│               └── html_loader.py
│
├── evals/
│   └── run_evals.py                ← Continuous evaluation suite
│
├── kubernetes/                     ← K8s manifests for production
│   ├── namespace.yaml
│   ├── ingress.yaml                ← Traefik + Let's Encrypt TLS
│   ├── api/
│   └── chroma/
│
├── scripts/
│   ├── build.sh
│   └── deploy.sh
└── docker-compose.yml               ← Local development
```

---

## Pages

| Route | Purpose |
|---|---|
| `/` | Landing page — overview, architecture, pipeline explanation |
| `/login` | Sign in with Google (JWT session set via httpOnly cookie) |
| `/console` | Authenticated chat console — upload documents, ask questions, hear answers read aloud |
| `/docs` | Interactive OpenAPI documentation |

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/health` | — | Liveness check — used by Kubernetes liveness probe |
| GET | `/ready` | — | Readiness check |
| GET | `/metrics` | — | Prometheus scrape endpoint |
| POST | `/auth/register` | — | Create an account (username/password) |
| POST | `/auth/login` | — | Get a JWT access token |
| GET | `/auth/google` | — | Start Google OAuth flow |
| GET | `/auth/me` | Required | Current session's username |
| POST | `/auth/logout` | — | Clear session cookie |
| POST | `/ingest` | Required | Index a document into your personal collection |
| POST | `/query` | Required | Ask a question against your indexed documents |
| GET | `/stats` | — | Vector store statistics |

### Ingest a document
```bash
curl -X POST https://farz-rag.duckdns.org/ingest \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "text": "Your document text here...",
    "doc_id": "my_doc_001",
    "metadata": {"source": "manual.pdf"},
    "chunking_strategy": "recursive"
  }'
```

### Query
```bash
curl -X POST https://farz-rag.duckdns.org/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "question": "What is the return policy?",
    "top_k": 5
  }'
```

Response includes the answer, cited sources, per-chunk vector/BM25/rerank scores, confidence labels, latency, and a `hallucination_risk` / `evidence_sufficient` flag:

```json
{
  "answer": "The RAG system is built on AWS, using Kubernetes k3s and ChromaDB.",
  "sources": [{"source": "test", "relevance": 0.56, "confidence": "medium"}],
  "hallucination_risk": "low",
  "evidence_sufficient": true,
  "model": "llama-3.1-8b-instant",
  "latency_ms": 180.4
}
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

## Continuous Evaluation

```bash
cd evals
python3 run_evals.py --url https://farz-rag.duckdns.org --token YOUR_JWT_TOKEN
```

Each case checks three things: whether the system found sufficient evidence when it should have (or correctly refused when it shouldn't), whether expected keywords appear in the answer, and the retrieval scores behind the response. This catches silent quality regressions — like an incomplete summary — that wouldn't otherwise throw an error.

---

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Git

### Quick Start (Local)

```bash
git clone https://github.com/farzeen-05/rag_system.git
cd rag_system
docker-compose up -d

curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'

open http://localhost:8080/docs
```

---

## Adding Your Own Documents

Use the `/ingest` endpoint directly, or upload through the console UI at `/console` — drag and drop `.txt`, `.md`, or `.pdf` files and they're chunked and indexed automatically.

---

## Switching LLM Providers

By default the system uses **Groq** (Llama 3.1 8B Instant) for fast, free-tier-friendly inference. To switch providers, update the ConfigMap / `.env`:

```
LLM_PROVIDER=ollama       # or groq, openai
LLM_MODEL=llama3.2:1b
OLLAMA_BASE_URL=http://<host-ip>:11434
```

Ollama is supported for fully local inference — useful for development, though noticeably slower on CPU-only instances.

---

## Deployment

**Infrastructure:**
- AWS EC2 (Amazon Linux 2023, t3.micro) with an Elastic IP
- k3s Kubernetes — namespace `rag-system`
- Amazon ECR — container registry
- Traefik ingress with Let's Encrypt (automatic HTTPS)
- DuckDNS for the custom domain

**Deploy:**
```bash
docker build -t rag-api ./services/api/
docker tag rag-api:latest <account-id>.dkr.ecr.<region>.amazonaws.com/rag-api:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/rag-api:latest
kubectl rollout restart deployment/rag-api-deployment -n rag-system
```

---

## Author

**Farzeen Abdul Khadir**
ECE Graduate | ML & Full-Stack Developer | Cloud & DevOps

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://www.linkedin.com/in/farzeen-abdul-khadir-8921ba2a1)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github)](https://github.com/farzeen-05)
[![Email](https://img.shields.io/badge/Email-farzeen99453@gmail.com-EA4335?style=flat&logo=gmail)](mailto:farzeen99453@gmail.com)

---

## License

This project is licensed under the MIT License.
