# Al-Muhami Al-Zaki — Technical Analysis & Roadmap

> Generated: 2026-01-09 | Status: MVP Complete

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Al-Muhami Al-Zaki                           │
│              Corrective RAG for Egyptian Law                    │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Ingest     │      │    Graph     │      │     UI       │
│   Pipeline   │      │   (CRAG)     │      │  (Streamlit) │
└──────────────┘      └──────────────┘      └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ loader.py    │      │ retrieve     │      │ app.py       │
│ chunker.py   │      │ grade        │      │ RTL Arabic   │
│ anonymizer   │      │ generate     │      │ Chat UI      │
│ embedder.py  │      │ rewrite      │      └──────────────┘
└──────────────┘      └──────────────┘
        │                     │
        ▼                     ▼
┌──────────────┐      ┌──────────────┐
│ Qdrant Cloud │      │ Ollama Local │
│ (Vectors)    │      │ (qwen2.5:7b) │
└──────────────┘      └──────────────┘
```

---

## 2. Module Breakdown

### 2.1 Ingestion Pipeline (`src/ingest/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `loader.py` | Load PDF/TXT/DOCX | `DocumentLoader.load()` |
| `chunker.py` | Arabic-aware splitting | `LegalChunker.chunk()`, `chunk_by_article()` |
| `anonymizer.py` | PII masking (Law 151) | CAMeLBERT-NER based |
| `embedder.py` | GPU-enabled embedding | `LegalEmbedder.embed_and_upload()` |
| `schemas.py` | Pydantic models | `LegalChunkPayload` |

### 2.2 CRAG Graph (`src/graph/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `state.py` | TypedDict for graph state | `GraphState`, `create_initial_state()` |
| `nodes.py` | Graph nodes | `retrieve()`, `grade_documents()`, `generate()`, `rewrite_query()`, `no_answer()` |
| `edges.py` | Conditional routing | `route_after_grading()` |
| `builder.py` | Graph compilation | `build_crag_graph()`, `run_query()` |

### 2.3 Prompts (`src/prompts/`)

| File | Purpose |
|------|---------|
| `grader.py` | Relevance scoring prompt (lenient) |
| `generator.py` | Answer generation with citations |
| `rewriter.py` | Query reformulation |

### 2.4 Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `ingest_laws.py` | CLI for document ingestion |
| `benchmark_egymmlu.py` | Full CRAG+RAG benchmark |
| `benchmark_simple.py` | LLM-only benchmark |
| `test_crag.py` | Single query test |
| `evaluate_ragas.py` | RAGAS metrics |

---

## 3. Current Performance

### Benchmark Results (Sample of 10)

| Metric | Score |
|--------|-------|
| **Accuracy** | 40.0% |
| **Faithfulness** | 90.0% |
| **Retrieval Rate** | 10.0% |
| **Avg Latency** | ~42s |

### Knowledge Base

| Document | Status |
|----------|--------|
| Civil Code 1948 | ✅ Ingested |
| Constitution 2014 | ⏳ Available |
| Penal Code | ✅ Ingested |
| Criminal Procedure | ✅ Ingested |
| Personal Status Law | ✅ Ingested |

---

## 4. Roadmap

### Phase 1: Optimize (Priority: High)

- [ ] **Tune Grader Prompt**: Make more lenient, add examples
- [ ] **Run Full Benchmark**: Get official accuracy score
- [ ] **Cache Embedder**: Avoid reloading model per query

### Phase 2: Expand (Priority: Medium)

- [ ] **Ingest More Laws**: Commercial, Labor, Administrative
- [ ] **Custom Q&A Dataset**: Create questions matching our data
- [ ] **Hybrid Search**: Combine vector + keyword (BM25)

### Phase 3: Production (Priority: Low)

- [ ] **Dockerfile**: Containerization
- [ ] **Unit Tests**: Achieve 80%+ coverage
- [ ] **Error Handling**: Graceful failures
- [ ] **Streaming**: Real-time answer display
- [ ] **Multi-turn Chat**: Conversation memory

---

## 5. Tech Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Framework | LangGraph | State machine orchestration |
| Vector DB | Qdrant Cloud | Free tier |
| Embeddings | multilingual-e5-large | GPU-accelerated |
| LLM | Ollama (qwen2.5:7b) | Local, unlimited |
| UI | Streamlit | RTL Arabic support |
| NLP | CAMeL Tools | Arabic NER for anonymization |

---

## 6. Environment Variables

```env
# API Keys
GROQ_API_KEY=...
GOOGLE_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...

# Models (Ollama)
GRADER_MODEL=qwen2.5:7b
GENERATOR_MODEL=qwen2.5:7b
EMBEDDING_MODEL=intfloat/multilingual-e5-large
```

---

## 7. Quick Commands

```bash
# Start Ollama
ollama serve

# Test single query
python scripts/test_crag.py --query "ما هي حقوق الملكية؟"

# Run benchmark
python scripts/benchmark_egymmlu.py --limit 10

# Ingest new law
python scripts/ingest_laws.py --input "data/raw/law.pdf" --source-name "..." --source-type law --law-year 2024 --skip-anonymization

# Start UI
streamlit run app.py
```
