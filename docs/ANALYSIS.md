# Al-Muhami Al-Zaki — Technical Analysis & Roadmap

> Last Updated: 2026-01-12 | Status: **MVP Complete**

---

## 1. System Architecture

### High-Level Overview

```text
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
│ (2,600 vecs) │      │ (llama3.1)   │
└──────────────┘      └──────────────┘
```

### CRAG Flow

1. **Retrieve**: Query Qdrant for top-5 similar legal chunks
2. **Grade**: LLM evaluates relevance of each chunk (binary: relevant/irrelevant)
3. **Route**: 
   - If relevant docs found → Generate answer with citations
   - If no relevant docs → Rewrite query and retry (max 2 attempts)
   - If max retries reached → Honestly decline to answer
4. **Generate**: Synthesize answer with mandatory article citations

---

## 2. Knowledge Base

### Current Coverage

| Document | Chunks | Key Content |
|----------|--------|-------------|
| Civil Code 1948 | ~54 | Contracts, property, obligations |
| Criminal Code 1937 | ~282 | Penalties, crimes, procedures |
| Family Laws 2000 | ~179 | Marriage, divorce, inheritance |
| Personal Status | ~7 | Custody, alimony, rights |
| Constitution 2014 | ~10 | Fundamental rights, governance |
| Criminal Procedure | ~19 | Investigation, trial procedures |

**Total**: ~2,600 legal article chunks in Qdrant Cloud

---

## 3. Module Breakdown

### 3.1 Ingestion Pipeline (`src/ingest/`)

| File | Purpose | Key Features |
|------|---------|--------------|
| `loader.py` | Load PDF/TXT/DOCX | pdfplumber for Arabic PDFs |
| `chunker.py` | Arabic-aware splitting | Respects article boundaries (مادة/باب/فصل) |
| `anonymizer.py` | PII masking | CAMeLBERT-NER, Law 151 compliant |
| `embedder.py` | GPU embedding | Singleton cache for 7s speedup |

### 3.2 CRAG Graph (`src/graph/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `nodes.py` | Graph nodes | `retrieve()`, `grade()`, `generate()`, `rewrite()` |
| `edges.py` | Routing logic | Routes to generate/rewrite/decline |
| `builder.py` | Graph compilation | LangGraph state machine |

### 3.3 Prompts (`src/prompts/`)

- **grader.py**: Binary relevance scoring in Arabic
- **generator.py**: Citation-aware generation (Arabic-only output)
- **rewriter.py**: Query reformulation for retry attempts

---

## 4. Key Technical Decisions

### Why Corrective RAG?

Standard RAG can hallucinate when documents aren't relevant. CRAG adds:
- **Grading step**: Filter irrelevant chunks before generation
- **Rewrite loop**: Self-correct when initial retrieval fails
- **Honest decline**: Admit "I don't know" rather than hallucinate

### Why Local LLM (Ollama)?

| Benefit | Impact |
|---------|--------|
| Zero API cost | Unlimited queries during development |
| Privacy | Legal documents never leave local machine |
| Speed | No network latency for grading |
| Arabic quality | llama3.1 handles Arabic well |

### Why E5-Large for Embeddings?

- **Multilingual**: Native Arabic support
- **GPU acceleration**: RTX 3060 → ~0.5s per batch
- **E5 prefixes**: `query:` and `passage:` for optimal retrieval

---

## 5. Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Avg Response Time** | ~14s | With GPU, embedder cached |
| **Retrieval Latency** | ~2s | Qdrant Cloud roundtrip |
| **Grading Latency** | ~6s | Ollama llama3.1:8b |
| **Generation Latency** | ~6s | Ollama qwen2.5:7b |
| **Knowledge Base** | 2,600 chunks | ~500KB total text |

### Bottlenecks Addressed

1. ✅ **Embedder Caching**: Singleton pattern saves 7s model load per query
2. ✅ **GPU Acceleration**: E5-Large runs on CUDA
3. ✅ **Batch Processing**: 20 chunks per Qdrant upload

---

## 6. Future Roadmap

### Phase 1: Data Expansion (High Priority)

| Task | Impact |
|------|--------|
| Complete Penal Code (400+ articles) | More comprehensive criminal law coverage |
| Commercial Law | Business and contract disputes |
| Labor Law | Employment-related questions |

### Phase 2: UX Improvements (Medium Priority)

| Task | Purpose |
|------|---------|
| Streaming responses | Reduce perceived latency |
| Multi-turn chat | Conversation memory |
| Source highlighting | Show exact article text |

### Phase 3: Production Readiness

| Task | Purpose |
|------|---------|
| Dockerfile | Containerized deployment |
| Rate limiting | API protection |
| Logging dashboard | Query analytics |

---

## 7. Quick Commands

```bash
# Start environment
.\venv\Scripts\Activate  # Windows
source venv/bin/activate # Linux/Mac

# Start Ollama (separate terminal)
ollama serve

# Test single query
python scripts/test_crag.py --query "ما هي عقوبة السرقة؟"

# Ingest new law
python scripts/ingest_laws.py \
  --input "data/raw/law.pdf" \
  --source-name "قانون جديد" \
  --source-type law \
  --law-year 2024 \
  --skip-anonymization

# Start UI
streamlit run app.py
```

---

## 8. Environment Variables

```env
# Qdrant Cloud (Required)
QDRANT_URL=https://xxx.cloud.qdrant.io:6333
QDRANT_API_KEY=xxx
QDRANT_COLLECTION_NAME=egyptian_law

# Models (Ollama - Local)
GRADER_MODEL=llama3.1:8b
GENERATOR_MODEL=qwen2.5:7b
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# Application Settings
RETRIEVAL_TOP_K=5
MAX_REWRITE_ATTEMPTS=2
```

---

## 9. Resources

- **GitHub**: [github.com/moazmo/Al-Muhami-Al-Zaki](https://github.com/moazmo/Al-Muhami-Al-Zaki)
- **Qdrant Docs**: [qdrant.tech/documentation](https://qdrant.tech/documentation/)
- **LangGraph Docs**: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
- **Ollama**: [ollama.ai](https://ollama.ai/)
