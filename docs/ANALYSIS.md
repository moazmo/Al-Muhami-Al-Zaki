# Al-Muhami Al-Zaki — Technical Analysis & Roadmap

> Last Updated: 2026-01-12 | Status: **MVP Complete** | Accuracy: **60%**

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
│ (~1900 vecs) │      │ (llama3.1)   │
└──────────────┘      └──────────────┘
```

### CRAG State Machine

```text
    ┌─────────┐
    │  START  │
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ retrieve│ ◄─────────────────┐
    └────┬────┘                   │
         │                        │
         ▼                        │
    ┌─────────┐                   │
    │  grade  │                   │
    └────┬────┘                   │
         │                        │
         ▼                        │
    ┌─────────────┐               │
    │   router    │               │
    └──┬────┬────┬┘               │
       │    │    │                │
       ▼    │    ▼                │
    ┌────┐  │  ┌────────┐         │
    │gen │  │  │no_answ │         │
    └──┬─┘  │  └───┬────┘         │
       │    │      │              │
       ▼    │      ▼              │
    ┌─────┐ │   ┌─────┐           │
    │ END │ │   │ END │           │
    └─────┘ │   └─────┘           │
            │                     │
            ▼                     │
       ┌─────────┐                │
       │ rewrite │────────────────┘
       └─────────┘
```

---

## 2. Module Breakdown

### 2.1 Ingestion Pipeline (`src/ingest/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `loader.py` | Load PDF/TXT/DOCX files | `DocumentLoader.load()`, `load_directory()` |
| `chunker.py` | Arabic-aware legal splitting | `LegalChunker.chunk()`, `chunk_by_article()` |
| `anonymizer.py` | PII masking (Law 151) | `ArabicAnonymizer` (CAMeLBERT-NER), `SimpleAnonymizer` (regex fallback) |
| `embedder.py` | GPU-enabled embedding | `LegalEmbedder.embed_and_upload()`, `search()` |
| `schemas.py` | Pydantic models | `LegalChunkPayload`, `DocumentMetadata` |

**Key Features:**
- **Legal-aware chunking**: Respects article boundaries (مادة/باب/فصل/بند)
- **Arabic normalization**: Removes diacritics, normalizes alef variants
- **E5 prefixes**: Uses `query:` and `passage:` prefixes for optimal retrieval

### 2.2 CRAG Graph (`src/graph/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `state.py` | TypedDict for state | `GraphState`, `create_initial_state()` |
| `nodes.py` | Graph nodes | `retrieve()`, `grade_documents()`, `generate()`, `rewrite_query()`, `no_answer()` |
| `edges.py` | Conditional routing | `route_after_grading()` |
| `builder.py` | Graph compilation | `build_crag_graph()`, `run_query()` |

**Routing Logic:**
```python
if relevant_docs >= 1:
    → generate (synthesize answer)
elif rewrite_count < max_retries:
    → rewrite (reformulate query)
else:
    → no_answer (admit ignorance)
```

### 2.3 Prompts (`src/prompts/`)

| File | Purpose | Language |
|------|---------|----------|
| `grader.py` | Binary relevance scoring | Arabic |
| `generator.py` | Answer generation with citations | Arabic-only (enforced) |
| `rewriter.py` | Query reformulation for retry | Arabic |

**Generator Constraints:**
- Arabic-only output (explicit instruction)
- Mandatory source citations (Law/Article/Year)
- MCQ answer format (A/B/C/D) for benchmarks

### 2.4 Clients (`src/clients/`)

| File | Purpose | Default |
|------|---------|---------|
| `gemini_client.py` | Google Gemini (fallback) | Disabled by default |
| `groq_client.py` | Groq/Llama (fallback) | Disabled by default |
| `qdrant_client.py` | Qdrant connection | Used for retrieval |

---

## 3. Current Performance

### Custom Egyptian Law Benchmark v2 (20 Questions)

| Category | Accuracy | Score | Notes |
|----------|----------|-------|-------|
| **Overall** | **60.0%** | 12/20 | Improved from 50% via answer extraction |
| Civil Code | 🏆 100.0% | 5/5 | Excellent coverage |
| Personal Status | ⭐ 66.7% | 2/3 | Improved from 33% |
| Constitution | ⭐ 66.7% | 2/3 | Good coverage |
| Criminal Procedure | ⭐ 50.0% | 1/2 | Fair |
| Penal Code | ⚠️ 28.6% | 2/7 | Needs more data |
| **Avg Latency** | ~23s | - | ~8s per query without embedder reload |

### Knowledge Base Status

| Document | Status | Chunks | Coverage |
|----------|--------|--------|----------|
| Civil Code 1948 | ✅ Ingested | ~54 | Excellent |
| Criminal Code 1937 | ✅ Ingested | ~282 | Good |
| Criminal Procedure | ✅ Ingested | ~19 | Fair |
| Constitution 2014 | ✅ Ingested | ~10 | Fair |
| Personal Status Law | ⚠️ Partial | ~7 | Limited |

**Total Vectors**: ~1,921 in Qdrant Cloud

---

## 4. Strategic Roadmap

### 🔴 Phase 1: Data Expansion (Highest Priority)

**Current bottleneck: Data coverage, not model quality.**

| Task | Impact | Effort | Status |
|------|--------|--------|--------|
| Ingest more Penal Code articles | +10-15% accuracy | Medium | ✅ Done (282 chunks) |
| Ingest more Personal Status articles | +5-10% accuracy | Low | 🔲 Todo |
| Improve answer extraction regex | +10% accuracy | Low | ✅ Done |

---

### 🟡 Phase 2: Model Optimization (Medium Priority)

| Task | Impact | Effort | Status |
|------|--------|--------|--------|
| Tune Grader Prompt with examples | +3-5% accuracy | Low | 🔲 Todo |
| Cache Embedder (singleton) | -5s latency | Low | 🔲 Todo |
| Fix language leak in generator | Quality | ✅ Done |
| Add Arabic-only constraint | Quality | ✅ Done |

---

### 🟢 Phase 3: Production Readiness (Lower Priority)

| Task | Purpose |
|------|---------|
| Dockerfile | Containerization for deployment |
| Unit Tests | 80%+ coverage |
| Streaming | Real-time answer display |
| Multi-turn Chat | Conversation memory |
| Error Handling | Graceful failures |

---

## 5. Tech Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Framework | LangGraph | State machine CRAG |
| Vector DB | Qdrant Cloud | Free tier, ~1.9k vectors |
| Embeddings | multilingual-e5-large | GPU-accelerated (RTX 3060) |
| Grader LLM | Ollama (llama3.1:8b) | Local, better Arabic |
| Generator LLM | Ollama (qwen2.5:7b) | Local, unlimited |
| UI | Streamlit | RTL Arabic support |
| Arabic NLP | CAMeL Tools | CAMeLBERT-NER for Law 151 |

---

## 6. Key Improvements Made

### Session Highlights (2026-01-11 → 2026-01-12)

1. **Answer Extraction Overhaul**
   - Rewrote `extract_answer_choice()` with 5 priority levels
   - Added content-based matching and scoring
   - Accuracy: 50% → **60%**

2. **Criminal Code Ingestion**
   - Ingested full Egyptian Criminal Code PDF (282 chunks)
   - Penal Code accuracy: 14% → **28%**

3. **Arabic-Only Constraint**
   - Added explicit instruction to generator prompt
   - Eliminated foreign language leakage (Chinese text bug)

4. **Grader Experimentation**
   - Tested strict vs lenient graders
   - Original lenient grader performed best

---

## 7. Quick Commands

```bash
# Activate environment
.\venv\Scripts\Activate  # Windows
source venv/bin/activate # Linux/Mac

# Start Ollama (separate terminal)
ollama serve

# Test single query
python scripts/test_crag.py --query "ما هو حق الاتفاق في القانون المدني؟"

# Run custom benchmark
python scripts/benchmark_egyptian.py

# Analyze Qdrant database
python scripts/analyze_qdrant.py

# Ingest new law
python scripts/ingest_laws.py \
  --input "data/raw/law.pdf" \
  --source-name "قانون العقوبات" \
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
GRADING_THRESHOLD=0.6
MAX_REWRITE_ATTEMPTS=2

# Optional (Cloud Fallback)
GROQ_API_KEY=xxx
GOOGLE_API_KEY=xxx
```

---

## 9. Known Issues & Limitations

1. **Embedder Reloading**: Model reloads for each query (~7s overhead) - needs singleton caching
2. **Penal Code Coverage**: Still limited despite ingestion (28.6% accuracy)
3. **Article Number Extraction**: Some PDFs don't extract article numbers properly
4. **MCQ Answer Extraction**: Some correct answers extracted as "NONE" when model uses descriptive language

---

## 10. Resources

- **GitHub**: [github.com/moazmo/Al-Muhami-Al-Zaki](https://github.com/moazmo/Al-Muhami-Al-Zaki)
- **Qdrant Docs**: [qdrant.tech/documentation](https://qdrant.tech/documentation/)
- **LangGraph Docs**: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
- **Ollama**: [ollama.ai](https://ollama.ai/)
