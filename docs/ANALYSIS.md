# Al-Muhami Al-Zaki — Technical Analysis & Roadmap

> Last Updated: 2026-01-11 | Status: **MVP Complete** | Accuracy: **50%**

---

## 1. Architecture Overview

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
│ (Vectors)    │      │ (llama3.1)   │
└──────────────┘      └──────────────┘
```

---

## 2. Module Breakdown

### 2.1 Ingestion Pipeline (`src/ingest/`)

| File            | Purpose                 | Key Functions                      |
| --------------- | ----------------------- | ---------------------------------- |
| `loader.py`     | Load PDF/TXT/DOCX       | `DocumentLoader.load()`            |
| `chunker.py`    | Arabic-aware splitting  | `LegalChunker.chunk()`             |
| `anonymizer.py` | PII masking (Law 151)   | CAMeLBERT-NER based                |
| `embedder.py`   | GPU-enabled embedding   | `LegalEmbedder.embed_and_upload()` |
| `schemas.py`    | Pydantic models         | `LegalChunkPayload`                |

### 2.2 CRAG Graph (`src/graph/`)

| File         | Purpose                 | Key Functions                              |
| ------------ | ----------------------- | ------------------------------------------ |
| `state.py`   | TypedDict for state     | `GraphState`, `create_initial_state()`     |
| `nodes.py`   | Graph nodes             | `retrieve()`, `grade_documents()`, `generate()` |
| `edges.py`   | Conditional routing     | `route_after_grading()`                    |
| `builder.py` | Graph compilation       | `build_crag_graph()`, `run_query()`        |

### 2.3 Prompts (`src/prompts/`)

| File           | Purpose                              |
| -------------- | ------------------------------------ |
| `grader.py`    | Binary relevance scoring             |
| `generator.py` | Answer generation with citations     |
| `rewriter.py`  | Query reformulation for retry        |

---

## 3. Current Performance

### Custom Egyptian Law Benchmark v2 (20 questions)

| Category             | Accuracy      | Score |
| -------------------- | ------------- | ----- |
| **Overall**          | **50.0%**     | 10/20 |
| Civil Code           | 🏆 100.0%     | 5/5   |
| Constitution         | ⭐ 66.7%      | 2/3   |
| Criminal Procedure   | ⭐ 50.0%      | 1/2   |
| Personal Status      | ⚠️ 33.3%     | 1/3   |
| Penal Code           | ⚠️ 14.3%     | 1/7   |
| **Avg Latency**      | ~22s          | -     |

### Knowledge Base Status

| Document             | Status       | Chunks | Coverage |
| -------------------- | ------------ | ------ | -------- |
| Civil Code 1948      | ✅ Ingested  | ~54    | Good     |
| Criminal Procedure   | ✅ Ingested  | ~19    | Fair     |
| Penal Code           | ⚠️ Partial  | ~10    | Limited  |
| Constitution 2014    | ✅ Ingested  | ~10    | Fair     |
| Personal Status Law  | ⚠️ Partial  | ~7     | Limited  |

---

## 4. Strategic Roadmap

### 🔴 Phase 1: Data Expansion (Highest Priority)

**Current bottleneck: Data coverage, not model quality.**

| Task | Impact | Effort | Status |
| ---- | ------ | ------ | ------ |
| Ingest more Penal Code articles | +15-25% accuracy | Medium | 🔲 Todo |
| Ingest more Personal Status articles | +5-10% accuracy | Low | 🔲 Todo |
| Create matching benchmark questions | Accurate measurement | Low | ✅ Done |

**Why this first?** The benchmark shows Penal Code at 14% with only 10 chunks. Adding 50+ more chunks could push this to 50%+.

---

### 🟡 Phase 2: Model Optimization (Medium Priority)

| Task | Impact | Effort | Status |
| ---- | ------ | ------ | ------ |
| Tune Grader Prompt with examples | +3-5% accuracy | Low | 🔲 Todo |
| Switch to semantic scoring (0-10) | Better relevance | Medium | 🔲 Todo |
| Cache Embedder (singleton) | -5s latency | Low | 🔲 Todo |
| Fix language leak in generator | Quality improvement | ✅ Done |

**Grader Tuning Vision:**
```python
# Add domain-specific examples:
"""
### مثال: متعلق ✅
السؤال: "ما عقوبة السرقة؟"
المستند: "مادة 318 - يعاقب على السرقة بالحبس..."
التقييم: متعلق

### مثال: غير متعلق ❌
السؤال: "ما عقوبة السرقة؟"
المستند: "مادة 1 - تسري أحكام هذا القانون..."
التقييم: غير متعلق
"""
```

---

### 🟢 Phase 3: Production Readiness (Lower Priority)

| Task | Purpose |
| ---- | ------- |
| Dockerfile | Containerization for deployment |
| Unit Tests | 80%+ coverage |
| Streaming | Real-time answer display |
| Multi-turn Chat | Conversation memory |
| Error Handling | Graceful failures |

---

## 5. Tech Stack

| Component   | Technology                  | Notes                    |
| ----------- | --------------------------- | ------------------------ |
| Framework   | LangGraph                   | State machine CRAG       |
| Vector DB   | Qdrant Cloud                | Free tier, 1639 vectors  |
| Embeddings  | multilingual-e5-large       | GPU-accelerated (RTX 3060) |
| Grader LLM  | Ollama (llama3.1:8b)        | Local, better Arabic     |
| Generator   | Ollama (qwen2.5:7b)         | Local, unlimited         |
| UI          | Streamlit                   | RTL Arabic support       |
| NLP         | CAMeL Tools                 | Arabic NER (Law 151)     |

---

## 6. Quick Commands

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

# Ingest new law
python scripts/ingest_laws.py --input "data/raw/law.pdf" \
  --source-name "قانون العقوبات" \
  --source-type law \
  --law-year 2024 \
  --skip-anonymization

# Start UI
streamlit run app.py
```

---

## 7. Environment Variables

```env
# Qdrant Cloud
QDRANT_URL=https://xxx.cloud.qdrant.io:6333
QDRANT_API_KEY=xxx
QDRANT_COLLECTION_NAME=egyptian_law

# Models (Ollama)
GRADER_MODEL=llama3.1:8b
GENERATOR_MODEL=qwen2.5:7b
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# Optional (for fallback)
GROQ_API_KEY=xxx
GOOGLE_API_KEY=xxx
```
