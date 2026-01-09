# Al-Muhami Al-Zaki (المحامي الذكي)
## The Intelligent Lawyer — Corrective RAG for Egyptian Law

> "The Lawyer Who Doesn't Lie" — A system that retrieves, grades, and validates legal information before answering.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Law 151 Compliant](https://img.shields.io/badge/Law%20151%2F2020-Compliant-orange.svg)
![EgyMMLU](https://img.shields.io/badge/EgyMMLU-40%25_Accuracy-yellow.svg)
![Ollama](https://img.shields.io/badge/LLM-Ollama_Local-purple.svg)

---

## 🎯 Overview

**Al-Muhami Al-Zaki** is a Corrective RAG (CRAG) system designed for Egyptian legal research. Unlike standard RAG systems that may hallucinate, this system:

1. **Retrieves** relevant legal documents from a vector database
2. **Grades** each document for relevance using a fast LLM (Llama-3)
3. **Validates** that sufficient context exists before answering
4. **Generates** answers with mandatory source citations
5. **Admits ignorance** when information is not available

### Key Features

- 🔍 **Semantic Search** on Egyptian laws using multilingual embeddings
- ⚖️ **Corrective Logic** that prevents hallucination
- 📖 **Mandatory Citations** (Law Number, Article, Year)
- 🔒 **Privacy Compliant** with Egypt Law 151/2020 (PII anonymization)
- 💸 **Zero Cost** — Uses free tiers (Groq, Gemini, Qdrant)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Question                            │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                         RETRIEVE                                │
│              Qdrant Vector Search (E5-Large)                    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                          GRADE                                  │
│              Llama-3 (Groq) - Relevance Scoring                 │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │ GENERATE │  │ REWRITE  │  │NO ANSWER │
              │ (Gemini) │  │ (Retry)  │  │  (Admit) │
              └──────────┘  └──────────┘  └──────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/moazmo/Al-Muhami-Al-Zaki.git
cd Al-Muhami-Al-Zaki

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys:
# - GROQ_API_KEY (https://console.groq.com/keys)
# - GOOGLE_API_KEY (https://aistudio.google.com/apikey)
# - QDRANT_URL & QDRANT_API_KEY (https://cloud.qdrant.io/)
```

### 3. Ingest Legal Documents

```bash
# Ingest a single law
python scripts/ingest_laws.py \
    --input data/raw/civil_code.pdf \
    --source-name "القانون المدني المصري" \
    --law-number 131 \
    --law-year 1948

# Ingest a directory
python scripts/ingest_laws.py \
    --input data/raw/ \
    --recursive \
    --source-name "القوانين المصرية" \
    --law-year 2020
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Al-Muhami-Al-Zaki/
├── src/
│   ├── ingest/          # Data Engineering (ETL)
│   │   ├── loader.py    # PDF/TXT/DOCX loading
│   │   ├── anonymizer.py # PII masking (Law 151)
│   │   ├── chunker.py   # Legal-aware text splitting
│   │   └── embedder.py  # Vector embedding & Qdrant
│   │
│   ├── graph/           # CRAG State Machine
│   │   ├── state.py     # GraphState definition
│   │   ├── nodes.py     # Retrieve/Grade/Generate
│   │   ├── edges.py     # Conditional routing
│   │   └── builder.py   # LangGraph compilation
│   │
│   ├── prompts/         # LLM System Prompts
│   │   ├── grader.py    # Relevance grader
│   │   ├── generator.py # Answer generator
│   │   └── rewriter.py  # Query rewriter
│   │
│   └── utils/           # Shared Utilities
│       ├── config.py    # Settings loader
│       └── logger.py    # Structured logging
│
├── scripts/
│   ├── ingest_laws.py   # Ingestion CLI
│   └── test_retrieval.py # Retrieval test
│
├── app.py               # Streamlit UI
├── requirements.txt
└── .env.example
```

---

## 🔒 Privacy & Compliance

This system is designed for **Egypt Data Protection Law 151/2020** compliance:

- **Anonymization Pipeline**: Names, locations, and organizations are masked before embedding
- **Audit Trail**: Every anonymization is logged for compliance review
- **No Permanent Storage**: User queries are not persisted

### Anonymization Example

```
Input:  "حكم ضد أحمد علي المقيم في القاهرة"
Output: "حكم ضد [شخص] المقيم في [مكان]"
```

---

## 📊 Evaluation

Run RAGAS evaluation against the EgyMMLU benchmark:

```bash
python scripts/evaluate_ragas.py --dataset data/egymlu_law_subset.json
```

Metrics tracked:
- **Faithfulness**: Does the answer match the sources?
- **Answer Relevance**: Is the answer useful?
- **Context Precision**: Are the right documents retrieved?

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Orchestration** | LangGraph | Cyclic state machine |
| **Vector DB** | Qdrant Cloud | Free tier, hybrid search |
| **Embeddings** | E5-Large | Multilingual, Arabic support |
| **Grader LLM** | Llama-3 (Groq) | Fast, free relevance scoring |
| **Generator LLM** | Gemini Flash | High context, free tier |
| **UI** | Streamlit | Python-only interface |
| **Arabic NLP** | CAMeLBERT-NER | PII detection |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 👤 Author

**moazmo**
- GitHub: [@moazmo](https://github.com/moazmo)
- Email: moazmo27@gmail.com

---

<div align="center">
  <strong>🏛️ Building the future of Justice in Egypt. Accuracy is Law. 🏛️</strong>
</div>
