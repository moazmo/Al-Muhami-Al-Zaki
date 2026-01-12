# Al-Muhami Al-Zaki (المحامي الذكي)
## The Intelligent Lawyer — Corrective RAG for Egyptian Law

> "The Lawyer Who Doesn't Lie" — A system that retrieves, grades, and validates legal information before answering.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Law 151 Compliant](https://img.shields.io/badge/Law%20151%2F2020-Compliant-orange.svg)
![Knowledge Base](https://img.shields.io/badge/Knowledge_Base-2600+_Articles-brightgreen.svg)
![Ollama](https://img.shields.io/badge/LLM-Ollama_Local-purple.svg)

---

## 🎯 Overview

**Al-Muhami Al-Zaki** is a Corrective RAG (CRAG) system designed for Egyptian legal research. Unlike standard RAG systems that may hallucinate, this system:

1. **Retrieves** relevant legal documents from a 2,600+ article knowledge base
2. **Grades** each document for relevance using a local LLM
3. **Validates** context before answering — admits ignorance when unsure
4. **Generates** answers with **mandatory source citations** (Article, Law, Year)
5. **Self-corrects** by rewriting queries when initial retrieval fails

---

## ✨ Key Capabilities

| Capability | Description |
|------------|-------------|
| 📚 **2,600+ Legal Articles** | Civil Code, Penal Code, Constitution, Personal Status, Criminal Procedure |
| 🔍 **Semantic Arabic Search** | Multilingual E5-Large embeddings optimized for Arabic legal text |
| ⚖️ **Mandatory Citations** | Every answer includes specific Law Number, Article, and Year |
| 🔒 **Privacy Compliant** | CAMeLBERT-NER for PII anonymization (Law 151/2020) |
| 🔄 **Self-Correction** | Automatic query rewriting when retrieval confidence is low |
| 💸 **Zero API Cost** | Fully local with Ollama — no cloud LLM bills |
| ⚡ **Fast Response** | ~14 seconds average response time with GPU acceleration |

---

## 🎬 How It Works

```
User: "ما هي عقوبة السرقة في القانون المصري؟"

┌─────────────────────────────────────────────────────────────────┐
│ 1. RETRIEVE: Search 2,600+ articles for "عقوبة السرقة"          │
│    → Found 5 relevant chunks from قانون العقوبات               │
├─────────────────────────────────────────────────────────────────┤
│ 2. GRADE: LLM evaluates each chunk for relevance                │
│    → 3/5 chunks marked as relevant                              │
├─────────────────────────────────────────────────────────────────┤
│ 3. GENERATE: Synthesize answer with citations                   │
│    → "وفقاً للمادة 318 من قانون العقوبات لسنة 1937..."         │
└─────────────────────────────────────────────────────────────────┘

Response includes:
✓ Direct answer to the legal question
✓ Specific article citations (المادة 318)
✓ Law name and year (قانون العقوبات 1937)
✓ Disclaimer to consult a real lawyer for specific cases
```

---

## 📖 Example Output

**Question**: ما هي عقوبة السرقة في القانون المصري؟

**Response** *(actual system output)*:

> لذا، وفقاً للمادة 318 من قانون العقوبات المصري لسنة 2017:
>
> **السُّرقة:**
> - **يعاقب بالحبس مدة لا تقل عن ستة أشهر ولا تتجاوز سنتين، أو بغرامة مالية لا تقل عن خمسة آلاف جنيه ولا تتجاوز عشرين ألف جنيه، أو بإحدى هاتين العقوبتين.**
>
> إذا كنت تحتاج إلى مزيد من التفاصيل أو الاستفسارات، يُنصح بالرجوع إلى النص القانوني الأصلي أو استشارة محامٍ متخصص.
>
> **المصدر:** المادة 318 من قانون العقوبات المصري لسنة 2017.

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
│         Qdrant Vector Search (2,600+ articles)                  │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                          GRADE                                  │
│              Ollama (llama3.1:8b) - Relevance Check             │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │ GENERATE │  │ REWRITE  │  │ DECLINE  │
              │ (Answer) │  │ (Retry)  │  │ (Honest) │
              └──────────┘  └──────────┘  └──────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- GPU recommended (RTX 3060 or better)

### 1. Clone & Install

```bash
git clone https://github.com/moazmo/Al-Muhami-Al-Zaki.git
cd Al-Muhami-Al-Zaki

python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate

pip install -r requirements.txt
```

### 2. Download Ollama Models

```bash
ollama serve  # Start in separate terminal
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Qdrant Cloud credentials
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
│   ├── ingest/           # Document processing pipeline
│   │   ├── loader.py     # PDF/TXT/DOCX loading
│   │   ├── chunker.py    # Arabic-aware legal splitting
│   │   ├── anonymizer.py # PII masking (Law 151)
│   │   └── embedder.py   # E5 embedding + Qdrant
│   │
│   ├── graph/            # CRAG State Machine
│   │   ├── nodes.py      # retrieve, grade, generate
│   │   ├── edges.py      # Conditional routing
│   │   └── builder.py    # LangGraph compilation
│   │
│   └── prompts/          # Arabic LLM Prompts
│       ├── grader.py     # Relevance scoring
│       └── generator.py  # Citation-aware generation
│
├── scripts/
│   ├── ingest_laws.py    # Document ingestion CLI
│   └── test_crag.py      # Query testing
│
├── app.py                # Streamlit UI (RTL Arabic)
└── requirements.txt
```

---

## 🔒 Privacy & Compliance

Designed for **Egypt Data Protection Law 151/2020** compliance:

- **PII Anonymization**: Names and locations masked using CAMeLBERT-NER
- **Audit Trail**: Every anonymization logged for compliance review
- **No Storage**: User queries are not persisted

```
Input:  "حكم ضد أحمد علي المقيم في القاهرة"
Output: "حكم ضد [شخص] المقيم في [مكان]"
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Framework** | LangGraph | Cyclic state machine for CRAG |
| **Vector DB** | Qdrant Cloud | 2,600+ legal article vectors |
| **Embeddings** | multilingual-e5-large | Arabic-optimized, GPU-accelerated |
| **Grader LLM** | Ollama (llama3.1:8b) | Local relevance scoring |
| **Generator** | Ollama (qwen2.5:7b) | Citation-aware generation |
| **UI** | Streamlit | RTL Arabic interface |
| **NLP** | CAMeLBERT-NER | Arabic PII detection |

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 👤 Author

**moazmo**
- GitHub: [@moazmo](https://github.com/moazmo)

---

<div align="center">
  <strong>🏛️ Building the future of Legal AI in Egypt 🏛️</strong>
</div>
