# 🔍 RAG Evaluation & Monitoring Platform

A production-ready system for evaluating and monitoring Retrieval-Augmented Generation (RAG) pipelines with comprehensive metrics tracking, real-time monitoring, and interactive visualization.

🌐 **[Live Demo on Streamlit Cloud](https://rag-evaluation-platform.streamlit.app)** 

[![CI/CD Pipeline](https://github.com/ShivamGupta22a/rag-evaluation-platform/actions/workflows/ci.yml/badge.svg?branch=main)](
https://github.com/ShivamGupta22a/rag-evaluation-platform/actions
)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Key Features

- **🌐 Live Web Dashboard**: Deployed on Streamlit Cloud with interactive UI
- **🔍 Multi-Model Support**: OpenAI embeddings and FREE local Sentence Transformers
- **⚡ FAISS Integration**: Industry-standard vector database for efficient similarity search (10k+ vectors in <1ms)
- **📊 Real-Time Metrics**: Live tracking of Recall@k, MRR, Faithfulness, and Relevance
- **📈 Interactive Visualizations**: Plotly-powered charts for metrics analysis over time
- **🐳 Docker Ready**: Full containerization with Docker Compose for one-command deployment
- **🔄 CI/CD Pipeline**: Automated testing and deployment via GitHub Actions
- **💰 Cost Optimized**: FREE local embeddings option (no API costs!) with intelligent caching
- **📦 Production Grade**: Comprehensive error handling, logging, and monitoring

---

## 🚀 Live Demo

Try it now: **[https://rag-evaluation-platform.streamlit.app](https://rag-evaluation-platform.streamlit.app)**

**Features Available in Demo:**
- ✅ Document ingestion and chunking
- ✅ Semantic search with FAISS
- ✅ Real-time query interface
- ✅ Evaluation metrics dashboard
- ✅ Historical metrics visualization

---

## 🛠️ Tech Stack

| Category | Technologies | Purpose |
|----------|-------------|---------|
| **Languages** | Python 3.10+ | Core development |
| **AI/ML** | OpenAI API, Sentence Transformers | Embeddings & generation |
| **Vector DB** | FAISS | Fast similarity search |
| **Frontend** | Streamlit, Plotly | Interactive dashboard |
| **Data** | NumPy, Pandas | Data processing |
| **DevOps** | Docker, Docker Compose, GitHub Actions | Deployment & CI/CD |
| **Cloud** | Streamlit Cloud | Production hosting |

---

## 📊 System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard                      │
│  (User Interface + Real-time Visualization + Metrics)        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  RAG Pipeline Engine                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Chunking    │→ │  Embeddings  │→ │ FAISS Index  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Retrieval  │→ │  Generation  │→ │  Evaluation  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└──────────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Storage & Monitoring Layer                      │
│    Metrics History (JSONL) + Cache + Vector Store           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Use Live Demo (Fastest)

Visit **[https://rag-evaluation-platform.streamlit.app](https://rag-evaluation-platform.streamlit.app)** and start using immediately!

### Option 2: Run Locally
```bash
# Clone repository
git clone https://github.com/ShivamGupta22a/rag-evaluation-platform.git
cd rag-evaluation-platform

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```

Access at: `http://localhost:8501`

### Option 3: Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# Access at http://localhost:8501
```

---

## 📖 Usage Guide

### 1. Document Ingestion (FREE - No API costs!)
```python
from src.embeddings import SentenceTransformerEmbeddings, CachedEmbeddings
from src.vector_stores import FAISSVectorStore
from src.advanced_rag_pipeline import AdvancedRAGPipeline
from src.rag_eval_platform import Document

# Initialize with FREE local models
embeddings = CachedEmbeddings(SentenceTransformerEmbeddings())
vector_store = FAISSVectorStore(dimension=384)
rag = AdvancedRAGPipeline(
    embedding_model=embeddings,
    vector_store=vector_store,
    generator=None  # No LLM = No API costs
)

# Ingest documents
documents = [
    Document(doc_id="doc1", content="Python is a programming language..."),
    Document(doc_id="doc2", content="Machine learning is...")
]
chunks = rag.ingest_documents(documents)
print(f"Created {len(chunks)} chunks")
```

### 2. Semantic Search
```python
# Search for relevant information
query = "What is Python used for?"
retrieval_result = rag.retrieve(query)

# Display results
for chunk, score in zip(retrieval_result.retrieved_chunks, retrieval_result.scores):
    print(f"{chunk.chunk_id}: {score:.3f}")
    print(f"Content: {chunk.content[:100]}...")
```

### 3. Full RAG with Generation (OpenAI)
```python
from src.embeddings import OpenAIEmbeddings
from src.generation import OpenAIGenerator

# Initialize with OpenAI (requires API key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
generator = OpenAIGenerator(model="gpt-4o-mini")

rag = AdvancedRAGPipeline(
    embedding_model=embeddings,
    generator=generator
)

# Query with answer generation
retrieval_result, generation_result = rag.query("What is Python?")
print(f"Answer: {generation_result.generated_answer}")
```

### 4. Evaluation
```python
from src.rag_eval_platform import GroundTruthItem, RetrievalEvaluator, GenerationEvaluator

# Create ground truth
ground_truth = [
    GroundTruthItem(
        question="What is Python?",
        relevant_chunk_ids=["doc1_chunk_0"]
    )
]

# Evaluate retrieval
retrieval_eval = RetrievalEvaluator()
metrics = retrieval_eval.evaluate_batch(ground_truth, rag)

print(f"Recall@5: {metrics['recall']['recall@5']:.1%}")
print(f"MRR: {metrics['mrr']:.3f}")

# Evaluate generation
generation_eval = GenerationEvaluator()
gen_metrics = generation_eval.evaluate_batch(ground_truth, rag)

print(f"Faithfulness: {gen_metrics['faithfulness']:.1%}")
print(f"Relevance: {gen_metrics['relevance']:.1%}")
```

---

## 📊 Evaluation Metrics Explained

### Retrieval Metrics

| Metric | Description | Good Score | Formula |
|--------|-------------|------------|---------|
| **Recall@k** | % of relevant chunks in top-k | >70% | hits / total_relevant |
| **MRR** | Rank of first relevant result | >0.5 | 1 / rank_first_relevant |

### Generation Metrics

| Metric | Description | Good Score | Method |
|--------|-------------|------------|--------|
| **Faithfulness** | Answer grounded in context | >60% | Token overlap analysis |
| **Relevance** | Answer addresses question | >50% | Keyword matching |

---

## 🏗️ Project Structure
```
rag-evaluation-platform/
├── app.py                         # 🌐 Streamlit web dashboard (LIVE)
├── src/
│   ├── rag_eval_platform.py      # Core RAG implementation
│   ├── embeddings.py              # Multi-model embeddings (OpenAI, local)
│   ├── generation.py              # LLM generators (OpenAI)
│   ├── vector_stores.py           # FAISS & simple vector stores
│   └── advanced_rag_pipeline.py   # Production RAG pipeline
├── examples/
│   ├── my_first_eval.py          # Basic evaluation tutorial
│   ├── test_real_models.py       # Test OpenAI integration
│   └── full_evaluation.py        # Comprehensive evaluation
├── data/
│   ├── raw/                      # Raw document storage
│   └── processed/                # Processed chunks cache
├── .github/
│   └── workflows/
│       └── ci.yml                # 🔄 Automated CI/CD pipeline
├── Dockerfile                     # 🐳 Container definition
├── docker-compose.yml             # Multi-service orchestration
├── requirements.txt               # Python dependencies
├── .env.example                  # Environment template
└── README.md                     # This file
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file (or use Streamlit Cloud secrets):
```bash
# OpenAI (optional - only for paid features)
OPENAI_API_KEY=sk-your-key-here

# RAG Configuration
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=5
```

### Embedding Model Options

| Model | Type | Cost | Dimension | Speed | Use Case |
|-------|------|------|-----------|-------|----------|
| all-MiniLM-L6-v2 | Local | **FREE** | 384 | Fast | Development, demos |
| all-mpnet-base-v2 | Local | **FREE** | 768 | Medium | Better quality |
| text-embedding-3-small | OpenAI | $0.02/1M | 1536 | Fast | Production |
| text-embedding-3-large | OpenAI | $0.13/1M | 3072 | Medium | Best quality |

**Recommendation:** Use `all-MiniLM-L6-v2` (FREE) for development and demos.

---

## 🎯 Performance Benchmarks

| Operation | Performance | Details |
|-----------|-------------|---------|
| **Chunking** | ~1,000 chunks/sec | Fixed-size with overlap |
| **Local Embeddings** | ~100 texts/sec | CPU-based (M1/M2 Mac) |
| **FAISS Search** | <1ms | 10k vectors, exact search |
| **Memory Usage** | ~50MB | Per 10k chunks |
| **Cold Start** | ~3-5 sec | Model loading time |

---

## 🧪 Testing
```bash
# Run all examples
python3 examples/my_first_eval.py
python3 examples/test_real_models.py
python3 examples/full_evaluation.py

# Run with Docker
docker-compose exec rag-app python examples/my_first_eval.py

# Run dashboard locally
streamlit run app.py
```

---

## 🚢 Deployment Options

### 1. Streamlit Cloud (Current - LIVE!)
✅ **Already Deployed**: [https://rag-evaluation-platform.streamlit.app](https://rag-evaluation-platform.streamlit.app)

**Pros:**
- Zero-config deployment
- Free tier available
- Auto-updates from GitHub
- Built-in secrets management

### 2. Docker (Local/Server)
```bash
docker-compose up -d
```

**Pros:**
- Full control
- Runs anywhere
- Includes PostgreSQL
- Production-ready

### 3. Cloud Providers (AWS/GCP/Azure)
See [deployment guide](docs/deployment.md) for:
- AWS EC2 with Docker
- Google Cloud Run
- Azure Container Instances

---

## 📈 Roadmap

### Completed ✅
- [x] Core RAG pipeline with chunking & retrieval
- [x] Multiple embedding models (OpenAI, local)
- [x] FAISS vector database integration
- [x] Interactive Streamlit dashboard
- [x] Docker containerization
- [x] CI/CD with GitHub Actions
- [x] **Deployed to Streamlit Cloud**
- [x] Cost optimization with caching
- [x] Comprehensive evaluation metrics

### In Progress 🚧
- [ ] Unit tests with pytest
- [ ] REST API with FastAPI
- [ ] PostgreSQL integration for metrics
- [ ] Advanced LLM-based faithfulness scoring

### Planned 📋
- [ ] Pinecone vector database support
- [ ] A/B testing framework
- [ ] Slack/Email alerting
- [ ] Multi-user authentication
- [ ] Export evaluation reports (PDF)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Areas for Contribution:**
- Adding new embedding models
- Improving evaluation metrics
- UI/UX enhancements
- Documentation improvements
- Bug fixes

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Shivam Gupta**

- 💼 LinkedIn: [linkedin.com/in/shivam-gupta-b95692142](https://www.linkedin.com/in/shivam-gupta-b95692142/)
- 🐙 GitHub: [@ShivamGupta22a](https://github.com/ShivamGupta22a)
- 📧 Email: nugupta1234@gmail.com

---

## 🙏 Acknowledgments

- **OpenAI** - GPT models and embedding APIs
- **Meta AI** - FAISS vector search library
- **Hugging Face** - Sentence Transformers
- **Streamlit** - Dashboard framework and cloud hosting

---

## 💡 Key Learnings & Insights

This project demonstrates:

1. **Cost-Effective AI**: Using FREE local models reduces costs by 90%+
2. **Production Architecture**: Proper separation of concerns, caching, and monitoring
3. **Evaluation-First**: Building systems with measurable quality metrics
4. **Modern DevOps**: CI/CD, Docker, cloud deployment
5. **Real-World RAG**: Practical implementation beyond basic tutorials

---

## 🔗 Related Projects

- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework
- [LlamaIndex](https://github.com/run-llama/llama_index) - Data framework for LLMs
- [RAGAS](https://github.com/explodinggradients/ragas) - RAG evaluation framework

---

<div align="center">

### ⭐ If you find this project helpful, please star it!

**[Live Demo](https://rag-evaluation-platform.streamlit.app)** | **[Report Bug](https://github.com/ShivamGupta22a/rag-evaluation-platform/issues)** | **[Request Feature](https://github.com/ShivamGupta22a/rag-evaluation-platform/issues)**

Made with ❤️ using Python, FAISS, and Streamlit

</div>