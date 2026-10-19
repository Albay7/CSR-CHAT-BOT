# 🎧 CSR Agent - Customer Service Representative AI

An intelligent Retrieval-Augmented Generation (RAG) assistant designed to handle customer support inquiries across billing, returns & refunds, account access, technical support, outages, and general customer care.

Powered by **FastAPI**, **Google Gemini** (`gemini-3.8-flash`), and Hugging Face's real-world customer support ticket knowledge base ([`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)).

---

## 📁 Clean Architecture

```text
csr-agent/
├── .github/
│   └── workflows/
│       └── main_azure-deploy.yml     # Azure App Service CI/CD GitHub Actions workflow
├── data/
│   └── csr_knowledge_base.json       # Hugging Face customer support tickets knowledge base
├── static/
│   └── index.html                    # Responsive, real-time CSR Agent chat UI
├── main.py                           # FastAPI application & API endpoints
├── rag_engine.py                     # Semantic vector embeddings & Gemini RAG generation
├── startup.sh                        # Azure App Service Linux startup script
├── requirements.txt                  # Minimal Python dependencies
├── .env.example                      # Environment variables template
├── .gitignore                        # Git exclusion rules (protects .env)
└── README.md                         # Documentation
```

---

## 🚀 Quick Start (Local Setup)

### 1. Activate the Virtual Environment

On **Windows (PowerShell)**:
```powershell
.venv\Scripts\Activate.ps1
```

On **macOS / Linux**:
```bash
source .venv/bin/activate
```

---

### 2. Configure Environment Variables

Create `.env` and add your Google Gemini API key:

```dotenv
MODEL_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_CHAT_MODEL=gemini-3.8-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

---

### 3. Run the CSR Agent

```powershell
.\.venv\Scripts\python -m uvicorn main:app --reload --port 8000
```

Open your browser and navigate to:
**http://localhost:8000**

Interactive API documentation (Swagger UI) is available at:
**http://localhost:8000/docs**

---

## ☁️ Azure App Service Deployment

1. Set the Startup Command in Azure Configuration > General settings:
   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```
2. Set Environment Variables in Azure Configuration > Application settings:
   - `GEMINI_API_KEY`
   - `MODEL_PROVIDER=gemini`
   - `SCM_DO_BUILD_DURING_DEPLOYMENT=true`
3. Push to `main` branch to deploy via GitHub Actions.

---

## 🛡️ License

MIT License.
