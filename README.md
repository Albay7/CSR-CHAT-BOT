# 🎓 Cambridge Support RAG Bot

An intelligent Retrieval-Augmented Generation (RAG) assistant designed to help applicants, students, and staff navigate University of Cambridge admissions, academic structures, supervisions, college life, financial aid, and student services.

Built with **FastAPI**, **Google Gemini / OpenAI**, a responsive vanilla web interface, and semantic vector similarity search with local disk caching.

---

## 📁 Clean Architecture

```text
cambridge-support-rag-bot/
├── .github/
│   └── workflows/
│       └── main_azure-deploy.yml     # Azure App Service CI/CD GitHub Actions workflow
├── data/
│   └── cambridge_knowledge_base.json # Curated student support knowledge base
├── static/
│   └── index.html                    # Lightweight, responsive collegiate chat UI
├── main.py                           # FastAPI application & API endpoints
├── rag_engine.py                     # Semantic vector embeddings & LLM generation
├── startup.sh                        # Azure App Service Linux startup script
├── requirements.txt                  # Python production dependencies
├── .env.example                      # Environment variables template
├── .gitignore                        # Git exclusion rules
└── README.md                         # Project documentation
```

---

## ⚡ Why This Architecture?

- **Zero Cold Start Lag**: FastAPI starts up in milliseconds, unlike heavy notebook/prototyping frameworks.
- **Fast Static Delivery**: The HTML/CSS/JS frontend loads instantly in the browser without multi-megabyte bundle downloads.
- **Async & Threadpool Ready**: API endpoints run concurrently without blocking request threads.
- **Cached Embeddings**: Embeddings are calculated once and cached on disk (`.npy`), avoiding startup delay.

---

## 🚀 Quick Start (Local Setup)

### 1. Activate the Virtual Environment

On **Windows (PowerShell)**:
```powershell
.venv\Scripts\Activate.ps1
```
*(Or on Command Prompt: `.venv\Scripts\activate.bat`)*

On **macOS / Linux**:
```bash
source .venv/bin/activate
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Configure Environment Variables

Create your local `.env` file by copying the template:

**Windows (PowerShell)**:
```powershell
Copy-Item .env.example .env
```

**macOS / Linux**:
```bash
cp .env.example .env
```

Open `.env` and fill in your Gemini or Azure OpenAI credentials:

```dotenv
MODEL_PROVIDER=gemini
GEMINI_KEY=your_gemini_api_key_here
```

> **Note:** The application includes an **Offline Knowledge Mode**. If no API keys are provided, it will still retrieve and display verified articles directly from the knowledge base.

---

### 4. Run the FastAPI Application

```bash
python -m uvicorn main:app --reload --port 8000
```

Open your browser and navigate to:
**http://localhost:8000**

Interactive API documentation (Swagger UI) is available at:
**http://localhost:8000/docs**

---

## ☁️ Deploying to Azure App Service

1. Create a **Linux Web App** in the Azure Portal (Runtime: Python 3.11 or 3.12).
2. Under **Configuration > General settings**:
   - Set **Startup Command** to:
     ```bash
     python -m uvicorn main:app --host 0.0.0.0 --port 8000
     ```
     *(or `./startup.sh`)*
   - Turn **Always On** to **On** (keeps container warm).
3. Under **Configuration > Application settings**, add:
   - `GEMINI_KEY` (or Azure OpenAI variables)
   - `MODEL_PROVIDER=gemini`
   - `SCM_DO_BUILD_DURING_DEPLOYMENT=true`
4. Add your Azure **Publish Profile** to GitHub repository secrets as:
   `AZUREAPPSERVICE_PUBLISHPROFILE`
5. Push your code to the `main` branch to trigger GitHub Actions deployment.

---

## 🛡️ License

MIT License. Designed for Cambridge student and applicant support.
