# 🎓 Cambridge Support RAG Bot

An intelligent Retrieval-Augmented Generation (RAG) assistant designed to help applicants, students, and staff navigate University of Cambridge admissions, academic structures, supervisions, college life, financial aid, and student services.

Built with **Streamlit**, **Azure OpenAI / OpenAI**, and semantic vector similarity search.

---

## 📁 Project Structure

```text
cambridge-support-rag-bot/
├── .github/
│   └── workflows/
│       └── main_azure-deploy.yml     # Azure App Service CI/CD GitHub Actions workflow
├── .venv/                             # Local Python virtual environment (ignored in git)
├── data/
│   └── cambridge_knowledge_base.json  # Curated student support knowledge base
├── app.py                             # Streamlit conversational chat interface
├── rag_engine.py                      # Retrieval, vector embeddings, & LLM generation
├── requirements.txt                   # Project Python dependencies
├── .env.example                       # Environment variable template
├── .gitignore                         # Git exclusion rules
└── README.md                          # Project documentation
```

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

Open `.env` and fill in your Azure OpenAI or standard OpenAI credentials:

```dotenv
MODEL_PROVIDER=azure

# Azure OpenAI settings
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

> **Note:** The application includes an **Offline Knowledge Mode**. If no API keys are provided, it will still perform keyword search and display retrieved articles directly.

---

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## ☁️ Deploying to Azure App Service

1. Create a **Linux Web App** in the Azure Portal (Runtime: Python 3.11).
2. Set the startup command in Azure App Service Configuration:
   ```bash
   python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0
   ```
3. Download the **Publish Profile** from your Azure Web App.
4. Add the publish profile content as a GitHub repository secret named:
   `AZUREAPPSERVICE_PUBLISHPROFILE`
5. Push your code to the `main` branch to trigger `.github/workflows/main_azure-deploy.yml`.

---

## 🛡️ License

MIT License. Designed for Cambridge student and applicant support.
