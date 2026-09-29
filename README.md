# 🧠 DocuMind - Smart RAG Assistant

A modern, fast, and intelligent **Retrieval-Augmented Generation (RAG)** application built with **LangChain**, **ChromaDB**, **Google Gemini**, and **Streamlit**.

Ask questions about your documents (PDF, TXT, Markdown) with real-time streaming answers, accurate page citations, and automatic model fallback.

---

## ✨ Features

- **Multi-Format Document Parsing:** Upload single or multiple PDF, TXT, and Markdown files.
- **Local Vector Storage:** Uses ChromaDB for fast similarity search without needing external cloud databases.
- **Smart Embeddings:** Powered by Google's `gemini-embedding-001`.
- **Multi-Model Fallback:** Auto-failover across Gemini models (`gemini-flash-lite-latest`, `gemini-3.5-flash`, etc.) to prevent quota exhaustion (`429 RESOURCE_EXHAUSTED`).
- **Conversational Memory:** Preserves multi-turn chat history for contextual follow-up questions.
- **Transparent Citations:** Expandable sources showing exact chunks, pages, and relevance scores.
- **Modern Glassmorphic UI:** Built with Streamlit and styled for clean, responsive interaction.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/mizanur090148/rag-app.git
cd rag-app
```

### 2. Set up virtual environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```
> Get your free Gemini API key from [Google AI Studio](https://aistudio.google.com/).

### 5. Run the application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
├── app.py                   # Main Streamlit application & RAG pipeline
├── sample_company_doc.txt   # Sample document for instant testing
├── requirements.txt         # Python dependencies
├── .env.example             # Example environment variables
├── .gitignore               # Ignored files (secrets, venv, Chroma data)
└── README.md                # Project documentation
```

---

## 🛠️ Tech Stack

- **Orchestration:** LangChain / LangGraph
- **Vector DB:** ChromaDB
- **LLM & Embeddings:** Google Generative AI (Gemini)
- **Frontend / UI:** Streamlit
