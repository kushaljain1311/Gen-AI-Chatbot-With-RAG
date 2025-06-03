# GenAI Chatbot with PDF RAG

A GenAI-powered chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions based on uploaded PDF content. Built with Streamlit, LangChain, HuggingFace, ChromaDB, and Groq's LLaMA3.

## Features

- 📄 Upload PDF files
- 🧠 Extract and chunk PDF content
- 🔍 Store and retrieve embeddings using ChromaDB
- 🤖 Get AI-generated answers using LLaMA3 via Groq
- ✅ Evaluate answers with feedback and scores
- 💬 View chat history

## Tech Stack

- Python
- Streamlit
- LangChain
- HuggingFace Embeddings
- ChromaDB
- Groq LLaMA3
- PyPDF2
- dotenv

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/genai-chatbot-pdf.git
cd genai-chatbot-pdf
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key
```

### 5. Run the app

```bash
streamlit run main.py
```

## Folder Structure

```
genai-chatbot-pdf/
│
├── main.py
├── requirements.txt
├── .env
├── chroma_db/  # auto-created local vector store
```

## Example

1. Upload a PDF.
2. Ask a question related to the content.
3. See AI-generated answers with context.
4. View evaluation feedback and chat history.

---

> 📌 This project demonstrates a basic RAG pipeline using local document embeddings and LLaMA3 inference via Groq API.

