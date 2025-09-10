# 📚 RAG LLM App

A simple **Retrieval-Augmented Generation (RAG)** app built with **Streamlit** and **LangChain**.  
It lets you chat with **Google Gemini 2.0 Flash** using your own documents (PDF, DOCX, TXT, MD) or URLs as context.

---

## 🚀 Features

- Upload documents (`.pdf`, `.docx`, `.txt`, `.md`) or provide a website URL.
- Documents are chunked and stored in a **Chroma vector database**.
- Query documents using **RAG** or chat with the LLM without RAG.
- Powered by **Google Gemini 2.0 Flash** via `langchain-google-genai`.

---

## 📦 Installation

Clone this repo and install dependencies:

```bash
git clone <your-repo-url>
cd rag_llm_app
pip install -r requirements.txt
```
