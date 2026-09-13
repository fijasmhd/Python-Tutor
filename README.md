# 🐍 Python Tutor – Self-Healing RAG

An intelligent AI-powered Python tutor built using **LangGraph**, **ChromaDB**, **Groq LLMs**, and **Streamlit**. Unlike a traditional RAG chatbot, this project implements a **Self-Healing Retrieval-Augmented Generation (RAG)** pipeline that automatically evaluates its own responses, rewrites ambiguous questions, and retries retrieval before returning an answer.

---

# 📌 Overview

Traditional Retrieval-Augmented Generation (RAG) systems retrieve relevant documents from a knowledge base and ask an LLM to generate an answer. However, they do not verify whether the generated answer is actually supported by the retrieved documents.

This project introduces a **Self-Healing RAG** workflow.

Instead of trusting the first generated response, the system:

- Retrieves relevant documents
- Generates an answer
- Grades whether the answer is grounded in the retrieved context
- Automatically rewrites the user's query if necessary
- Retrieves new context
- Generates a better answer
- Returns only reliable responses

This significantly reduces hallucinations while improving answer quality.

---

# ✨ Features

- 📚 Retrieval-Augmented Generation (RAG)
- 🔍 Semantic Search using ChromaDB
- 🤖 Multi-LLM Pipeline
- 🧠 Automatic Question Rewriting
- ✅ Answer Grounding Verification
- 🔄 Self-Healing Retry Loop
- 💬 Interactive Streamlit Chat Interface
- 📖 Python Knowledge Base
- ⚡ Fast inference using Groq API

---

# 🏗️ System Architecture

```
User
   │
   ▼
Streamlit UI
   │
   ▼
LangGraph Workflow
   │
   ▼
Retrieve Documents
   │
   ▼
Generate Answer
   │
   ▼
Grade Answer
   │
 ┌─┴──────────────┐
 │                │
PASS             FAIL
 │                │
 ▼                ▼
Return      Rewrite Question
Answer             │
                   ▼
           Retrieve Again
                   │
                   ▼
            Generate Again
                   │
                   ▼
             Grade Again
```

---

# ⚙️ Workflow

## Step 1 — User asks a question

Example

> What is Python?

The question is received through the Streamlit interface.

---

## Step 2 — Retrieval

The question is converted into embeddings and searched against the Chroma vector database.

Top matching document chunks are retrieved.

Example:

```
Question

↓

Embedding

↓

Vector Similarity Search

↓

Top-3 Relevant Chunks
```

---

## Step 3 — Answer Generation

The retrieved chunks are passed to the Generation LLM.

The model is instructed to:

- Answer ONLY from retrieved documents
- Never hallucinate
- Never mention document numbers
- Respond naturally

---

## Step 4 — Answer Grading

A second LLM acts as a **grader**.

It evaluates whether

- the answer is supported by retrieved documents
- the answer actually answers the question
- no outside information is introduced

Output

```
PASS
```

or

```
FAIL
```

---

## Step 5 — Self-Healing

If grading returns FAIL,

the workflow automatically

```
Rewrite Question

↓

Retrieve Again

↓

Generate Again

↓

Grade Again
```

This retry happens until

- PASS
- Maximum retry limit reached

---

## Step 6 — Final Response

Only grounded answers are shown to the user.

Otherwise the chatbot responds with

> Sorry! I am your Python assistant. This doesn't look like what I am trained in.

---

# 📂 Project Structure

```
Python-Tutor/
│
├── app.py                 # Streamlit UI
├── rag_agent.py           # LangGraph workflow
├── ingest.py              # Knowledge ingestion
│
├── docs/
│   └── python_basics.txt
│
├── chroma_db/
│
├── .env
├── .gitignore
├── README.md
├── pyproject.toml
└── uv.lock
```

---

# 🛠 Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Backend |
| Streamlit | Frontend UI |
| LangGraph | Workflow Orchestration |
| ChromaDB | Vector Database |
| LangChain | LLM Integration |
| Groq API | LLM Inference |
| Sentence Transformers | Embeddings |
| dotenv | Environment Variables |

---

# 🤖 LLM Pipeline

### Retriever

- ChromaDB

### Generation Model

- Qwen 3 32B (Groq)

### Grader Model

- Llama 3.1 8B

### Rewrite Model

- Llama 3.1 8B

---

# 📚 Knowledge Base Pipeline

```
TXT Files

↓

Text Loader

↓

Text Splitter

↓

Embeddings

↓

ChromaDB

↓

Vector Search
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/fijasmhd/Python-Tutor.git

cd Python-Tutor
```

Install dependencies

```bash
uv sync
```

or

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file

```env
API_KEY=your_groq_api_key
```

---

# 📥 Ingest Documents

Place your knowledge base inside

```
docs/
```

Then run

```bash
uv run ingest.py
```

This will

- Load documents
- Split into chunks
- Generate embeddings
- Store vectors inside ChromaDB

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

Open

```
http://localhost:8501
```

---

# 💡 Example

### User

```
Who created Python?
```

### System

```
Retrieve Documents

↓

Generate Answer

↓

Grade Answer (PASS)

↓

Return Response
```

---

### Ambiguous Query

User

```
hlo
```

Workflow

```
Retrieve

↓

Generate

↓

FAIL

↓

Rewrite

↓

Retrieve Again

↓

Generate Again

↓

PASS or FAIL
```

---

# 🎯 Advantages of Self-Healing RAG

- Reduces hallucinations
- Improves answer reliability
- Automatically corrects ambiguous questions
- Produces grounded responses
- Better than traditional RAG pipelines

---

# 🔮 Future Improvements

- PDF support
- DOCX ingestion
- Conversation memory
- Source citation highlighting
- Hybrid search (Keyword + Vector)
- Multi-agent workflow
- Admin dashboard for document management
- Chat history
- Voice interaction

---

# 👨‍💻 Author

**Fijas Muhammed C P**

AI Engineer | Full-Stack Developer

GitHub: https://github.com/fijasmhd/Python-Tutor

---

# ⭐ If you found this project useful, consider giving it a star!