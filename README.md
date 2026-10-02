# AI PDF Question Answering System (RAG)

An AI-powered PDF Question Answering System built using Python, Sentence Transformers, FAISS, and Gemini API.

The application reads a PDF, converts its content into vector embeddings, stores them in a FAISS vector database, retrieves the most relevant information based on the user's question, and generates accurate answers using Google's Gemini API.

---

## Features

- Read PDF documents
- Automatic text chunking
- Generate semantic embeddings
- Store embeddings using FAISS
- Similarity-based search
- Context-aware question answering
- Gemini API integration
- Interactive command-line interface
- Error handling

---

## Tech Stack

- Python
- Google Gemini API
- Sentence Transformers
- FAISS
- PyPDF
- NumPy
- python-dotenv

---

## Project Workflow

```text
PDF
   ↓
Read Text
   ↓
Create Chunks
   ↓
Generate Embeddings
   ↓
Store in FAISS
   ↓
User Question
   ↓
Query Embedding
   ↓
Similarity Search
   ↓
Relevant Context
   ↓
Gemini API
   ↓
Final Answer
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/ai-pdf-question-answering-system.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```text
API_KEY=YOUR_GEMINI_API_KEY
```

Run the project

```bash
python main.py
```

---

## Example

```
Ask Question:
What is Artificial Intelligence?

Answer:
Artificial Intelligence is the simulation of human intelligence by machines that can learn, reason, and solve problems.
```

---

## Folder Structure

```
ai-pdf-question-answering-system/
│
├── main.py
├── sample.pdf
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## Learning Outcomes

Through this project, I learned:

- PDF text extraction
- Text chunking
- Embedding generation
- Vector databases
- FAISS similarity search
- Retrieval-Augmented Generation (RAG)
- Gemini API integration
- Building AI-powered document assistants

---

## Future Improvements

- Support multiple PDFs
- Streamlit Web Interface
- Chat history
- Source citation
- Upload PDF from UI
- Conversation memory
- Multi-document retrieval

---

## Author

Mudit Joshi
