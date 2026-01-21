# Project 8: Legal Document Navigator - Complete Implementation Guide

A beginner-friendly guide to building a system that helps users understand legal documents by answering questions about contracts, terms, and compliance.

## 📋 Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Testing Your Application](#testing-your-application)
6. [Troubleshooting](#troubleshooting)
7. [Next Steps & Enhancements](#next-steps--enhancements)

---

## Overview

### What We're Building
A system that helps users understand legal documents by answering questions about contracts, terms, and compliance. Converts legal jargon into plain language and highlights important clauses.

### Example Questions
- "What are my non-disclosure obligations in this contract?"
- "What happens if I breach this agreement?"
- "What are the payment terms?"
- "What are my rights as a tenant?"

### How It Works
1. User uploads or selects a legal document
2. User asks questions about the document
3. System retrieves relevant clauses and sections
4. LLM explains legal terms in plain language
5. User receives clear explanation with risk highlights

### Key Features
- **Plain Language Translation**: Converts legal jargon to simple language
- **Risk Identification**: Highlights important clauses and risks
- **Document-Specific**: Answers based on user's specific document
- **Clause Extraction**: Identifies and explains specific clauses
- **Comparison**: Compare different contract versions

---

## Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of RAG concepts
- Basic knowledge of document processing

### Required Software
- Python 3.8 or higher
- pip (Python package installer)

### Required Accounts
- OpenAI API account (or alternative LLM provider)

---

## Project Structure

```
legal-navigator/
│
├── documents/             # Legal documents
│   ├── contracts/
│   ├── terms_of_service/
│   └── regulations/
│
├── user_documents/        # User-uploaded documents
│
├── vector_db/             # Vector database
│
├── .env                   # API keys
│
├── requirements.txt       # Dependencies
│
├── config.py             # Configuration
│
├── main.py               # Main application
│
└── README.md             # This file
```

---

## Step-by-Step Implementation

### Step 1: Set Up Project

```bash
mkdir legal-navigator
cd legal-navigator
python3 -m venv venv
source venv/bin/activate
mkdir -p documents/{contracts,terms_of_service,regulations} user_documents vector_db
```

### Step 2: Install Dependencies

Create `requirements.txt`:
```txt
langchain==0.1.0
openai==1.10.0
chromadb==0.4.22
sentence-transformers==2.3.1
pypdf2==3.0.1
python-docx==1.1.0
streamlit==1.31.0
python-dotenv==1.0.0
```

Install: `pip install -r requirements.txt`

### Step 3: Create Configuration

Create `config.py`:
```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DOCUMENTS_DIR = "documents"
USER_DOCS_DIR = "user_documents"
VECTOR_DB_DIR = "vector_db"
CHUNK_SIZE = 800  # Smaller chunks for legal clauses
CHUNK_OVERLAP = 150
NUM_RETRIEVED_DOCS = 5
TEMPERATURE = 0.1  # Very low for accurate legal information
MODEL_NAME = "gpt-3.5-turbo"
```

### Step 4: Create Main Application

Create `main.py`:
```python
import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from config import *

def load_legal_document(file_path, doc_type='contract'):
    """Load a legal document."""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        return None
    
    documents = loader.load()
    for doc in documents:
        doc.metadata['doc_type'] = doc_type
        doc.metadata['filename'] = os.path.basename(file_path)
    return documents

def create_vector_store_for_document(documents, doc_id):
    """Create vector store for a specific document."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Create separate vector store for this document
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=os.path.join(VECTOR_DB_DIR, doc_id)
    )
    return vectorstore

def load_vector_store(doc_id):
    """Load vector store for a document."""
    db_path = os.path.join(VECTOR_DB_DIR, doc_id)
    if os.path.exists(db_path) and os.listdir(db_path):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        return Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
    return None

def create_legal_explainer_chain(vectorstore):
    """Create chain for explaining legal documents."""
    llm = OpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE,
        model_name=MODEL_NAME
    )
    
    template = """You are a legal assistant helping people understand legal documents.
    Explain legal terms and clauses in plain, simple language. Highlight important points and potential risks.
    
    IMPORTANT DISCLAIMER: This is not legal advice. Users should consult with a qualified attorney for legal matters.

    Document Content:
    {context}

    Question: {question}

    Provide a clear answer that:
    1. Explains the legal concept in plain language
    2. Highlights important points or obligations
    3. Identifies potential risks or concerns (if any)
    4. References specific clauses or sections
    5. Suggests when professional legal advice might be needed

    Use simple language and avoid legal jargon. Be clear and helpful.

    Answer:"""
    
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": NUM_RETRIEVED_DOCS}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

def ask_legal_question(qa_chain, question):
    """Ask question about legal document."""
    try:
        result = qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": result.get("source_documents", [])
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }

def main():
    """Main function."""
    print("=" * 60)
    print("⚖️  Legal Document Navigator")
    print("=" * 60)
    print("\n⚠️  DISCLAIMER: This tool provides general information only.")
    print("   It is not a substitute for professional legal advice.\n")
    
    # Document selection
    print("Select a document to analyze:")
    print("1. Upload new document")
    print("2. Use existing document")
    
    choice = input("\nChoice (1/2): ").strip()
    
    if choice == "1":
        file_path = input("Enter document path: ").strip()
        if not os.path.exists(file_path):
            print("File not found!")
            return
        
        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\nLoading document: {doc_id}")
        
        documents = load_legal_document(file_path)
        if not documents:
            print("Error loading document!")
            return
        
        vectorstore = create_vector_store_for_document(documents, doc_id)
        print("✓ Document loaded and indexed")
    else:
        # List available documents
        if not os.path.exists(VECTOR_DB_DIR):
            print("No documents available!")
            return
        
        doc_ids = [d for d in os.listdir(VECTOR_DB_DIR) 
                   if os.path.isdir(os.path.join(VECTOR_DB_DIR, d))]
        if not doc_ids:
            print("No documents available!")
            return
        
        print("\nAvailable documents:")
        for i, doc_id in enumerate(doc_ids, 1):
            print(f"{i}. {doc_id}")
        
        doc_choice = int(input("\nSelect document number: ")) - 1
        doc_id = doc_ids[doc_choice]
        vectorstore = load_vector_store(doc_id)
    
    qa_chain = create_legal_explainer_chain(vectorstore)
    
    print("\n" + "=" * 60)
    print("System ready! Ask questions about the document")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        result = ask_legal_question(qa_chain, question)
        
        print("\n" + "=" * 60)
        print("📋 Explanation:")
        print("=" * 60)
        print(result["answer"])
        
        print("\n⚠️  Remember: This is not legal advice. Consult an attorney for legal matters.")
        
        if result["sources"]:
            print("\n📄 Referenced Sections:")
            for i, source in enumerate(result["sources"][:3], 1):
                page = source.metadata.get('page', 'N/A')
                print(f"{i}. Page {page}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
```

---

## Testing Your Application

### Test 1: Upload a Contract
1. Prepare a sample contract PDF
2. Run: `python main.py`
3. Upload the document
4. Ask questions about specific clauses

### Test 2: Legal Questions
Try:
- "What are my obligations?"
- "What happens if I breach?"
- "What are the payment terms?"

---

## Troubleshooting

### Problem: Legal explanations too complex
**Solution:** Adjust the prompt template to emphasize plain language and add examples.

### Problem: Missing important clauses
**Solution:** Reduce chunk size to capture more specific clauses, or increase retrieved documents.

---

## Next Steps & Enhancements

1. **Clause Highlighting**: Visually highlight important clauses
2. **Risk Scoring**: Score contracts for risk levels
3. **Comparison Tool**: Compare two contracts side-by-side
4. **Template Library**: Common contract templates
5. **Legal Term Dictionary**: Built-in dictionary of legal terms
6. **Export Reports**: Generate analysis reports
7. **Attorney Integration**: Connect with legal professionals

---

## Summary

You've built a legal document navigator that:
✅ **Plain Language** - Converts legal jargon to simple language  
✅ **Risk Identification** - Highlights important clauses  
✅ **Document-Specific** - Answers based on user's document  
✅ **Clause Extraction** - Identifies specific clauses  

**Key Difference:** This project focuses on making legal documents accessible to non-lawyers, with emphasis on plain language translation and risk identification.

Making legal documents accessible! ⚖️
