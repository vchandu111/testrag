# Project 4: Enterprise Document Assistant - Complete Implementation Guide

A beginner-friendly guide to building an internal Q&A system for enterprise documents with access control and multi-document support.

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
An internal Q&A system where employees can ask questions and get answers from company documents (HR policies, IT guides, SOPs, compliance documents). Includes basic access control and document organization by department.

### Example Questions
- "What is the paid time off policy?"
- "How do I reset my password?"
- "What are the safety procedures for the warehouse?"
- "What is the expense reimbursement process?"

### How It Works
1. Employee asks a question
2. System searches relevant company documents
3. System retrieves information based on document type/department
4. LLM provides answer with source citations
5. Employee receives accurate, company-specific answer

### Key Features
- **Multi-Document Support**: PDFs, Word docs, markdown files
- **Department Organization**: Documents organized by department
- **Access Control**: Basic user authentication and permissions
- **Citation Tracking**: Shows source documents
- **Search Filters**: Filter by department or document type

---

## Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of RAG concepts
- Basic knowledge of authentication

### Required Software
- Python 3.8 or higher
- pip (Python package installer)

### Required Accounts
- OpenAI API account (or alternative LLM provider)

---

## Project Structure

```
enterprise-doc-assistant/
│
├── documents/              # Company documents
│   ├── hr/                # HR documents
│   ├── it/                # IT documents
│   ├── compliance/        # Compliance documents
│   └── operations/        # Operations/SOPs
│
├── vector_db/             # Vector database
│
├── users.json             # User database (simple)
│
├── .env                   # API keys
│
├── requirements.txt       # Dependencies
│
├── config.py             # Configuration
│
├── auth.py               # Authentication module
│
├── main.py               # Main application
│
└── README.md             # This file
```

---

## Step-by-Step Implementation

### Step 1: Set Up Project

```bash
mkdir enterprise-doc-assistant
cd enterprise-doc-assistant
python3 -m venv venv
source venv/bin/activate
mkdir -p documents/{hr,it,compliance,operations} vector_db
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
bcrypt==4.1.2
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
VECTOR_DB_DIR = "vector_db"
USERS_FILE = "users.json"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
NUM_RETRIEVED_DOCS = 5
TEMPERATURE = 0
MODEL_NAME = "gpt-3.5-turbo"

# Department access mapping
DEPARTMENT_ACCESS = {
    "hr": ["hr", "compliance"],
    "it": ["it", "compliance"],
    "operations": ["operations", "compliance"],
    "admin": ["hr", "it", "compliance", "operations"]  # Full access
}
```

### Step 4: Create Authentication Module

Create `auth.py`:
```python
import json
import bcrypt
import os
from config import USERS_FILE

def load_users():
    """Load users from JSON file."""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file."""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    """Hash a password."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    """Verify a password."""
    return bcrypt.checkpw(password.encode(), hashed.encode())

def authenticate(username, password):
    """Authenticate a user."""
    users = load_users()
    if username in users:
        if verify_password(password, users[username]['password']):
            return users[username]
    return None

def create_user(username, password, department, email):
    """Create a new user."""
    users = load_users()
    if username in users:
        return False, "Username already exists"
    
    users[username] = {
        'password': hash_password(password),
        'department': department,
        'email': email
    }
    save_users(users)
    return True, "User created successfully"

# Initialize with admin user if file doesn't exist
if not os.path.exists(USERS_FILE):
    users = {
        'admin': {
            'password': hash_password('admin123'),
            'department': 'admin',
            'email': 'admin@company.com'
        }
    }
    save_users(users)
    print("Created default admin user: admin/admin123")
```

### Step 5: Create Document Loader

Create `main.py`:
```python
import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from config import *
from auth import authenticate, DEPARTMENT_ACCESS

def load_document(file_path, department):
    """Load a document based on its type."""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.txt') or file_path.endswith('.md'):
        loader = TextLoader(file_path)
    else:
        return None
    
    documents = loader.load()
    # Add department metadata
    for doc in documents:
        doc.metadata['department'] = department
        doc.metadata['source'] = os.path.basename(file_path)
    return documents

def load_all_documents(base_dir):
    """Load all documents from all departments."""
    all_documents = []
    
    for department in os.listdir(base_dir):
        dept_path = os.path.join(base_dir, department)
        if not os.path.isdir(dept_path):
            continue
        
        print(f"Loading {department} documents...")
        for filename in os.listdir(dept_path):
            file_path = os.path.join(dept_path, filename)
            if os.path.isfile(file_path):
                docs = load_document(file_path, department)
                if docs:
                    all_documents.extend(docs)
                    print(f"  ✓ Loaded {filename}")
    
    return all_documents

def create_vector_store(documents):
    """Create vector database."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    return vectorstore

def load_vector_store():
    """Load existing vector database."""
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        return Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
    return None

def create_qa_chain(vectorstore, user_department):
    """Create Q&A chain with department filtering."""
    llm = OpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE,
        model_name=MODEL_NAME
    )
    
    # Get accessible departments
    accessible_depts = DEPARTMENT_ACCESS.get(user_department, [])
    
    # Create retriever with metadata filter
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": NUM_RETRIEVED_DOCS,
            "filter": {"department": {"$in": accessible_depts}}
        }
    )
    
    template = """You are a helpful assistant answering questions about company documents.
    Provide accurate, concise answers based on the document content. Always cite your sources.

    Context from documents:
    {context}

    Question: {question}

    Answer:"""
    
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

def ask_question(qa_chain, question):
    """Ask a question and get answer."""
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
    print("🏢 Enterprise Document Assistant")
    print("=" * 60)
    
    # Login
    print("\nPlease login:")
    username = input("Username: ").strip()
    password = input("Password: ").strip()
    
    user = authenticate(username, password)
    if not user:
        print("❌ Authentication failed!")
        return
    
    print(f"✓ Logged in as {username} ({user['department']} department)")
    
    # Load vector store
    vectorstore = load_vector_store()
    if vectorstore is None:
        print("\nCreating database from documents...")
        documents = load_all_documents(DOCUMENTS_DIR)
        if not documents:
            print("No documents found!")
            return
        vectorstore = create_vector_store(documents)
    
    # Create Q&A chain
    qa_chain = create_qa_chain(vectorstore, user['department'])
    
    print("\n" + "=" * 60)
    print("System ready! Ask questions about company documents")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        result = ask_question(qa_chain, question)
        
        print("\n" + "-" * 60)
        print("💡 Answer:")
        print("-" * 60)
        print(result["answer"])
        
        if result["sources"]:
            print("\n📄 Sources:")
            for i, source in enumerate(result["sources"][:3], 1):
                dept = source.metadata.get('department', 'Unknown')
                doc = source.metadata.get('source', 'Unknown')
                print(f"{i}. {doc} ({dept} department)")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
```

---

## Testing Your Application

### Test 1: Set Up Test Documents
1. Add sample PDFs to `documents/hr/` (e.g., PTO policy)
2. Add sample docs to `documents/it/` (e.g., password reset guide)
3. Run: `python main.py`

### Test 2: Test Access Control
- Login as different users from different departments
- Verify users only see documents from their accessible departments

---

## Troubleshooting

### Problem: Can't access documents
**Solution:** Check `DEPARTMENT_ACCESS` mapping in config.py and ensure user's department is correctly set.

### Problem: Authentication not working
**Solution:** Verify bcrypt is installed correctly and users.json file has correct format.

---

## Next Steps & Enhancements

1. **Advanced Authentication**: Use JWT tokens, OAuth integration
2. **Document Versioning**: Track document versions and updates
3. **Audit Logging**: Log all queries and access
4. **Feedback System**: Allow users to rate answer quality
5. **Advanced Search**: Full-text search with filters
6. **Document Upload**: Web interface for uploading new documents
7. **Notifications**: Notify when documents are updated

---

## Summary

You've built an enterprise document assistant with:
✅ **Multi-Document Support** - PDFs, Word, markdown  
✅ **Department Organization** - Documents organized by department  
✅ **Access Control** - Basic authentication and permissions  
✅ **Citation Tracking** - Source documents shown  

**Key Difference:** This project focuses on security, organization, and enterprise use cases with multiple document types and access control.

Secure and efficient! 🔒
