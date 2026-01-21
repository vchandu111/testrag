# Project 9: Business Registration and Startup Guide - Complete Implementation Guide

A beginner-friendly guide to building a comprehensive RAG system guiding users through starting businesses, covering registration, licensing, and compliance.

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
A comprehensive RAG system guiding users through starting businesses, covering registration, licensing, and compliance for all business types. Provides personalized checklists and step-by-step guidance.

### Example Questions
- "What licenses do I need for a food truck in Hyderabad?"
- "How do I register a private limited company?"
- "What are the compliance requirements for a restaurant?"
- "What documents do I need for GST registration?"

### How It Works
1. User specifies business type and location
2. System retrieves relevant regulations and requirements
3. System generates personalized checklist
4. LLM provides step-by-step guidance
5. User receives complete registration roadmap

### Key Features
- **Location-Specific**: Different requirements by city/state
- **Business Type Specific**: Tailored for different business types
- **Checklist Generation**: Personalized checklists
- **Step-by-Step Guidance**: Clear instructions
- **Form Assistance**: Help filling registration forms

---

## Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of RAG concepts
- Basic knowledge of business registration

### Required Software
- Python 3.8 or higher
- pip (Python package installer)

### Required Accounts
- OpenAI API account (or alternative LLM provider)

---

## Project Structure

```
business-registration-guide/
│
├── documents/             # Registration documents
│   ├── registration/
│   ├── licensing/
│   ├── compliance/
│   └── location_specific/
│       ├── hyderabad/
│       ├── mumbai/
│       └── delhi/
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
mkdir business-registration-guide
cd business-registration-guide
python3 -m venv venv
source venv/bin/activate
mkdir -p documents/{registration,licensing,compliance,location_specific/{hyderabad,mumbai,delhi}} vector_db
```

### Step 2: Install Dependencies

Create `requirements.txt`:
```txt
langchain==0.1.0
openai==1.10.0
chromadb==0.4.22
sentence-transformers==2.3.1
pypdf2==3.0.1
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
VECTOR_DB_DIR = "vector_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
NUM_RETRIEVED_DOCS = 6  # More docs for comprehensive guidance
TEMPERATURE = 0.2
MODEL_NAME = "gpt-3.5-turbo"

# Business types
BUSINESS_TYPES = [
    'Food Truck', 'Restaurant', 'Retail Store', 'E-commerce',
    'Consulting', 'Manufacturing', 'IT Services', 'Healthcare',
    'Education', 'Real Estate', 'Other'
]

# Locations
LOCATIONS = [
    'Hyderabad', 'Mumbai', 'Delhi', 'Bangalore', 'Chennai',
    'Kolkata', 'Pune', 'Ahmedabad', 'Other'
]
```

### Step 4: Create Main Application

Create `main.py`:
```python
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from config import *

def load_business_documents(base_dir):
    """Load all business registration documents."""
    all_documents = []
    
    for category in os.listdir(base_dir):
        cat_path = os.path.join(base_dir, category)
        if not os.path.isdir(cat_path):
            continue
        
        print(f"Loading {category} documents...")
        for root, dirs, files in os.walk(cat_path):
            for filename in files:
                if filename.endswith('.pdf'):
                    file_path = os.path.join(root, filename)
                    try:
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata['category'] = category
                            # Extract location from path
                            if 'location_specific' in file_path:
                                location = os.path.basename(os.path.dirname(file_path))
                                doc.metadata['location'] = location
                            doc.metadata['source'] = filename
                        all_documents.extend(docs)
                        print(f"  ✓ Loaded {filename}")
                    except Exception as e:
                        print(f"  ✗ Error: {e}")
    
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

def create_registration_guide_chain(vectorstore, location=None):
    """Create chain for registration guidance."""
    llm = OpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE,
        model_name=MODEL_NAME
    )
    
    # Filter by location if specified
    search_kwargs = {"k": NUM_RETRIEVED_DOCS}
    if location:
        # Note: This is a simplified filter - enhance based on your metadata structure
        pass
    
    template = """You are a business registration advisor helping entrepreneurs start their businesses.
    Provide comprehensive, step-by-step guidance based on official documents and regulations.

    Registration Information:
    {context}

    Question: {question}
    Location: {location}
    Business Type: {business_type}

    Provide a detailed answer that includes:
    1. Required licenses and registrations (numbered list)
    2. Required documents for each registration
    3. Step-by-step application process
    4. Fees and costs involved
    5. Timeline for each step
    6. Important compliance requirements
    7. Where to apply (office locations/websites if available)
    8. Common mistakes to avoid

    Format your answer clearly with numbered steps and bullet points.
    Be specific to the location and business type mentioned.

    Answer:"""
    
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question", "location", "business_type"]
    )
    
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

def ask_registration_question(qa_chain, question, location, business_type):
    """Ask question about business registration."""
    try:
        # Enhance question with location and business type context
        enhanced_question = f"{question} Location: {location}. Business Type: {business_type}."
        
        result = qa_chain({
            "query": enhanced_question,
            "location": location,
            "business_type": business_type
        })
        
        return {
            "answer": result["result"],
            "sources": result.get("source_documents", [])
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }

def generate_checklist(vectorstore, location, business_type):
    """Generate personalized checklist."""
    question = f"Create a complete checklist for registering a {business_type} business in {location}"
    qa_chain = create_registration_guide_chain(vectorstore, location)
    result = ask_registration_question(qa_chain, question, location, business_type)
    return result["answer"]

def main():
    """Main function."""
    print("=" * 60)
    print("🚀 Business Registration and Startup Guide")
    print("=" * 60)
    
    vectorstore = load_vector_store()
    
    if vectorstore is None:
        print("\nCreating database from documents...")
        documents = load_business_documents(DOCUMENTS_DIR)
        if not documents:
            print("No documents found!")
            return
        vectorstore = create_vector_store(documents)
    
    print("\n" + "=" * 60)
    print("Welcome! Let's get your business started.")
    print("=" * 60)
    
    # Get user information
    print("\nPlease provide some information:")
    business_type = input(f"Business Type ({', '.join(BUSINESS_TYPES[:5])}...): ").strip()
    location = input(f"Location ({', '.join(LOCATIONS[:5])}...): ").strip()
    
    qa_chain = create_registration_guide_chain(vectorstore, location)
    
    print("\n" + "=" * 60)
    print("Options:")
    print("1. Ask a specific question")
    print("2. Generate complete checklist")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        choice = input("\nChoice (1/2/quit): ").strip().lower()
        
        if choice == 'quit':
            break
        elif choice == '2':
            print("\nGenerating checklist...")
            checklist = generate_checklist(vectorstore, location, business_type)
            print("\n" + "=" * 60)
            print("📋 Complete Registration Checklist:")
            print("=" * 60)
            print(checklist)
        elif choice == '1':
            question = input("\n❓ Your question: ").strip()
            if question:
                result = ask_registration_question(qa_chain, question, location, business_type)
                print("\n" + "=" * 60)
                print("💡 Answer:")
                print("=" * 60)
                print(result["answer"])
                
                if result["sources"]:
                    print("\n📄 Sources:")
                    for i, source in enumerate(result["sources"][:3], 1):
                        print(f"{i}. {source.metadata.get('source', 'Unknown')}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
```

---

## Testing Your Application

### Test 1: Add Registration Documents
1. Collect business registration PDFs
2. Organize by location and category
3. Run: `python main.py`

### Test 2: Registration Questions
Try:
- "What licenses do I need for a food truck?"
- "How do I register for GST?"
- "What are the compliance requirements?"

---

## Troubleshooting

### Problem: Location-specific info not accurate
**Solution:** Ensure documents are properly organized by location and metadata is correctly set.

### Problem: Checklist incomplete
**Solution:** Increase `NUM_RETRIEVED_DOCS` and enhance the prompt to be more comprehensive.

---

## Next Steps & Enhancements

1. **Form Filler**: Help fill registration forms automatically
2. **Fee Calculator**: Calculate total registration costs
3. **Timeline Tracker**: Track registration progress
4. **Document Generator**: Generate required documents
5. **Compliance Calendar**: Track compliance deadlines
6. **Expert Consultation**: Connect with business consultants
7. **Multi-State Support**: Support for multiple states/countries

---

## Summary

You've built a business registration guide that:
✅ **Location-Specific** - Tailored to specific locations  
✅ **Business Type Specific** - Customized for business types  
✅ **Checklist Generation** - Personalized checklists  
✅ **Step-by-Step Guidance** - Clear instructions  

**Key Difference:** This project focuses on structured guidance, checklists, and location-specific information, making it perfect for entrepreneurs starting businesses.

Empowering entrepreneurs! 🚀
