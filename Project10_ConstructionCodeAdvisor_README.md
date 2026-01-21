# Project 10: Construction and Building Code Advisor - Complete Implementation Guide

A beginner-friendly guide to building a RAG system providing guidance on building codes, permits, construction standards, and approval processes.

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
A RAG system providing guidance on building codes, permits, construction standards, and approval processes from local and national regulations. Helps property owners and contractors navigate complex building requirements.

### Example Questions
- "What approvals do I need to add a second floor in Bangalore?"
- "What are the setback requirements for residential buildings?"
- "What is the maximum height allowed for my building?"
- "What documents do I need for a building permit?"

### How It Works
1. User asks about building codes, permits, or construction requirements
2. System retrieves relevant sections from building codes and regulations
3. System extracts specific requirements, restrictions, and procedures
4. LLM explains requirements in clear language
5. User receives comprehensive guidance with code references

### Key Features
- **Location-Specific Codes**: Different codes by city/region
- **Code References**: Cites specific code sections
- **Visual Explanations**: Includes diagrams and visual guides
- **Permit Guidance**: Step-by-step permit application process
- **Compliance Checking**: Verify if plans meet code requirements

---

## Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of RAG concepts
- Basic knowledge of construction/building codes

### Required Software
- Python 3.8 or higher
- pip (Python package installer)

### Required Accounts
- OpenAI API account (or alternative LLM provider)

---

## Project Structure

```
construction-code-advisor/
│
├── building_codes/        # Building code documents
│   ├── national/
│   ├── bangalore/
│   ├── mumbai/
│   └── delhi/
│
├── permits/               # Permit application guides
│
├── standards/              # Construction standards
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
mkdir construction-code-advisor
cd construction-code-advisor
python3 -m venv venv
source venv/bin/activate
mkdir -p building_codes/{national,bangalore,mumbai,delhi} permits standards vector_db
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
BUILDING_CODES_DIR = "building_codes"
PERMITS_DIR = "permits"
STANDARDS_DIR = "standards"
VECTOR_DB_DIR = "vector_db"
CHUNK_SIZE = 1200  # Larger chunks for code sections
CHUNK_OVERLAP = 300
NUM_RETRIEVED_DOCS = 6
TEMPERATURE = 0.1  # Very low for accurate code information
MODEL_NAME = "gpt-3.5-turbo"

# Locations
LOCATIONS = [
    'Bangalore', 'Mumbai', 'Delhi', 'Chennai', 'Hyderabad',
    'Pune', 'Kolkata', 'National'
]

# Construction types
CONSTRUCTION_TYPES = [
    'Residential', 'Commercial', 'Industrial', 'Mixed Use',
    'Renovation', 'Extension', 'New Construction'
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

def load_building_documents(base_dirs):
    """Load all building code documents."""
    all_documents = []
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
        
        print(f"Loading documents from {base_dir}...")
        for root, dirs, files in os.walk(base_dir):
            for filename in files:
                if filename.endswith('.pdf'):
                    file_path = os.path.join(root, filename)
                    try:
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        for doc in docs:
                            # Extract location from path
                            if 'building_codes' in file_path:
                                if 'national' in file_path:
                                    doc.metadata['location'] = 'National'
                                else:
                                    location = os.path.basename(os.path.dirname(file_path))
                                    doc.metadata['location'] = location
                                doc.metadata['category'] = 'building_codes'
                            elif 'permits' in file_path:
                                doc.metadata['category'] = 'permits'
                            elif 'standards' in file_path:
                                doc.metadata['category'] = 'standards'
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

def create_code_advisor_chain(vectorstore, location=None):
    """Create chain for building code advice."""
    llm = OpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE,
        model_name=MODEL_NAME
    )
    
    template = """You are a building code advisor helping property owners and contractors understand building codes and regulations.
    Provide accurate, detailed information based on official building codes and regulations.
    Always cite specific code sections when referencing requirements.

    Building Code Information:
    {context}

    Question: {question}
    Location: {location}
    Construction Type: {construction_type}

    Provide a comprehensive answer that includes:
    1. Specific code requirements and restrictions
    2. Code section references (if available)
    3. Required approvals and permits
    4. Setback requirements (if applicable)
    5. Height restrictions (if applicable)
    6. Required documents
    7. Application process and timeline
    8. Important compliance notes
    9. Common violations to avoid

    Format clearly with numbered points and code references.
    Be specific to the location and construction type mentioned.
    Use clear, professional language but avoid excessive jargon.

    Answer:"""
    
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question", "location", "construction_type"]
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

def ask_code_question(qa_chain, question, location, construction_type):
    """Ask question about building codes."""
    try:
        enhanced_question = f"{question} Location: {location}. Construction Type: {construction_type}."
        
        result = qa_chain({
            "query": enhanced_question,
            "location": location,
            "construction_type": construction_type
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

def main():
    """Main function."""
    print("=" * 60)
    print("🏗️  Construction and Building Code Advisor")
    print("=" * 60)
    print("\n⚠️  DISCLAIMER: This tool provides general guidance only.")
    print("   Always verify with local authorities and consult professionals.\n")
    
    vectorstore = load_vector_store()
    
    if vectorstore is None:
        print("Creating database from building codes...")
        documents = load_building_documents([
            BUILDING_CODES_DIR,
            PERMITS_DIR,
            STANDARDS_DIR
        ])
        if not documents:
            print("No documents found!")
            return
        vectorstore = create_vector_store(documents)
    
    print("\n" + "=" * 60)
    print("Welcome! Let's navigate building codes and permits.")
    print("=" * 60)
    
    # Get user information
    print("\nPlease provide some information:")
    location = input(f"Location ({', '.join(LOCATIONS[:5])}...): ").strip()
    construction_type = input(f"Construction Type ({', '.join(CONSTRUCTION_TYPES[:3])}...): ").strip()
    
    qa_chain = create_code_advisor_chain(vectorstore, location)
    
    print("\n" + "=" * 60)
    print("System ready! Ask questions about building codes and permits")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        result = ask_code_question(qa_chain, question, location, construction_type)
        
        print("\n" + "=" * 60)
        print("📋 Answer:")
        print("=" * 60)
        print(result["answer"])
        
        print("\n⚠️  Important: Always verify with local building authorities!")
        
        if result["sources"]:
            print("\n📄 Sources:")
            for i, source in enumerate(result["sources"][:3], 1):
                source_name = source.metadata.get('source', 'Unknown')
                code_location = source.metadata.get('location', 'Unknown')
                print(f"{i}. {source_name} ({code_location})")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
```

---

## Testing Your Application

### Test 1: Add Building Code Documents
1. Collect building code PDFs and permit guides
2. Organize by location in `building_codes/` folder
3. Run: `python main.py`

### Test 2: Code Questions
Try:
- "What are the setback requirements?"
- "What approvals do I need for a second floor?"
- "What is the maximum building height?"

---

## Troubleshooting

### Problem: Code references not accurate
**Solution:** Ensure documents include code section numbers. Enhance metadata extraction to capture section numbers.

### Problem: Location-specific info missing
**Solution:** Verify documents are properly organized by location and metadata is set correctly.

---

## Next Steps & Enhancements

1. **Visual Diagrams**: Include building code diagrams and illustrations
2. **Permit Tracker**: Track permit application status
3. **Code Calculator**: Calculate setbacks, heights, FSI automatically
4. **Plan Reviewer**: Upload building plans for code compliance check
5. **Inspector Integration**: Connect with building inspectors
6. **Violation Checker**: Check for potential code violations
7. **Cost Estimator**: Estimate permit and compliance costs
8. **Timeline Planner**: Plan construction timeline with approvals

---

## Summary

You've built a construction code advisor that:
✅ **Location-Specific Codes** - Tailored to specific locations  
✅ **Code References** - Cites specific code sections  
✅ **Permit Guidance** - Step-by-step permit process  
✅ **Compliance Checking** - Verify code compliance  

**Key Difference:** This project focuses on technical building codes, permits, and compliance, with emphasis on accuracy and code references.

Building safely and legally! 🏗️
