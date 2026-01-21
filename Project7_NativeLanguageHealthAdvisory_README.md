# Project 7: Native Language Health Advisory System - Complete Implementation Guide

A beginner-friendly guide to building a multilingual RAG system providing health guidance in local languages from WHO and health ministry documents.

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
A multilingual RAG system providing health guidance, symptoms, prevention measures, and treatment information in local languages from WHO guidelines and health ministry documents.

### Example Questions
- "What are dengue symptoms?" (in native language)
- "How to prevent malaria?"
- "What should I do if I have fever?"
- "COVID-19 prevention measures"

### How It Works
1. User asks health question in their native language
2. System detects language and translates query
3. System retrieves relevant health information from WHO/ministry docs
4. LLM generates answer in user's language
5. User receives health guidance in their preferred language

### Key Features
- **Multilingual Support**: Multiple regional languages
- **Health Categories**: Organized by disease, symptoms, prevention
- **Crisis Response**: Quick access during pandemics/outbreaks
- **Mobile-Friendly**: Simple interface for mobile devices
- **Emergency Guidance**: When to seek immediate medical help

---

## Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of RAG concepts
- Basic knowledge of translation

### Required Software
- Python 3.8 or higher
- pip (Python package installer)

### Required Accounts
- OpenAI API account (or alternative LLM provider)
- Google Translate API (optional)

---

## Project Structure

```
health-advisory/
│
├── health_docs/           # Health documents
│   ├── who_guidelines/
│   ├── ministry_docs/
│   ├── disease_info/
│   └── prevention/
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
mkdir health-advisory
cd health-advisory
python3 -m venv venv
source venv/bin/activate
mkdir -p health_docs/{who_guidelines,ministry_docs,disease_info,prevention} vector_db
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
googletrans==4.0.0rc1
```

Install: `pip install -r requirements.txt`

### Step 3: Create Configuration

Create `config.py`:
```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HEALTH_DOCS_DIR = "health_docs"
VECTOR_DB_DIR = "vector_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
NUM_RETRIEVED_DOCS = 5
TEMPERATURE = 0.2  # Lower for accurate medical information
MODEL_NAME = "gpt-3.5-turbo"

# Supported languages
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'te': 'Telugu',
    'ta': 'Tamil',
    'mr': 'Marathi',
    'kn': 'Kannada',
    'bn': 'Bengali',
    'gu': 'Gujarati'
}

# Health categories
HEALTH_CATEGORIES = {
    'symptoms': 'Symptoms and Signs',
    'prevention': 'Prevention Measures',
    'treatment': 'Treatment Information',
    'emergency': 'Emergency Situations',
    'vaccination': 'Vaccination Information'
}
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
from googletrans import Translator
from config import *

translator = Translator()

def detect_language(text):
    """Detect language of input."""
    try:
        return translator.detect(text).lang
    except:
        return 'en'

def translate_text(text, target_lang='en'):
    """Translate text."""
    if target_lang == 'en':
        return text
    try:
        return translator.translate(text, dest=target_lang).text
    except:
        return text

def load_health_documents(base_dir):
    """Load all health documents."""
    all_documents = []
    
    for category in os.listdir(base_dir):
        cat_path = os.path.join(base_dir, category)
        if not os.path.isdir(cat_path):
            continue
        
        print(f"Loading {category} documents...")
        for filename in os.listdir(cat_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(cat_path, filename)
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata['category'] = category
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

def create_health_advisory_chain(vectorstore):
    """Create chain for health advisory."""
    llm = OpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE,
        model_name=MODEL_NAME
    )
    
    template = """You are a health advisory assistant providing information based on WHO guidelines and health ministry documents.
    Provide accurate, clear health information. Always include when to seek immediate medical help.

    Health Information:
    {context}

    Question: {question}

    Provide a comprehensive answer that includes:
    1. Clear explanation of the health topic
    2. Symptoms (if applicable)
    3. Prevention measures
    4. Treatment information (if applicable)
    5. IMPORTANT: When to seek immediate medical attention
    6. Additional resources or helpline numbers (if available)

    Use simple, clear language that is easy to understand.
    Always emphasize seeking professional medical help for serious symptoms.

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

def ask_health_question(qa_chain, question, user_language='en'):
    """Ask health question and get answer."""
    # Translate to English for processing
    if user_language != 'en':
        question_en = translate_text(question, 'en')
    else:
        question_en = question
    
    try:
        result = qa_chain({"query": question_en})
        answer = result["result"]
        sources = result.get("source_documents", [])
        
        # Translate answer back
        if user_language != 'en':
            answer = translate_text(answer, user_language)
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }

def main():
    """Main function."""
    print("=" * 60)
    print("🏥 Native Language Health Advisory System")
    print("=" * 60)
    
    vectorstore = load_vector_store()
    
    if vectorstore is None:
        print("\nCreating database from health documents...")
        documents = load_health_documents(HEALTH_DOCS_DIR)
        if not documents:
            print("No health documents found!")
            return
        vectorstore = create_vector_store(documents)
    
    qa_chain = create_health_advisory_chain(vectorstore)
    
    print("\n" + "=" * 60)
    print("System ready! Ask health questions in your language")
    print("Supported languages:", ", ".join(SUPPORTED_LANGUAGES.values()))
    print("Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        question = input("\n🏥 Your health question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        # Detect language
        lang = detect_language(question)
        lang_name = SUPPORTED_LANGUAGES.get(lang, 'English')
        print(f"Language: {lang_name}")
        
        result = ask_health_question(qa_chain, question, lang)
        
        print("\n" + "=" * 60)
        print("💊 Health Information:")
        print("=" * 60)
        print(result["answer"])
        
        # Emergency warning
        print("\n⚠️  REMEMBER: For serious symptoms or emergencies, seek immediate medical help!")
        
        if result["sources"]:
            print("\n📄 Sources:")
            for i, source in enumerate(result["sources"][:3], 1):
                source_name = source.metadata.get('source', 'Unknown')
                category = source.metadata.get('category', 'Unknown')
                print(f"{i}. {source_name} ({category})")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
```

---

## Testing Your Application

### Test 1: Add Health Documents
1. Download WHO guidelines and health ministry PDFs
2. Organize by category in `health_docs/` folder
3. Run: `python main.py`

### Test 2: Health Questions
Try in different languages:
- "What are dengue symptoms?"
- "How to prevent malaria?"
- "COVID-19 prevention"

---

## Troubleshooting

### Problem: Medical information accuracy
**Solution:** Always verify with official sources. Consider adding disclaimers and source verification.

### Problem: Translation quality
**Solution:** Use professional translation services for medical terms or create a medical terminology dictionary.

---

## Next Steps & Enhancements

1. **Symptom Checker**: Interactive symptom assessment
2. **Vaccination Tracker**: Track vaccination schedules
3. **Telemedicine Integration**: Connect with doctors
4. **Voice Input**: Voice queries in regional languages
5. **SMS Alerts**: Send health alerts via SMS
6. **Image Recognition**: Identify rashes/symptoms from images
7. **Emergency Contacts**: Quick access to local emergency numbers
8. **Health Calendar**: Vaccination and checkup reminders

---

## Summary

You've built a health advisory system that:
✅ **Multilingual Support** - Health info in local languages  
✅ **Health Categories** - Organized by disease/prevention  
✅ **Crisis Response** - Quick access during outbreaks  
✅ **Emergency Guidance** - Clear when to seek help  

**Key Difference:** This project focuses on accessibility, multilingual support, and public health information dissemination, especially important for rural populations.

Promoting public health! 🏥
