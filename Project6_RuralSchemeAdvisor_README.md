# Project 6: Rural Scheme Advisor - Complete Implementation Guide

A beginner-friendly guide to building a system that helps villagers access information about government schemes and application processes using RAG.

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
A system where users ask questions about government schemes and receive step-by-step guidance from official documents. Includes eligibility checking and multilingual support.

### Example Questions
- "How do I apply for PM-KISAN scheme?"
- "What documents do I need for housing scheme?"
- "Am I eligible for the pension scheme?"
- "What is the application process for the scholarship?"

### How It Works
1. User asks a question about a government scheme
2. System retrieves relevant information from scheme documents
3. System extracts eligibility criteria, benefits, and procedures
4. LLM generates step-by-step instructions
5. User receives clear, actionable guidance

### Key Features
- **Step-by-Step Instructions**: Clear, numbered steps for applications
- **Eligibility Checking**: Verify user eligibility for schemes
- **Multilingual Support**: Support for regional languages
- **Scheme Organization**: Organized by scheme type and category
- **Document Requirements**: Lists required documents clearly

---

## Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of RAG concepts
- Basic knowledge of translation APIs (for multilingual)

### Required Software
- Python 3.8 or higher
- pip (Python package installer)

### Required Accounts
- OpenAI API account (or alternative LLM provider)
- Google Translate API (optional, for translation)

---

## Project Structure

```
rural-scheme-advisor/
│
├── schemes/               # Government scheme documents
│   ├── agriculture/
│   ├── housing/
│   ├── education/
│   └── healthcare/
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
mkdir rural-scheme-advisor
cd rural-scheme-advisor
python3 -m venv venv
source venv/bin/activate
mkdir -p schemes/{agriculture,housing,education,healthcare} vector_db
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
SCHEMES_DIR = "schemes"
VECTOR_DB_DIR = "vector_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
NUM_RETRIEVED_DOCS = 5
TEMPERATURE = 0.2  # Lower for accurate instructions
MODEL_NAME = "gpt-3.5-turbo"

# Supported languages
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'te': 'Telugu',
    'ta': 'Tamil',
    'mr': 'Marathi',
    'kn': 'Kannada'
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
    """Detect language of input text."""
    try:
        detected = translator.detect(text)
        return detected.lang
    except:
        return 'en'

def translate_text(text, target_lang='en'):
    """Translate text to target language."""
    if target_lang == 'en':
        return text
    try:
        translated = translator.translate(text, dest=target_lang)
        return translated.text
    except:
        return text

def load_scheme_documents(base_dir):
    """Load all scheme documents."""
    all_documents = []
    
    for category in os.listdir(base_dir):
        cat_path = os.path.join(base_dir, category)
        if not os.path.isdir(cat_path):
            continue
        
        print(f"Loading {category} schemes...")
        for filename in os.listdir(cat_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(cat_path, filename)
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata['category'] = category
                        doc.metadata['scheme_name'] = os.path.splitext(filename)[0]
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

def create_step_by_step_chain(vectorstore):
    """Create chain for step-by-step instructions."""
    llm = OpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE,
        model_name=MODEL_NAME
    )
    
    template = """You are a helpful assistant providing guidance on government schemes.
    Based on the scheme documents, provide clear, step-by-step instructions.
    Format your answer as numbered steps. Include eligibility criteria, required documents, and application process.

    Scheme Information:
    {context}

    Question: {question}

    Provide a comprehensive answer with:
    1. Eligibility criteria (if applicable)
    2. Required documents
    3. Step-by-step application process (numbered)
    4. Important dates/deadlines (if mentioned)
    5. Contact information or helpline (if available)

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

def ask_question(qa_chain, question, user_language='en'):
    """Ask question and get step-by-step answer."""
    # Translate question to English for processing
    if user_language != 'en':
        question_en = translate_text(question, 'en')
    else:
        question_en = question
    
    try:
        result = qa_chain({"query": question_en})
        answer = result["result"]
        sources = result.get("source_documents", [])
        
        # Translate answer back to user's language
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
    print("🏛️  Rural Scheme Advisor")
    print("=" * 60)
    
    vectorstore = load_vector_store()
    
    if vectorstore is None:
        print("\nCreating database from scheme documents...")
        documents = load_scheme_documents(SCHEMES_DIR)
        if not documents:
            print("No scheme documents found!")
            return
        vectorstore = create_vector_store(documents)
    
    qa_chain = create_step_by_step_chain(vectorstore)
    
    print("\n" + "=" * 60)
    print("System ready! Ask questions about government schemes")
    print("Supported languages: English, Hindi, Telugu, Tamil, Marathi, Kannada")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        # Detect language
        lang = detect_language(question)
        lang_name = SUPPORTED_LANGUAGES.get(lang, 'English')
        print(f"Detected language: {lang_name}")
        
        result = ask_question(qa_chain, question, lang)
        
        print("\n" + "=" * 60)
        print("📋 Answer:")
        print("=" * 60)
        print(result["answer"])
        
        if result["sources"]:
            print("\n📄 Source Documents:")
            for i, source in enumerate(result["sources"][:3], 1):
                scheme = source.metadata.get('scheme_name', 'Unknown')
                category = source.metadata.get('category', 'Unknown')
                print(f"{i}. {scheme} ({category})")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
```

---

## Testing Your Application

### Test 1: Add Scheme Documents
1. Download government scheme PDFs
2. Organize by category in `schemes/` folder
3. Run: `python main.py`

### Test 2: Multilingual Questions
Try asking in different languages:
- English: "How do I apply for PM-KISAN?"
- Hindi: "PM-KISAN के लिए कैसे आवेदन करें?"

---

## Troubleshooting

### Problem: Translation not working
**Solution:** Google Translate API has rate limits. Consider using paid APIs or implementing caching.

### Problem: Steps not clear
**Solution:** Adjust the prompt template to emphasize numbered steps and clarity.

---

## Next Steps & Enhancements

1. **Eligibility Checker**: Interactive form to check eligibility
2. **Application Form Filler**: Help fill application forms
3. **SMS/WhatsApp Integration**: Send information via SMS/WhatsApp
4. **Voice Input**: Support voice queries in regional languages
5. **Offline Mode**: Download scheme info for offline access
6. **Status Tracking**: Track application status
7. **Notifications**: Notify about new schemes or deadlines

---

## Summary

You've built a rural scheme advisor that:
✅ **Step-by-Step Instructions** - Clear, numbered guidance  
✅ **Multilingual Support** - Multiple regional languages  
✅ **Scheme Organization** - Organized by category  
✅ **Eligibility Information** - Clear eligibility criteria  

**Key Difference:** This project focuses on accessibility, multilingual support, and step-by-step guidance for users who may not be tech-savvy.

Empowering rural communities! 🌾
