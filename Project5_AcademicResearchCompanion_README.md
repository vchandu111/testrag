# Project 5: Academic Research Companion - Complete Implementation Guide

A beginner-friendly guide to building a system that helps researchers extract and synthesize information from multiple academic papers using RAG.

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
A system where researchers ask questions about academic papers and receive synthesized answers that combine information from multiple research papers, with proper citations.

### Example Questions
- "What are recent trends in deep learning for medical imaging?"
- "Summarize the findings on climate change impacts from the last 5 years."
- "What methods are used for protein structure prediction?"
- "Compare approaches to natural language processing."

### How It Works
1. Researcher asks a question about research topics
2. System retrieves relevant sections from multiple papers
3. System extracts key findings, methodologies, and results
4. LLM synthesizes information from multiple sources
5. Researcher receives comprehensive answer with paper citations

### Key Features
- **Multi-Paper Synthesis**: Combines information from multiple papers
- **Citation Management**: Proper academic citations (authors, year, title)
- **Metadata Extraction**: Extracts authors, titles, abstracts, keywords
- **Trend Analysis**: Identifies patterns across papers
- **Literature Review Support**: Helps with comprehensive reviews

---

## Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of RAG concepts
- Basic knowledge of academic paper structure

### Required Software
- Python 3.8 or higher
- pip (Python package installer)

### Required Accounts
- OpenAI API account (or alternative LLM provider)

---

## Project Structure

```
research-companion/
│
├── papers/                # Store research papers (PDFs)
│   ├── medical_imaging/
│   ├── nlp/
│   └── climate/
│
├── vector_db/             # Vector database
│
├── metadata/              # Extracted metadata (JSON)
│
├── .env                   # API keys
│
├── requirements.txt       # Dependencies
│
├── config.py             # Configuration
│
├── metadata_extractor.py # Extract paper metadata
│
├── main.py               # Main application
│
└── README.md             # This file
```

---

## Step-by-Step Implementation

### Step 1: Set Up Project

```bash
mkdir research-companion
cd research-companion
python3 -m venv venv
source venv/bin/activate
mkdir -p papers/{medical_imaging,nlp,climate} vector_db metadata
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
pymupdf==1.23.8
```

Install: `pip install -r requirements.txt`

### Step 3: Create Configuration

Create `config.py`:
```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PAPERS_DIR = "papers"
VECTOR_DB_DIR = "vector_db"
METADATA_DIR = "metadata"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
NUM_RETRIEVED_DOCS = 8  # More papers for synthesis
TEMPERATURE = 0.3
MODEL_NAME = "gpt-3.5-turbo"
```

### Step 4: Create Metadata Extractor

Create `metadata_extractor.py`:
```python
import re
import json
import os
from pypdf import PdfReader
from config import METADATA_DIR

def extract_metadata(pdf_path):
    """Extract metadata from PDF."""
    try:
        reader = PdfReader(pdf_path)
        metadata = reader.metadata
        
        # Extract from first page (usually contains title, authors)
        first_page = reader.pages[0].extract_text()
        
        # Try to extract title (usually first few lines)
        lines = first_page.split('\n')[:10]
        title = ""
        authors = ""
        
        # Simple extraction (enhance based on paper format)
        for i, line in enumerate(lines):
            if len(line) > 20 and not title:
                title = line.strip()
            if 'author' in line.lower() or '@' in line:
                authors = line.strip()
        
        return {
            'title': title or metadata.get('/Title', 'Unknown'),
            'authors': authors or metadata.get('/Author', 'Unknown'),
            'year': extract_year(first_page),
            'filename': os.path.basename(pdf_path)
        }
    except Exception as e:
        return {
            'title': os.path.basename(pdf_path),
            'authors': 'Unknown',
            'year': 'Unknown',
            'filename': os.path.basename(pdf_path)
        }

def extract_year(text):
    """Extract publication year."""
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    if years:
        return years[0]
    return 'Unknown'

def save_metadata(filename, metadata):
    """Save metadata to JSON file."""
    os.makedirs(METADATA_DIR, exist_ok=True)
    json_path = os.path.join(METADATA_DIR, f"{os.path.splitext(filename)[0]}.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
```

### Step 5: Create Main Application

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
from metadata_extractor import extract_metadata, save_metadata
from config import *
import json

def load_paper_with_metadata(file_path, category):
    """Load paper and extract metadata."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Extract metadata
    metadata = extract_metadata(file_path)
    metadata['category'] = category
    save_metadata(metadata['filename'], metadata)
    
    # Add metadata to documents
    for doc in documents:
        doc.metadata.update(metadata)
    
    return documents

def load_all_papers(base_dir):
    """Load all papers from all categories."""
    all_documents = []
    
    for category in os.listdir(base_dir):
        cat_path = os.path.join(base_dir, category)
        if not os.path.isdir(cat_path):
            continue
        
        print(f"Loading {category} papers...")
        for filename in os.listdir(cat_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(cat_path, filename)
                try:
                    docs = load_paper_with_metadata(file_path, category)
                    all_documents.extend(docs)
                    print(f"  ✓ Loaded {filename}")
                except Exception as e:
                    print(f"  ✗ Error loading {filename}: {e}")
    
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

def load_metadata(filename):
    """Load metadata for a paper."""
    json_path = os.path.join(METADATA_DIR, f"{os.path.splitext(filename)[0]}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return None

def create_synthesis_chain(vectorstore):
    """Create chain for synthesizing multiple papers."""
    llm = OpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE,
        model_name=MODEL_NAME
    )
    
    template = """You are a research assistant helping synthesize information from multiple academic papers.
    Analyze the following content from various research papers and provide a comprehensive answer.
    When referencing specific findings, cite the paper using the format: (Author, Year) or (Title, Year).

    Content from papers:
    {context}

    Research Question: {question}

    Provide a synthesized answer that:
    1. Summarizes key findings across papers
    2. Identifies common themes and trends
    3. Highlights differences or contradictions
    4. Cites specific papers when referencing findings
    5. Provides a comprehensive overview

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

def ask_question(qa_chain, question):
    """Ask question and get synthesized answer."""
    try:
        result = qa_chain({"query": question})
        answer = result["result"]
        sources = result.get("source_documents", [])
        
        # Get unique papers cited
        cited_papers = {}
        for source in sources:
            filename = source.metadata.get('filename', 'Unknown')
            if filename not in cited_papers:
                metadata = load_metadata(filename)
                if metadata:
                    cited_papers[filename] = metadata
        
        return {
            "answer": answer,
            "sources": sources,
            "cited_papers": cited_papers
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "cited_papers": {}
        }

def main():
    """Main function."""
    print("=" * 60)
    print("🔬 Academic Research Companion")
    print("=" * 60)
    
    vectorstore = load_vector_store()
    
    if vectorstore is None:
        print("\nCreating database from papers...")
        documents = load_all_papers(PAPERS_DIR)
        if not documents:
            print("No papers found! Add PDFs to 'papers' folder.")
            return
        vectorstore = create_vector_store(documents)
    
    qa_chain = create_synthesis_chain(vectorstore)
    
    print("\n" + "=" * 60)
    print("System ready! Ask research questions")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        question = input("\n🔬 Research question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        print("\nAnalyzing papers...")
        result = ask_question(qa_chain, question)
        
        print("\n" + "=" * 60)
        print("📊 Synthesized Answer:")
        print("=" * 60)
        print(result["answer"])
        
        if result["cited_papers"]:
            print("\n" + "=" * 60)
            print("📚 Papers Referenced:")
            print("=" * 60)
            for filename, metadata in result["cited_papers"].items():
                print(f"\n• {metadata.get('title', filename)}")
                print(f"  Authors: {metadata.get('authors', 'Unknown')}")
                print(f"  Year: {metadata.get('year', 'Unknown')}")
                print(f"  Category: {metadata.get('category', 'Unknown')}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
```

---

## Testing Your Application

### Test 1: Add Research Papers
1. Download papers from arXiv, PubMed, or other sources
2. Organize by category in `papers/` folder
3. Run: `python main.py`

### Test 2: Research Questions
Try:
- "What are recent advances in medical imaging?"
- "Summarize findings on climate change"
- "Compare NLP approaches"

---

## Troubleshooting

### Problem: Metadata extraction not working
**Solution:** Papers have different formats. Enhance `extract_metadata()` to handle various formats or use specialized libraries like `grobid` for better extraction.

### Problem: Synthesis not comprehensive
**Solution:** Increase `NUM_RETRIEVED_DOCS` to include more papers in synthesis.

---

## Next Steps & Enhancements

1. **Better Metadata Extraction**: Use specialized tools like GROBID or ScienceParse
2. **Citation Formatting**: Support different citation styles (APA, MLA, Chicago)
3. **Paper Recommendations**: Suggest related papers based on query
4. **Visualization**: Create graphs showing paper relationships
5. **Export**: Export synthesized answers as literature reviews
6. **Paper Upload**: Web interface for uploading new papers
7. **Annotation**: Highlight and annotate important sections

---

## Summary

You've built an academic research companion that:
✅ **Multi-Paper Synthesis** - Combines information from multiple papers  
✅ **Citation Management** - Proper academic citations  
✅ **Metadata Extraction** - Extracts paper information  
✅ **Trend Analysis** - Identifies patterns across papers  

**Key Difference:** This project focuses on synthesizing information from multiple sources and academic citation, perfect for literature reviews and research.

Happy researching! 🔬
