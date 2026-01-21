# Project 2: Interactive Textbook Q&A - Complete Implementation Guide

A beginner-friendly guide to building an interactive Q&A system for textbooks that helps students find specific information and engage with educational material.

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
An interactive Q&A system where students can ask questions about textbook content and receive answers grounded in the textbook's text, with proper citations to chapters and pages.

### Example Questions
- "Explain Newton's first law with an example."
- "What is photosynthesis and how does it work?"
- "Summarize Chapter 5 on World War II."
- "What are the main causes of climate change according to the textbook?"

### How It Works
1. Student asks a question about textbook content
2. System retrieves relevant sections from the textbook
3. System identifies chapter and page numbers
4. LLM provides explanation with examples from the text
5. Student receives answer with citations (chapter, page)

### Key Features
- **Citation Support**: Every answer includes chapter and page references
- **Context-Aware**: Maintains understanding of textbook structure
- **Multi-Chapter Retrieval**: Can pull information from multiple chapters
- **Student-Friendly**: Answers in clear, educational language

---

## Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of RAG concepts (see Project 1)
- Basic knowledge of document processing

### Required Software
- Python 3.8 or higher
- pip (Python package installer)

### Required Accounts
- OpenAI API account (or alternative LLM provider)

---

## Project Structure

```
textbook-qa/
│
├── textbooks/              # Store textbook PDFs here
│   ├── physics_textbook.pdf
│   ├── biology_textbook.pdf
│   └── history_textbook.pdf
│
├── vector_db/              # Vector database (created automatically)
│
├── .env                    # API keys
│
├── requirements.txt        # Python dependencies
│
├── config.py              # Configuration settings
│
├── main.py                # Main application code
│
└── README.md              # This file
```

---

## Step-by-Step Implementation

### Step 1: Set Up Project Environment

```bash
mkdir textbook-qa
cd textbook-qa
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
mkdir textbooks vector_db
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
tiktoken==0.5.2
```

Install:
```bash
pip install -r requirements.txt
```

### Step 3: Set Up API Keys

Create `.env`:
```
OPENAI_API_KEY=your_api_key_here
```

### Step 4: Create Configuration File

Create `config.py`:
```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEXTBOOKS_DIR = "textbooks"
VECTOR_DB_DIR = "vector_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
NUM_RETRIEVED_DOCS = 5  # More docs for better context
TEMPERATURE = 0.3  # Slightly higher for educational explanations
MODEL_NAME = "gpt-3.5-turbo"
```

### Step 5: Create Enhanced Document Loader with Metadata

Create `main.py`:
```python
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from config import *

def load_textbook_with_metadata(file_path, textbook_name):
    """
    Load textbook and add metadata (chapter, page, textbook name).
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Add metadata to each document
    for doc in documents:
        doc.metadata['textbook'] = textbook_name
        doc.metadata['page'] = doc.metadata.get('page', 0)
        # Try to extract chapter from page content (basic approach)
        # You can enhance this with more sophisticated chapter detection
        doc.metadata['chapter'] = extract_chapter_info(doc.page_content)
    
    return documents

def extract_chapter_info(content):
    """
    Simple chapter extraction - looks for chapter patterns in text.
    Enhance this based on your textbook format.
    """
    content_lower = content[:500].lower()  # Check first 500 chars
    if 'chapter' in content_lower:
        # Try to find chapter number
        import re
        match = re.search(r'chapter\s+(\d+)', content_lower)
        if match:
            return f"Chapter {match.group(1)}"
    return "Unknown Chapter"

def load_all_textbooks(directory):
    """Load all textbooks from directory."""
    all_documents = []
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist!")
        return all_documents
    
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in '{directory}'")
        return all_documents
    
    print(f"Found {len(pdf_files)} textbook(s)")
    
    for filename in pdf_files:
        file_path = os.path.join(directory, filename)
        textbook_name = os.path.splitext(filename)[0]
        print(f"Loading: {textbook_name}")
        
        try:
            docs = load_textbook_with_metadata(file_path, textbook_name)
            all_documents.extend(docs)
            print(f"  ✓ Loaded {len(docs)} pages")
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    print(f"\nTotal pages loaded: {len(all_documents)}")
    return all_documents

def split_documents_with_metadata(documents):
    """Split documents while preserving metadata."""
    print("\nSplitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Preserve metadata in chunks
    for chunk in chunks:
        if 'textbook' not in chunk.metadata:
            chunk.metadata['textbook'] = 'Unknown'
        if 'page' not in chunk.metadata:
            chunk.metadata['page'] = 0
        if 'chapter' not in chunk.metadata:
            chunk.metadata['chapter'] = 'Unknown Chapter'
    
    print(f"Created {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    """Create vector database with metadata."""
    print("\nCreating vector database...")
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found!")
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    
    print(f"✓ Vector database created")
    return vectorstore

def load_vector_store():
    """Load existing vector database."""
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        print("Loading existing vector database...")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
        print("✓ Database loaded")
        return vectorstore
    return None

def create_qa_chain(vectorstore):
    """Create Q&A chain with citation support."""
    print("\nSetting up Q&A chain...")
    
    llm = OpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE,
        model_name=MODEL_NAME
    )
    
    # Custom prompt for educational answers
    from langchain.prompts import PromptTemplate
    
    template = """You are a helpful educational assistant. Answer the student's question based on the textbook content provided. 
    Be clear, educational, and provide examples when possible. Always cite the source when referencing specific information.

    Context from textbook:
    {context}

    Question: {question}

    Answer (include citations):"""
    
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
    
    print("✓ Q&A chain ready")
    return qa_chain

def ask_question(qa_chain, question):
    """Ask question and get answer with citations."""
    print(f"\nQuestion: {question}")
    print("Searching textbook...")
    
    try:
        result = qa_chain({"query": question})
        answer = result["result"]
        sources = result.get("source_documents", [])
        
        # Format citations
        citations = []
        seen = set()
        for source in sources:
            textbook = source.metadata.get('textbook', 'Unknown')
            page = source.metadata.get('page', 'N/A')
            chapter = source.metadata.get('chapter', 'Unknown')
            
            citation_key = f"{textbook}_page_{page}"
            if citation_key not in seen:
                citations.append({
                    'textbook': textbook,
                    'chapter': chapter,
                    'page': page
                })
                seen.add(citation_key)
        
        return {
            "answer": answer,
            "sources": sources,
            "citations": citations
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "citations": []
        }

def main():
    """Main function."""
    print("=" * 60)
    print("📚 Interactive Textbook Q&A System")
    print("=" * 60)
    
    vectorstore = load_vector_store()
    
    if vectorstore is None:
        print("\nCreating new database...")
        documents = load_all_textbooks(TEXTBOOKS_DIR)
        
        if not documents:
            print("No textbooks found! Add PDF files to 'textbooks' folder.")
            return
        
        chunks = split_documents_with_metadata(documents)
        vectorstore = create_vector_store(chunks)
    
    qa_chain = create_qa_chain(vectorstore)
    
    print("\n" + "=" * 60)
    print("System ready! Ask questions about your textbooks")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        question = input("\n👤 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        if not question:
            continue
        
        result = ask_question(qa_chain, question)
        
        print("\n" + "-" * 60)
        print("📖 Answer:")
        print("-" * 60)
        print(result["answer"])
        
        if result["citations"]:
            print("\n" + "-" * 60)
            print("📚 Citations:")
            print("-" * 60)
            for i, citation in enumerate(result["citations"], 1):
                print(f"{i}. {citation['textbook']} - {citation['chapter']}, Page {citation['page']}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
```

### Step 6: Gather Textbooks

1. Download or obtain textbook PDFs
2. Place them in the `textbooks/` folder
3. Ensure PDFs are text-searchable (not just images)

---

## Testing Your Application

### Test 1: Basic Questions
```bash
python main.py
```

Try questions like:
- "What is Newton's first law?"
- "Explain photosynthesis"
- "What happened in World War II?"

### Test 2: Citation Verification
- Ask a specific question
- Verify that citations show correct textbook, chapter, and page
- Check that answers reference the textbook content

### Test 3: Multi-Chapter Questions
- Ask questions that might span multiple chapters
- Verify system retrieves relevant information from different sections

---

## Troubleshooting

### Problem: Citations show "Unknown Chapter"
**Solution:** Enhance the `extract_chapter_info()` function to match your textbook's format. Some textbooks use "Chapter X", others use "Unit X" or "Section X".

### Problem: Answers not citing sources properly
**Solution:** Check that metadata is being preserved through the splitting process. Verify the prompt template includes citation instructions.

### Problem: Can't find information across multiple textbooks
**Solution:** Increase `NUM_RETRIEVED_DOCS` in config.py to retrieve more chunks.

---

## Next Steps & Enhancements

### Basic Enhancements
1. **Better Chapter Detection**: Use regex patterns or ML to detect chapters more accurately
2. **Textbook Selection**: Allow students to select which textbook to query
3. **Study Notes Generator**: Generate study notes from specific chapters
4. **Quiz Generator**: Create quizzes based on textbook content

### Advanced Enhancements
1. **Multi-Modal Support**: Include diagrams and images from textbooks
2. **Note-Taking Integration**: Allow students to save answers as notes
3. **Progress Tracking**: Track which chapters students have studied
4. **Collaborative Features**: Share questions and answers with classmates
5. **Audio Support**: Read answers aloud for accessibility

### Web Interface (Streamlit)

Create `app.py`:
```python
import streamlit as st
from main import *

st.title("📚 Interactive Textbook Q&A")
st.markdown("Ask questions about your textbooks and get answers with citations")

if 'qa_chain' not in st.session_state:
    with st.spinner("Loading system..."):
        vectorstore = load_vector_store()
        if vectorstore:
            st.session_state.qa_chain = create_qa_chain(vectorstore)
            st.success("Ready!")
        else:
            st.error("No database found. Run main.py first.")

question = st.text_input("Ask a question:", placeholder="e.g., Explain Newton's first law")

if st.button("Get Answer") and question:
    if 'qa_chain' in st.session_state:
        result = ask_question(st.session_state.qa_chain, question)
        
        st.markdown("### Answer:")
        st.write(result["answer"])
        
        if result["citations"]:
            st.markdown("### Citations:")
            for citation in result["citations"]:
                st.write(f"📖 {citation['textbook']} - {citation['chapter']}, Page {citation['page']}")
    else:
        st.error("System not initialized.")
```

Run: `streamlit run app.py`

---

## Summary

You've built an interactive textbook Q&A system with:
✅ **Citation Support** - Every answer includes source references  
✅ **Metadata Preservation** - Chapter and page information maintained  
✅ **Educational Focus** - Answers tailored for students  
✅ **Multi-Textbook Support** - Query across multiple textbooks  

**Key Difference from Project 1:** This project focuses on maintaining and displaying citations, making it perfect for educational use where source attribution is important.

---

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Metadata Filtering](https://www.trychroma.com/)
- [Educational RAG Patterns](https://python.langchain.com/docs/use_cases/question_answering/)

Happy learning! 🎓
