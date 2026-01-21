# Project 1: Agri Crop Management Q&A - Complete Implementation Guide

A beginner-friendly guide to building a Q&A system for agricultural crop management and pest control using RAG (Retrieval-Augmented Generation).

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
A question-answering system that allows farmers to ask natural language questions about crop management, pest control, and agricultural practices. The system retrieves relevant information from agricultural documents and provides concise, accurate answers.

### Example Questions
- "What are common cotton pests and how to manage them?"
- "How do I prevent fungal diseases in tomatoes?"
- "What is the best time to plant wheat?"
- "How much water does rice need during different growth stages?"

### How It Works
1. User asks a question in natural language
2. System searches through agricultural documents to find relevant information
3. System retrieves the most relevant document chunks
4. LLM generates a comprehensive answer based on retrieved information
5. User receives a clear, actionable answer

---

## Prerequisites

### Required Knowledge
- Basic Python programming (variables, functions, loops)
- Basic understanding of file operations
- Familiarity with command line/terminal

### Required Software
- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for version control)

### Required Accounts
- OpenAI API account (or alternative LLM provider)
  - Sign up at: https://platform.openai.com/
  - Get your API key from: https://platform.openai.com/api-keys

---

## Project Structure

Create the following folder structure:

```
agri-crop-qa/
│
├── documents/              # Store your PDF documents here
│   ├── crop_management.pdf
│   ├── pest_control.pdf
│   └── agricultural_guide.pdf
│
├── agri_db/               # Vector database (created automatically)
│
├── .env                   # API keys (create this file)
│
├── requirements.txt       # Python dependencies
│
├── main.py               # Main application code
│
├── config.py            # Configuration settings
│
└── README.md            # This file
```

---

## Step-by-Step Implementation

### Step 1: Set Up Your Project Environment

#### 1.1 Create Project Directory
```bash
mkdir agri-crop-qa
cd agri-crop-qa
```

#### 1.2 Create Virtual Environment
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your terminal prompt.

#### 1.3 Create Required Folders
```bash
mkdir documents
mkdir agri_db
```

---

### Step 2: Install Dependencies

#### 2.1 Create requirements.txt
Create a file named `requirements.txt` with the following content:

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

#### 2.2 Install Packages
```bash
pip install -r requirements.txt
```

**Note:** If you encounter errors, try installing packages one by one:
```bash
pip install langchain openai chromadb sentence-transformers pypdf2 streamlit python-dotenv tiktoken
```

---

### Step 3: Set Up API Keys

#### 3.1 Create .env File
Create a file named `.env` in your project root directory:

```bash
touch .env  # On macOS/Linux
# Or create manually on Windows
```

#### 3.2 Add Your API Key
Open `.env` and add:
```
OPENAI_API_KEY=your_actual_api_key_here
```

**Important:** 
- Replace `your_actual_api_key_here` with your actual OpenAI API key
- Never commit `.env` to version control (add it to `.gitignore`)

#### 3.3 Create .gitignore (Optional but Recommended)
Create `.gitignore`:
```
venv/
.env
agri_db/
__pycache__/
*.pyc
```

---

### Step 4: Gather Agricultural Documents

#### 4.1 Find Documents
You need PDF documents about agriculture. Here are some sources:
- Government agricultural extension websites
- University agricultural research papers
- Agricultural best practices guides
- Crop management manuals

**Example sources:**
- USDA Extension Services
- FAO (Food and Agriculture Organization) documents
- Local agricultural department websites

#### 4.2 Download and Place Documents
1. Download 3-5 PDF documents about crop management, pest control, or agricultural practices
2. Save them in the `documents/` folder
3. Name them clearly (e.g., `cotton_pest_control.pdf`, `tomato_growing_guide.pdf`)

**For testing purposes:** You can create simple text files converted to PDF, or use any agricultural PDFs you have access to.

---

### Step 5: Create Configuration File

#### 5.1 Create config.py
Create a file named `config.py`:

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Document Configuration
DOCUMENTS_DIR = "documents"
VECTOR_DB_DIR = "agri_db"

# Chunking Configuration
CHUNK_SIZE = 1000  # Number of characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# Retrieval Configuration
NUM_RETRIEVED_DOCS = 3  # Number of document chunks to retrieve

# LLM Configuration
TEMPERATURE = 0  # Lower temperature = more focused answers
MODEL_NAME = "gpt-3.5-turbo"  # or "gpt-4" for better quality
```

---

### Step 6: Create Document Loading Function

#### 6.1 Create main.py - Part 1
Create a file named `main.py` and start with imports and document loading:

```python
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from config import (
    DOCUMENTS_DIR,
    VECTOR_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    NUM_RETRIEVED_DOCS,
    TEMPERATURE,
    MODEL_NAME,
    OPENAI_API_KEY
)

def load_documents(directory):
    """
    Load all PDF documents from the specified directory.
    
    Args:
        directory: Path to directory containing PDF files
    
    Returns:
        List of loaded documents
    """
    documents = []
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist!")
        return documents
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in '{directory}'")
        return documents
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    # Load each PDF
    for filename in pdf_files:
        file_path = os.path.join(directory, filename)
        print(f"Loading: {filename}")
        try:
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
            print(f"  ✓ Loaded {len(loaded_docs)} pages")
        except Exception as e:
            print(f"  ✗ Error loading {filename}: {str(e)}")
    
    print(f"\nTotal documents loaded: {len(documents)} pages")
    return documents
```

**Test this step:**
```python
# Add at the end of main.py temporarily
if __name__ == "__main__":
    docs = load_documents(DOCUMENTS_DIR)
    print(f"Loaded {len(docs)} document pages")
```

Run: `python main.py` - You should see your PDFs being loaded.

---

### Step 7: Split Documents into Chunks

#### 7.1 Add Chunking Function to main.py
Add this function after the `load_documents` function:

```python
def split_documents(documents):
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of document objects
    
    Returns:
        List of document chunks
    """
    print("\nSplitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    print(f"Average chunk size: {CHUNK_SIZE} characters")
    
    return chunks
```

**Why chunking?**
- Large documents are split into smaller pieces
- Makes it easier to find relevant information
- Improves retrieval accuracy

---

### Step 8: Create Vector Database

#### 8.1 Add Vector Store Creation Function
Add this function:

```python
def create_vector_store(chunks):
    """
    Create a vector database from document chunks.
    
    Args:
        chunks: List of document chunks
    
    Returns:
        Vector store object
    """
    print("\nCreating vector database...")
    
    # Check if API key is set
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in .env file!")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    
    print(f"✓ Vector database created in '{VECTOR_DB_DIR}'")
    return vectorstore
```

**What is a vector database?**
- Stores document chunks as numerical vectors (embeddings)
- Allows semantic search (finding similar meaning, not just keywords)
- Enables fast retrieval of relevant information

---

### Step 9: Load Existing Vector Store (Optional)

#### 9.1 Add Function to Load Existing Database
Add this function to avoid recreating the database every time:

```python
def load_vector_store():
    """
    Load existing vector database if it exists.
    
    Returns:
        Vector store object or None if not found
    """
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        print(f"Loading existing vector database from '{VECTOR_DB_DIR}'...")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
        print("✓ Vector database loaded")
        return vectorstore
    return None
```

---

### Step 10: Set Up Q&A Chain

#### 10.1 Add Q&A Chain Creation Function
Add this function:

```python
def create_qa_chain(vectorstore):
    """
    Create a question-answering chain.
    
    Args:
        vectorstore: Vector database object
    
    Returns:
        Q&A chain object
    """
    print("\nSetting up Q&A chain...")
    
    # Initialize LLM
    llm = OpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE,
        model_name=MODEL_NAME
    )
    
    # Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Simple chain type for beginners
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": NUM_RETRIEVED_DOCS}
        ),
        return_source_documents=True  # Return source documents
    )
    
    print("✓ Q&A chain ready")
    return qa_chain
```

**What is a Q&A chain?**
- Combines retrieval (finding relevant info) with generation (creating answer)
- Uses LLM to generate answers based on retrieved documents
- Ensures answers are grounded in your documents

---

### Step 11: Create Query Function

#### 11.1 Add Query Function
Add this function:

```python
def ask_question(qa_chain, question):
    """
    Ask a question and get an answer.
    
    Args:
        qa_chain: Q&A chain object
        question: User's question as string
    
    Returns:
        Dictionary with answer and source documents
    """
    print(f"\nQuestion: {question}")
    print("Searching documents...")
    
    try:
        result = qa_chain({"query": question})
        answer = result["result"]
        sources = result.get("source_documents", [])
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }
```

---

### Step 12: Create Main Function

#### 12.1 Add Main Function
Add this function to tie everything together:

```python
def main():
    """
    Main function to initialize and run the Q&A system.
    """
    print("=" * 60)
    print("🌾 Agri Crop Management Q&A System")
    print("=" * 60)
    
    # Try to load existing vector store
    vectorstore = load_vector_store()
    
    # If not found, create new one
    if vectorstore is None:
        print("\nNo existing database found. Creating new one...")
        
        # Load documents
        documents = load_documents(DOCUMENTS_DIR)
        
        if not documents:
            print("No documents found! Please add PDF files to the 'documents' folder.")
            return
        
        # Split into chunks
        chunks = split_documents(documents)
        
        # Create vector store
        vectorstore = create_vector_store(chunks)
    
    # Create Q&A chain
    qa_chain = create_qa_chain(vectorstore)
    
    # Interactive Q&A loop
    print("\n" + "=" * 60)
    print("System ready! Ask your questions (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        question = input("\n👤 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        # Get answer
        result = ask_question(qa_chain, question)
        
        # Display answer
        print("\n" + "-" * 60)
        print("🤖 Answer:")
        print("-" * 60)
        print(result["answer"])
        
        # Display sources
        if result["sources"]:
            print("\n" + "-" * 60)
            print("📚 Sources:")
            print("-" * 60)
            for i, source in enumerate(result["sources"][:3], 1):
                page = source.metadata.get('page', 'N/A')
                source_name = source.metadata.get('source', 'Unknown')
                print(f"{i}. Page {page} from {os.path.basename(source_name)}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
```

---

### Step 13: Complete main.py File

Your complete `main.py` should look like this (combining all the functions above):

```python
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from config import (
    DOCUMENTS_DIR,
    VECTOR_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    NUM_RETRIEVED_DOCS,
    TEMPERATURE,
    MODEL_NAME,
    OPENAI_API_KEY
)

def load_documents(directory):
    """Load all PDF documents from the specified directory."""
    documents = []
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist!")
        return documents
    
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{directory}'")
        return documents
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    for filename in pdf_files:
        file_path = os.path.join(directory, filename)
        print(f"Loading: {filename}")
        try:
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
            print(f"  ✓ Loaded {len(loaded_docs)} pages")
        except Exception as e:
            print(f"  ✗ Error loading {filename}: {str(e)}")
    
    print(f"\nTotal documents loaded: {len(documents)} pages")
    return documents

def split_documents(documents):
    """Split documents into smaller chunks for better retrieval."""
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    """Create a vector database from document chunks."""
    print("\nCreating vector database...")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in .env file!")
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    print(f"✓ Vector database created in '{VECTOR_DB_DIR}'")
    return vectorstore

def load_vector_store():
    """Load existing vector database if it exists."""
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        print(f"Loading existing vector database from '{VECTOR_DB_DIR}'...")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
        print("✓ Vector database loaded")
        return vectorstore
    return None

def create_qa_chain(vectorstore):
    """Create a question-answering chain."""
    print("\nSetting up Q&A chain...")
    llm = OpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE,
        model_name=MODEL_NAME
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": NUM_RETRIEVED_DOCS}
        ),
        return_source_documents=True
    )
    print("✓ Q&A chain ready")
    return qa_chain

def ask_question(qa_chain, question):
    """Ask a question and get an answer."""
    print(f"\nQuestion: {question}")
    print("Searching documents...")
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
    """Main function to initialize and run the Q&A system."""
    print("=" * 60)
    print("🌾 Agri Crop Management Q&A System")
    print("=" * 60)
    
    vectorstore = load_vector_store()
    if vectorstore is None:
        print("\nNo existing database found. Creating new one...")
        documents = load_documents(DOCUMENTS_DIR)
        if not documents:
            print("No documents found! Please add PDF files to the 'documents' folder.")
            return
        chunks = split_documents(documents)
        vectorstore = create_vector_store(chunks)
    
    qa_chain = create_qa_chain(vectorstore)
    
    print("\n" + "=" * 60)
    print("System ready! Ask your questions (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        question = input("\n👤 Your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        if not question:
            print("Please enter a question.")
            continue
        
        result = ask_question(qa_chain, question)
        print("\n" + "-" * 60)
        print("🤖 Answer:")
        print("-" * 60)
        print(result["answer"])
        
        if result["sources"]:
            print("\n" + "-" * 60)
            print("📚 Sources:")
            print("-" * 60)
            for i, source in enumerate(result["sources"][:3], 1):
                page = source.metadata.get('page', 'N/A')
                source_name = source.metadata.get('source', 'Unknown')
                print(f"{i}. Page {page} from {os.path.basename(source_name)}")
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
```

---

## Testing Your Application

### Test 1: Basic Functionality
1. Run: `python main.py`
2. You should see the system loading documents and creating the database
3. Ask a test question: "What are common pests in agriculture?"
4. Verify you get an answer with sources

### Test 2: Different Question Types
Try these questions:
- "How do I manage pests in cotton?"
- "What is the best time to plant wheat?"
- "How much water does rice need?"

### Test 3: Error Handling
- Try asking a question when no documents are loaded
- Try with an invalid API key (temporarily)
- Verify error messages are clear

---

## Troubleshooting

### Problem: "No module named 'langchain'"
**Solution:** Make sure your virtual environment is activated and run `pip install -r requirements.txt`

### Problem: "OPENAI_API_KEY not found"
**Solution:** 
1. Check that `.env` file exists
2. Verify the key is correct: `OPENAI_API_KEY=sk-...`
3. Make sure there are no spaces around the `=` sign

### Problem: "No PDF files found"
**Solution:** 
1. Check that PDFs are in the `documents/` folder
2. Verify file extensions are `.pdf` (not `.PDF`)
3. Check file permissions

### Problem: "Rate limit exceeded"
**Solution:** 
- You've hit OpenAI API rate limits
- Wait a few minutes and try again
- Consider upgrading your OpenAI plan

### Problem: Answers are not accurate
**Solution:**
- Add more relevant documents
- Increase `NUM_RETRIEVED_DOCS` in config.py
- Try using GPT-4 instead of GPT-3.5-turbo

---

## Next Steps & Enhancements

### Basic Enhancements
1. **Add Web Interface** - Use Streamlit for a user-friendly UI
2. **Add More Documents** - Expand your knowledge base
3. **Improve Chunking** - Experiment with different chunk sizes
4. **Add Categories** - Organize by crop type, pest type, etc.

### Advanced Enhancements
1. **Multilingual Support** - Add translation for regional languages
2. **Image Support** - Process images of pests/diseases
3. **Voice Input** - Allow farmers to ask questions via voice
4. **Mobile App** - Create a mobile-friendly version
5. **Offline Mode** - Use local LLMs for offline access

### Web Interface Example (Streamlit)
Create `app.py`:

```python
import streamlit as st
from main import main, load_vector_store, create_qa_chain, ask_question
from config import DOCUMENTS_DIR, OPENAI_API_KEY

st.set_page_config(page_title="🌾 Agri Crop Q&A", page_icon="🌾")

st.title("🌾 Agri Crop Management Q&A System")
st.markdown("Ask questions about crop management and pest control")

# Initialize session state
if 'qa_chain' not in st.session_state:
    with st.spinner("Loading system..."):
        vectorstore = load_vector_store()
        if vectorstore:
            st.session_state.qa_chain = create_qa_chain(vectorstore)
            st.success("System loaded!")
        else:
            st.error("No database found. Please run main.py first to create the database.")

# Question input
question = st.text_input("Ask your question:", placeholder="e.g., What are common cotton pests?")

if st.button("Get Answer") and question:
    if 'qa_chain' in st.session_state:
        result = ask_question(st.session_state.qa_chain, question)
        st.markdown("### Answer:")
        st.write(result["answer"])
        
        if result["sources"]:
            st.markdown("### Sources:")
            for i, source in enumerate(result["sources"][:3], 1):
                st.write(f"{i}. {source.metadata.get('source', 'Unknown')}")
    else:
        st.error("System not initialized. Please run main.py first.")
```

Run with: `streamlit run app.py`

---

## Summary

You've built a complete RAG-based Q&A system! Here's what you learned:

✅ **Document Loading** - How to load PDF documents  
✅ **Text Chunking** - How to split documents for better retrieval  
✅ **Vector Databases** - How to create and use embeddings  
✅ **Retrieval** - How to find relevant information  
✅ **Generation** - How to generate answers using LLMs  
✅ **Q&A Chain** - How to combine retrieval and generation  

**Congratulations!** 🎉 You now have a working RAG application that can answer questions about agricultural documents.

---

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [ChromaDB Documentation](https://www.trychroma.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## Support

If you encounter issues:
1. Check the Troubleshooting section above
2. Review error messages carefully
3. Verify all dependencies are installed
4. Check that your API key is valid
5. Ensure documents are in the correct format

Happy coding! 🚀
