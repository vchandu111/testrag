# Project 3: Teacher's Lesson Plan Generator - Complete Implementation Guide

A beginner-friendly guide to building a tool that helps teachers create lesson plans aligned with specific textbook topics using RAG.

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
A tool where teachers input a textbook and request lesson plans for specific sections or topics. The system generates structured lesson plans with objectives, discussion points, activities, and assessments.

### Example Requests
- "Create a lesson plan for Chapter 3 on the Civil War."
- "Generate a lesson plan about photosynthesis for 9th grade."
- "Make a lesson plan for Unit 5: World War II, duration 90 minutes."

### How It Works
1. Teacher specifies a topic or chapter from the textbook
2. System retrieves relevant sections from the textbook
3. System generates a structured lesson plan with:
   - Learning objectives
   - Key discussion points
   - Activities and exercises
   - Assessment questions
   - Time allocation

### Key Features
- **Structured Output**: Generates formatted lesson plans
- **Topic-Specific**: Aligned with textbook content
- **Customizable**: Adjustable duration and grade level
- **Complete Plans**: Includes all components teachers need

---

## Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of RAG concepts
- Basic knowledge of educational planning

### Required Software
- Python 3.8 or higher
- pip (Python package installer)

### Required Accounts
- OpenAI API account (or alternative LLM provider)

---

## Project Structure

```
lesson-plan-generator/
│
├── textbooks/              # Store textbook PDFs
│
├── generated_plans/        # Generated lesson plans (PDF/Word)
│
├── vector_db/              # Vector database
│
├── templates/              # Lesson plan templates
│
├── .env                    # API keys
│
├── requirements.txt        # Dependencies
│
├── config.py              # Configuration
│
├── main.py                # Main application
│
└── README.md              # This file
```

---

## Step-by-Step Implementation

### Step 1: Set Up Project

```bash
mkdir lesson-plan-generator
cd lesson-plan-generator
python3 -m venv venv
source venv/bin/activate
mkdir textbooks generated_plans vector_db templates
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
TEXTBOOKS_DIR = "textbooks"
VECTOR_DB_DIR = "vector_db"
PLANS_DIR = "generated_plans"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
NUM_RETRIEVED_DOCS = 5
TEMPERATURE = 0.7  # Higher for creative lesson planning
MODEL_NAME = "gpt-3.5-turbo"
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
from docx import Document
from config import *

def load_textbook(file_path):
    """Load textbook PDF."""
    loader = PyPDFLoader(file_path)
    return loader.load()

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

def create_lesson_plan_prompt():
    """Create prompt template for lesson plan generation."""
    template = """You are an expert teacher creating a lesson plan. Based on the following textbook content, create a comprehensive lesson plan.

Textbook Content:
{context}

Topic/Chapter Request: {topic}
Grade Level: {grade_level}
Duration: {duration} minutes

Create a detailed lesson plan with the following structure:

## Lesson Plan: [Topic Name]

### Learning Objectives
- [List 3-5 specific learning objectives]

### Materials Needed
- [List required materials]

### Introduction (5-10 minutes)
[Engaging introduction activity]

### Main Content ({main_duration} minutes)
[Key points to cover based on textbook content]

### Activities
1. [Activity 1 with description]
2. [Activity 2 with description]
3. [Activity 3 with description]

### Discussion Points
- [Discussion point 1]
- [Discussion point 2]
- [Discussion point 3]

### Assessment
[Create 3-5 assessment questions based on the content]

### Homework/Extension
[Suggestions for homework or extended learning]

### Notes
[Additional teaching notes and tips]

Ensure the lesson plan is:
- Age-appropriate for {grade_level}
- Aligned with the textbook content
- Engaging and interactive
- Complete within {duration} minutes"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "topic", "grade_level", "duration", "main_duration"]
    )

def generate_lesson_plan(vectorstore, topic, grade_level="9th grade", duration=60):
    """Generate a lesson plan for the given topic."""
    print(f"\nGenerating lesson plan for: {topic}")
    print(f"Grade Level: {grade_level}")
    print(f"Duration: {duration} minutes")
    
    # Retrieve relevant content
    retriever = vectorstore.as_retriever(search_kwargs={"k": NUM_RETRIEVED_DOCS})
    relevant_docs = retriever.get_relevant_documents(topic)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Calculate main content duration (60% of total time)
    main_duration = int(duration * 0.6)
    
    # Create LLM chain
    llm = OpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE,
        model_name=MODEL_NAME
    )
    
    prompt = create_lesson_plan_prompt()
    
    # Generate lesson plan
    response = llm(prompt.format(
        context=context,
        topic=topic,
        grade_level=grade_level,
        duration=duration,
        main_duration=main_duration
    ))
    
    return response

def save_lesson_plan(content, filename):
    """Save lesson plan as Word document."""
    doc = Document()
    
    # Split content into sections
    lines = content.split('\n')
    current_paragraph = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_paragraph:
                doc.add_paragraph(current_paragraph)
                current_paragraph = ""
            continue
        
        # Check for headers
        if line.startswith('##'):
            if current_paragraph:
                doc.add_paragraph(current_paragraph)
                current_paragraph = ""
            doc.add_heading(line.replace('##', '').strip(), level=1)
        elif line.startswith('###'):
            if current_paragraph:
                doc.add_paragraph(current_paragraph)
                current_paragraph = ""
            doc.add_heading(line.replace('###', '').strip(), level=2)
        elif line.startswith('-') or line.startswith('*'):
            if current_paragraph:
                doc.add_paragraph(current_paragraph)
                current_paragraph = ""
            doc.add_paragraph(line, style='List Bullet')
        else:
            if current_paragraph:
                current_paragraph += " " + line
            else:
                current_paragraph = line
    
    if current_paragraph:
        doc.add_paragraph(current_paragraph)
    
    filepath = os.path.join(PLANS_DIR, filename)
    doc.save(filepath)
    print(f"✓ Lesson plan saved to {filepath}")

def main():
    """Main function."""
    print("=" * 60)
    print("📝 Teacher's Lesson Plan Generator")
    print("=" * 60)
    
    # Load or create vector store
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        print("Loading existing database...")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
    else:
        print("Creating new database...")
        # Get first PDF from textbooks folder
        pdf_files = [f for f in os.listdir(TEXTBOOKS_DIR) if f.endswith('.pdf')]
        if not pdf_files:
            print("No textbooks found! Add PDFs to 'textbooks' folder.")
            return
        
        file_path = os.path.join(TEXTBOOKS_DIR, pdf_files[0])
        documents = load_textbook(file_path)
        vectorstore = create_vector_store(documents)
    
    # Interactive loop
    print("\n" + "=" * 60)
    print("System ready! Generate lesson plans")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        topic = input("\n📚 Topic/Chapter: ").strip()
        if topic.lower() in ['quit', 'exit', 'q']:
            break
        
        grade_level = input("🎓 Grade Level (default: 9th grade): ").strip() or "9th grade"
        duration = input("⏱️  Duration in minutes (default: 60): ").strip()
        duration = int(duration) if duration.isdigit() else 60
        
        print("\nGenerating lesson plan...")
        plan_content = generate_lesson_plan(vectorstore, topic, grade_level, duration)
        
        print("\n" + "=" * 60)
        print("Generated Lesson Plan:")
        print("=" * 60)
        print(plan_content)
        
        save = input("\n💾 Save as Word document? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"lesson_plan_{topic.replace(' ', '_')[:30]}.docx"
            save_lesson_plan(plan_content, filename)
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
```

---

## Testing Your Application

### Test 1: Basic Lesson Plan
```bash
python main.py
```

Try:
- Topic: "Civil War"
- Grade: "8th grade"
- Duration: 45

### Test 2: Different Subjects
- History: "World War II"
- Science: "Photosynthesis"
- Math: "Algebra basics"

---

## Troubleshooting

### Problem: Lesson plans too generic
**Solution:** Increase `NUM_RETRIEVED_DOCS` to get more context, or refine your topic description.

### Problem: Plans too long/short
**Solution:** Adjust the duration parameter and the prompt template to better allocate time.

---

## Next Steps & Enhancements

1. **Template Customization**: Allow teachers to customize lesson plan templates
2. **Standards Alignment**: Align with curriculum standards (Common Core, etc.)
3. **Resource Links**: Include links to videos, worksheets, and activities
4. **Multi-Day Plans**: Generate week-long or unit-long plans
5. **Export Formats**: Support PDF, Google Docs, and other formats

---

## Summary

You've built a lesson plan generator that:
✅ **Generates Structured Plans** - Complete with all components  
✅ **Textbook-Aligned** - Based on actual textbook content  
✅ **Customizable** - Adjustable for grade level and duration  
✅ **Exportable** - Saves as Word documents  

**Key Difference:** This project generates structured, formatted output rather than just Q&A, making it perfect for teachers who need complete lesson plans.

Happy teaching! 📚
