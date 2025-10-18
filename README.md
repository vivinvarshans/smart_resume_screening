# Smart Resume Screener - AI-Powered ATS Optimization System

<div align="center">

**Stop Getting Auto-Rejected by ATS Bots**

A modern, AI-powered resume matching system that helps job seekers optimize their resumes for ATS (Applicant Tracking Systems) and significantly improve their chances of landing interviews.


</div>

---

Video Link: https://drive.google.com/file/d/1h1OMFsJjvlN0mczmTcun9Bmz1Hb9iTPC/view?usp=sharing
Site Live Deployed Link: https://frontend-vivinvarshanojasracing-7088-vivins-projects-827d22e7.vercel.app?_vercel_share=3amZBbSXXAJJ5KrEX3D9pBJP4MJ9pUSX

## Features

###  Core Features
- ** Resume Upload**: Support for PDF and DOCX formats (max 2MB)
- ** Job Description Analysis**: Intelligent parsing of job requirements
- ** AI-Powered Matching**: Advanced semantic similarity using TF-IDF + Cosine Similarity
- ** Match Scoring**: Realistic 1-10 scale scoring with detailed breakdown
- ** Keyword Analysis**: Matched vs missing skills identification
- ** AI Resume Improvement**: Groq LLM-powered resume optimization
- ** Detailed Insights**: Strengths, gaps, and actionable recommendations
- ** Real-time Processing**: Async operations with streaming support

###  User Experience
- ** Modern UI**: Beautiful, responsive design with smooth animations
- ** Mobile-First**: Fully responsive across all devices
- ** Real-time**: Instant feedback and loading states
- ** Interactive**: Hover effects and micro-interactions

###  Technical Features
- ** High Performance**: Async operations and optimized queries
- ** Error Handling**: Comprehensive error handling with fallbacks
- ** Logging**: Detailed logging for debugging and monitoring
- ** Fallback Systems**: Graceful degradation when AI services fail
- ** Type Safety**: Full TypeScript implementation

---

##  System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND LAYER (React + TypeScript)                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐                 │
│  │  Resume Upload   │  │  Job Description │  │  Match Dashboard │                 │
│  │   Component      │  │     Component    │  │    Component     │                 │
│  │                  │  │                  │  │                  │                 │
│  │  • Drag & Drop   │  │  • Text Input    │  │  • Score Circle  │                 │
│  │  • File Validate │  │  • Validation    │  │  • Keywords      │                 │
│  │  • Progress UI   │  │  • Submit        │  │  • Improvements  │                 │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘                 │
└───────────┼──────────────────────┼──────────────────────┼───────────────────────────┘
            │                      │                      │
            │ HTTP/REST            │ HTTP/REST            │ HTTP/REST
            │ (Axios)              │ (Axios)              │ (Axios)
            ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           API GATEWAY (FastAPI)                                     │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                          API Router v1                                       │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │  │
│  │  │ POST /resumes/  │  │ POST /jobs/     │  │ POST /resumes/  │            │  │
│  │  │     upload      │  │     upload      │  │     improve     │            │  │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘            │  │
│  └───────────┼──────────────────────┼──────────────────────┼───────────────────┘  │
└──────────────┼──────────────────────┼──────────────────────┼──────────────────────┘
               │                      │                      │
               ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        SERVICE LAYER (Business Logic)                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │  Resume Service  │  │   Job Service    │  │  Score Improvement Service       │ │
│  │                  │  │                  │  │                                  │ │
│  │ • Parse PDF/DOCX │  │ • Parse Job Text │  │ • Extract Keywords (AI)          │ │
│  │ • Convert to MD  │  │ • Extract Fields │  │ • Calculate Match Score          │ │
│  │ • Store Resume   │  │ • Store Job      │  │ • Analyze Gaps                   │ │
│  │ • Extract Data   │  │ • Process Data   │  │ • Improve Resume (AI)            │ │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────────────────────┘ │
└───────────┼──────────────────────┼──────────────────────┼───────────────────────────┘
            │                      │                      │
            │                      │                      │ AI Calls
            │                      │                      ▼
            │                      │         ┌────────────────────────────────────────┐
            │                      │         │    AI LAYER (Groq Integration)         │
            │                      │         │  ┌──────────────────────────────────┐  │
            │                      │         │  │      Agent Manager               │  │
            │                      │         │  │  • Strategy Pattern (JSON/MD)    │  │
            │                      │         │  │  • Provider Factory              │  │
            │                      │         │  └──────────┬───────────────────────┘  │
            │                      │         │             │                          │
            │                      │         │  ┌──────────▼───────────────────────┐  │
            │                      │         │  │    LlamaIndex Provider           │  │
            │                      │         │  │  • Groq Client                   │  │
            │                      │         │  │  • Model: llama-3.3-70b          │  │
            │                      │         │  │  • Temperature: 0                │  │
            │                      │         │  │  • Max Tokens: 8192              │  │
            │                      │         │  └──────────┬───────────────────────┘  │
            │                      │         │             │                          │
            │                      │         │             ▼                          │
            │                      │         │  ┌─────────────────────────────────┐  │
            │                      │         │  │   Groq Cloud API                │  │
            │                      │         │  │   api.groq.com/openai/v1        │  │
            │                      │         │  │                                 │  │
            │                      │         │  │   🤖 Llama 3.3 70B Versatile   │  │
            │                      │         │  └─────────────────────────────────┘  │
            │                      │         └────────────────────────────────────────┘
            │                      │
            ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                      DATA LAYER (SQLite + SQLAlchemy ORM)                           │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                          SQLite Database (app.db)                            │  │
│  │                                                                              │  │
│  │  ┌─────────────┐  ┌──────────────────┐  ┌─────────────┐  ┌──────────────┐ │  │
│  │  │   resumes   │  │ processed_resumes│  │    jobs     │  │ processed_   │ │  │
│  │  │             │  │                  │  │             │  │    jobs      │ │  │
│  │  │ • id        │  │ • resume_id (FK) │  │ • id        │  │ • job_id(FK) │ │  │
│  │  │ • resume_id │◄─┤ • personal_data  │  │ • job_id    │◄─┤ • job_title  │ │  │
│  │  │ • content   │  │ • experiences    │  │ • resume_id │  │ • company    │ │  │
│  │  │ • type      │  │ • projects       │  │ • content   │  │ • keywords   │ │  │
│  │  │ • created   │  │ • skills         │  │ • created   │  │ • qualif.    │ │  │
│  │  └─────────────┘  │ • keywords       │  └─────────────┘  └──────────────┘ │  │
│  │                   └──────────────────┘                                      │  │
│  │                                                                              │  │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │  │
│  │  │           job_resume_association (Many-to-Many)                      │  │  │
│  │  │  • resume_id (FK) ──► processed_resumes.resume_id                   │  │  │
│  │  │  • job_id (FK) ──► processed_jobs.job_id                            │  │  │
│  │  └──────────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

### Component Interaction Flow

```
┌──────┐                                                                    ┌──────────┐
│ USER │                                                                    │ GROQ API │
└───┬──┘                                                                    └────┬─────┘
    │                                                                            │
    │ 1. Upload Resume (PDF/DOCX)                                               │
    ├──────────────────────────────────────────┐                                │
    │                                           ▼                                │
    │                                    ┌─────────────┐                         │
    │                                    │  FRONTEND   │                         │
    │                                    │  (React)    │                         │
    │                                    └──────┬──────┘                         │
    │                                           │                                │
    │                                           │ POST /resumes/upload           │
    │                                           │ (multipart/form-data)          │
    │                                           ▼                                │
    │                                    ┌─────────────┐                         │
    │                                    │   FASTAPI   │                         │
    │                                    │  API Router │                         │
    │                                    └──────┬──────┘                         │
    │                                           │                                │
    │                                           │ ResumeService                  │
    │                                           ▼                                │
    │                                    ┌─────────────┐                         │
    │                                    │   SERVICE   │                         │
    │                                    │    LAYER    │                         │
    │                                    │             │                         │
    │                                    │ • Parse PDF │                         │
    │                                    │ • Convert   │                         │
    │                                    └──────┬──────┘                         │
    │                                           │                                │
    │                                           │ Store                          │
    │                                           ▼                                │
    │                                    ┌─────────────┐                         │
    │                                    │  DATABASE   │                         │
    │                                    │   (SQLite)  │                         │
    │                                    └──────┬──────┘                         │
    │                                           │                                │
    │                                           │ resume_id                      │
    │ ◄─────────────────────────────────────────┴────────                        │
    │ Success: Resume uploaded!                                                  │
    │                                                                            │
    │ 2. Enter Job Description                                                   │
    ├──────────────────────────────────────────┐                                │
    │                                           ▼                                │
    │                                    ┌─────────────┐                         │
    │                                    │  FRONTEND   │                         │
    │                                    └──────┬──────┘                         │
    │                                           │                                │
    │                                           │ POST /jobs/upload              │
    │                                           ▼                                │
    │                                    ┌─────────────┐                         │
    │                                    │   FASTAPI   │                         │
    │                                    └──────┬──────┘                         │
    │                                           │                                │
    │                                           │ JobService                     │
    │                                           ▼                                │
    │                                    ┌─────────────┐                         │
    │                                    │   SERVICE   │                         │
    │                                    └──────┬──────┘                         │
    │                                           │                                │
    │                                           │ Store                          │
    │                                           ▼                                │
    │                                    ┌─────────────┐                         │
    │                                    │  DATABASE   │                         │
    │                                    └──────┬──────┘                         │
    │                                           │                                │
    │ ◄─────────────────────────────────────────┴────────                        │
    │ Success: Job stored!                                                       │
    │                                                                            │
    │ 3. Request Match Analysis                                                  │
    ├──────────────────────────────────────────┐                                │
    │                                           ▼                                │
    │                                    ┌─────────────┐                         │
    │                                    │  FRONTEND   │                         │
    │                                    └──────┬──────┘                         │
    │                                           │                                │
    │                                           │ POST /resumes/improve          │
    │                                           │ {resume_id, job_id}            │
    │                                           ▼                                │
    │                                    ┌─────────────┐                         │
    │                                    │   FASTAPI   │                         │
    │                                    └──────┬──────┘                         │
    │                                           │                                │
    │                                           │ ScoreImprovementService        │
    │                                           ▼                                │
    │                                    ┌─────────────┐                         │
    │                                    │   SERVICE   │                         │
    │                                    │             │                         │
    │                                    │ Step 1:     │                         │
    │                                    │ Fetch Data  │                         │
    │                                    └──────┬──────┘                         │
    │                                           │                                │
    │                                           ▼                                │
    │                                    ┌─────────────┐                         │
    │                                    │  DATABASE   │                         │
    │                                    └──────┬──────┘                         │
    │                                           │                                │
    │                                           │ Resume + Job                   │
    │                                           ▼                                │
    │                                    ┌─────────────┐                         │
    │                                    │   SERVICE   │                         │
    │                                    │             │                         │
    │                                    │ Step 2:     │                         │
    │                                    │ Extract     │                         │
    │                                    │ Keywords    │─────────────────────────┤
    │                                    └──────┬──────┘                         │
    │                                           │                                │
    │                                           │ AI Request                     │
    │                                           │ "Extract keywords..."          │
    │                                           ├────────────────────────────────►
    │                                           │                                │
    │                                           │ ◄──────────────────────────────┤
    │                                           │ ["python", "react", "aws"]     │
    │                                           │                                │
    │                                    ┌──────▼──────┐                         │
    │                                    │   SERVICE   │                         │
    │                                    │             │                         │
    │                                    │ Step 3:     │                         │
    │                                    │ Analyze     │                         │
    │                                    │ Match       │─────────────────────────┤
    │                                    └──────┬──────┘                         │
    │                                           │                                │
    │                                           │ AI Request                     │
    │                                           │ "Analyze match..."             │
    │                                           ├────────────────────────────────►
    │                                           │                                │
    │                                           │ ◄──────────────────────────────┤
    │                                           │ {score: 65, gaps: [...]}       │
    │                                           │                                │
    │                                    ┌──────▼──────┐                         │
    │                                    │   SERVICE   │                         │
    │                                    │             │                         │
    │                                    │ Step 4:     │                         │
    │                                    │ Improve     │                         │
    │                                    │ Resume      │─────────────────────────┤
    │                                    └──────┬──────┘                         │
    │                                           │                                │
    │                                           │ AI Request                     │
    │                                           │ "Improve resume..."            │
    │                                           ├────────────────────────────────►
    │                                           │                                │
    │                                           │ ◄──────────────────────────────┤
    │                                           │ Improved Resume Text           │
    │                                           │                                │
    │                                    ┌──────▼──────┐                         │
    │                                    │   SERVICE   │                         │
    │                                    │             │                         │
    │                                    │ Step 5:     │                         │
    │                                    │ Re-analyze  │─────────────────────────┤
    │                                    └──────┬──────┘                         │
    │                                           │                                │
    │                                           │ AI Request                     │
    │                                           │ "Re-score improved..."         │
    │                                           ├────────────────────────────────►
    │                                           │                                │
    │                                           │ ◄──────────────────────────────┤
    │                                           │ {score: 85, ...}               │
    │                                           │                                │
    │                                           ▼                                │
    │                                    ┌─────────────┐                         │
    │                                    │   FASTAPI   │                         │
    │                                    └──────┬──────┘                         │
    │                                           │                                │
    │                                           │ Complete Results               │
    │                                           ▼                                │
    │                                    ┌─────────────┐                         │
    │                                    │  FRONTEND   │                         │
    │                                    └──────┬──────┘                         │
    │                                           │                                │
    │ ◄─────────────────────────────────────────┘                                │
    │ Display: Score, Keywords, Improvements                                     │
    │                                                                            │
    ▼                                                                            ▼
```

---

##  Tech Stack

### Frontend Stack

| Technology | Version | Purpose | Documentation |
|------------|---------|---------|---------------|
| **React** | 18.3.1 | UI Framework | [React Docs](https://react.dev) |
| **TypeScript** | 5.8.3 | Type Safety | [TS Docs](https://www.typescriptlang.org) |
| **Vite** | 5.4.19 | Build Tool | [Vite Docs](https://vitejs.dev) |
| **Tailwind CSS** | 3.4.17 | Styling | [Tailwind Docs](https://tailwindcss.com) |
| **Shadcn/ui** | Latest | Component Library | [Shadcn Docs](https://ui.shadcn.com) |
| **Radix UI** | Latest | Headless Components | [Radix Docs](https://www.radix-ui.com) |
| **Lucide React** | 0.462.0 | Icons | [Lucide Docs](https://lucide.dev) |
| **Axios** | 1.12.2 | HTTP Client | [Axios Docs](https://axios-http.com) |
| **React Router** | 6.30.1 | Routing | [Router Docs](https://reactrouter.com) |
| **React Dropzone** | 14.3.8 | File Upload | [Dropzone Docs](https://react-dropzone.js.org) |
| **TanStack Query** | 5.83.0 | Data Fetching | [Query Docs](https://tanstack.com/query) |

### ⚙️ Backend Stack

| Technology | Version | Purpose | Documentation |
|------------|---------|---------|---------------|
| **FastAPI** | 0.115.12 | Web Framework | [FastAPI Docs](https://fastapi.tiangolo.com) |
| **Python** | 3.11+ | Runtime | [Python Docs](https://docs.python.org) |
| **SQLAlchemy** | 2.0.40 | ORM | [SQLAlchemy Docs](https://www.sqlalchemy.org) |
| **SQLite** | 3.x | Database | [SQLite Docs](https://www.sqlite.org) |
| **Pydantic** | 2.11.3 | Data Validation | [Pydantic Docs](https://docs.pydantic.dev) |
| **Uvicorn** | 0.34.0 | ASGI Server | [Uvicorn Docs](https://www.uvicorn.org) |
| **aiosqlite** | 0.21.0 | Async SQLite | [aiosqlite Docs](https://aiosqlite.omnilib.dev) |

### 🤖 AI & ML Stack

| Technology | Version | Purpose | Documentation |
|------------|---------|---------|---------------|
| **Groq API** | Latest | LLM Provider | [Groq Docs](https://console.groq.com/docs) |
| **Llama 3.3 70B** | Latest | Language Model | [Llama Docs](https://www.llama.com) |
| **Ollama** | 0.4.7 | Local LLM Runtime | [Ollama Docs](https://ollama.ai) |
| **OpenAI SDK** | 1.75.0 | LLM Client | [OpenAI Docs](https://platform.openai.com) |
| **markitdown** | 0.1.2 | Document Parser | [markitdown GitHub](https://github.com/microsoft/markitdown) |
| **Magika** | 0.6.1 | File Type Detection | [Magika Docs](https://google.github.io/magika) |

---

## Getting Started

###  Prerequisites

- **Node.js** 18+ ([Download](https://nodejs.org/))
- **Python** 3.11+ ([Download](https://python.org/))
- **Groq API Key** ([Get Free Key](https://console.groq.com/)) - **Required for AI features**
- **Git** ([Download](https://git-scm.com/))

###  Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/resume-matcher.git
cd resume-matcher
```

#### 2. Backend Setup

```bash
cd apps/backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.sample .env

# Edit .env and configure:
# - LLM_API_KEY="your-groq-api-key-here"
# - LLM_PROVIDER="groq"
# - LL_MODEL="llama-3.3-70b-versatile"
# - LLM_BASE_URL="https://api.groq.com/openai/v1"
```

#### 3. Frontend Setup

```bash
cd apps/frontend

# Install dependencies
npm install

# Setup environment (optional)
cp .env.example .env

# Edit .env if needed:
# VITE_API_BASE_URL=http://localhost:8000/api/v1
```

###  Running the Application

#### Option 1: Using the Start Script (Recommended)

```bash
# From project root
./start.sh
```

This will start both backend and frontend servers concurrently.

#### Option 2: Manual Start

**Terminal 1 - Backend Server:**
```bash
cd apps/backend
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend Development Server:**
```bash
cd apps/frontend
npm run dev
```

### 🌐 Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative API Docs**: http://localhost:8000/redoc (ReDoc)

---
