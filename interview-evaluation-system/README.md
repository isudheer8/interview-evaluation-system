# Interview Evaluation System  
### DSP + RAG Augmented Small Language Model for High-Precision Interview Evaluation

---

## 1. Project Overview

This project implements a **multimodal technical interview evaluation system** that assesses student responses using a combination of:

- **Digital Signal Processing (DSP)** for audio preprocessing  
- **Automatic Speech Recognition (ASR)** for speech-to-text conversion  
- **Small Language Model (SLM)** for semantic understanding  
- **Retrieval-Augmented Generation (RAG)** for evidence grounding  
- **Hybrid scoring** for fair and explainable evaluation  

The system is designed for **academic interview evaluation**, prioritizing **determinism, explainability, and fairness** over generative behavior.

---

## 2. Key Features

- Text-based interview answer evaluation  
- Audio-based interview answer evaluation (DSP + ASR)  
- Semantic similarity scoring using a Small Language Model (SBERT)  
- Keyword / concept coverage scoring  
- RAG-based evidence retrieval from authoritative textbooks  
- Weighted fusion of multiple scoring signals  
- Bias-aware audio delivery analysis (non-punitive)  
- Streamlit-based demo UI  
- Offline evaluation metrics for system validation  

---

## 3. System Architecture (High-Level)
Audio/Text Input
↓
DSP (Audio only)
↓
ASR (Audio only)
↓
Text Processing
↓
SLM Semantic Scorer (SBERT)
↓
Keyword / Concept Scorer
↓
RAG Retriever (FAISS)
↓
Weighted Fusion Engine
↓
Final Score + Verdict + Evidence
---

## 4. Technology Stack

### Core Technologies
- Python 3.10+
- FastAPI (Backend API)
- Streamlit (Frontend Demo UI)

### Machine Learning & NLP
- Sentence-Transformers (SBERT – all-MiniLM-L6-v2)
- FAISS (Vector similarity search)
- Faster-Whisper (ASR)

### Audio Processing
- Librosa
- NumPy
- SciPy

### Utilities
- Pydantic (Request/response validation)
- Requests (UI ↔ API communication)

---

## 5. Folder Structure


interview-evaluation-system/
│
├── api/ # FastAPI backend
├── core/ # Core ML, DSP, RAG, orchestration logic
├── data/ # Questions, corpus, embeddings, samples
├── frontend/ # Streamlit UI
├── experiments/ # Offline evaluation & metrics
├── tests/ # Unit tests
├── scripts/ # Utility scripts
├── README.md
└── requirements.txt


---

## 6. How to Run the Project

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt


### Step 2: Start the Backend (FastAPI)
uvicorn api.main:app --reload


Backend runs at:

http://127.0.0.1:8000

### Step 3: Start the Frontend (Streamlit)

streamlit run frontend/streamlit_app.py


## 7. Evaluation Metrics (Offline)

- The system is validated using offline evaluation:
- Pearson correlation between system scores and human scores
- Mean Absolute Error (MAE)
- Ablation study:
- Semantic-only
- Keyword-only
- Semantic + Keyword (proposed system)

- Metrics are computed via:
experiments/evaluation/metrics.py


## 8. Design Decisions

- No generative feedback models to avoid hallucinations
- SLM used for semantic reasoning, not text generation
- RAG used for grounding and explainability, not scoring dominance
- Audio delivery metrics are non-punitive to ensure fairness for non-native speakers
- Stateless evaluation to maintain deterministic behavior