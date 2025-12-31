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

The system follows a modular pipeline architecture designed to transform raw student input into a multi-dimensional score:

1.  **Ingestion Layer**: Receives raw audio (WAV/MP3) or text.
2.  **Processing Layer**: 
    - **DSP Module**: Noise reduction and feature extraction (Pitch/Intensity).
    - **ASR Module**: Whisper-based transcription of audio to text.
3.  **Analysis Layer (The Scoring Triad)**:
    - **Semantic Scorer**: Uses SBERT to compare student intent vs. ideal answers.
    - **Concept Scorer**: Regex-based verification of technical keyword density.
    - **RAG Scorer**: FAISS-based retrieval from textbooks to verify factual grounding.
4.  **Fusion Layer**: A weighted engine that combines signals into a final 0.0–10.0 score.
5.  **Presentation Layer**: Streamlit UI for real-time feedback.
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

```
interview-evaluation-system/
│
├── api/             # FastAPI backend (Routes & Main entry)
├── core/            # Core ML, DSP, RAG, and Orchestration logic
├── data/            # Questions, corpus chunks, FAISS indices, and samples
├── frontend/        # Streamlit UI implementation
├── experiments/     # Offline evaluation & system validation metrics
├── tests/           # Unit tests for core components
├── scripts/         # Utility scripts (Indexing, Data Cleaning)
├── README.md        # Project documentation
└── requirements.txt # Python dependencies
```

---

## 6. How to Run the Project

### Step 1: Install Dependencies
```
pip install -r requirements.txt
```

### Step 2: Start the Backend (FastAPI)
```
uvicorn api.main:app --reload
```

Backend runs at:
```
http://127.0.0.1:8000
```
### Step 3: Start the Frontend (Streamlit)
```
streamlit run frontend/streamlit_app.py
```

## 7. Evaluation Metrics (Offline)

The system is rigorously validated using an offline evaluation suite to ensure alignment between AI grading and human expert benchmarks.

### Statistical Validation
We utilize two primary metrics to quantify system performance:

| Metric | Purpose | Target |
| :--- | :--- | :--- |
| **Pearson Correlation** | Measures the linear relationship between system scores and human scores. | $> 0.7$ |
| **Mean Absolute Error (MAE)** | Measures the average magnitude of error in the predicted scores. | $< 1.5$ |


### Running Evaluation
Metrics are computed and reported via the dedicated evaluation script:
```
python -m experiments.evaluation.metrics
```

## 8. Design Decisions

- No generative feedback models to avoid hallucinations
- SLM used for semantic reasoning, not text generation
- RAG used for grounding and explainability, not scoring dominance
- Audio delivery metrics are non-punitive to ensure fairness for non-native speakers
- Stateless evaluation to maintain deterministic behavior
