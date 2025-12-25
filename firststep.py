import os

BASE_DIR = "interview-evaluation-system"

# List of directories to create
directories = [
    "config",
    "core/interfaces",
    "core/models/semantic",
    "core/models/keyword",
    "core/models/audio",
    "core/models/rag",
    "core/models/fusion",
    "core/orchestration",
    "core/utils",
    "data/questions",
    "data/corpus/raw_docs",
    "data/corpus/processed_chunks",
    "data/embeddings",
    "data/samples",
    "api/routes",
    "api/schemas",
    "frontend/components",
    "frontend/assets",
    "experiments/notebooks",
    "experiments/evaluation",
    "tests",
    "scripts"
]

# Files to create (empty stubs)
files = [
    "README.md",
    "requirements.txt",
    ".gitignore",

    "config/default.yaml",
    "config/text_only.yaml",
    "config/multimodal_stub.yaml",
    "config/weights.yaml",

    "core/interfaces/dsp.py",
    "core/interfaces/asr.py",
    "core/interfaces/text_preprocessor.py",
    "core/interfaces/semantic_scorer.py",
    "core/interfaces/concept_scorer.py",
    "core/interfaces/retriever.py",
    "core/interfaces/generator.py",
    "core/interfaces/fusion_engine.py",
    "core/interfaces/orchestrator.py",
    "core/interfaces/config.py",

    "core/models/semantic/sbert_scorer.py",
    "core/models/semantic/__init__.py",

    "core/models/keyword/regex_concept_scorer.py",
    "core/models/keyword/__init__.py",

    "core/models/audio/dsp_stub.py",
    "core/models/audio/asr_stub.py",
    "core/models/audio/__init__.py",

    "core/models/rag/faiss_retriever.py",
    "core/models/rag/t5_generator_stub.py",
    "core/models/rag/__init__.py",

    "core/models/fusion/weighted_fusion.py",
    "core/models/fusion/__init__.py",

    "core/orchestration/interview_orchestrator.py",
    "core/orchestration/session_manager.py",

    "core/utils/text_utils.py",
    "core/utils/audio_utils.py",
    "core/utils/logging.py",
    "core/utils/metrics.py",

    "data/questions/ece_questions.csv",
    "data/questions/key_concepts.csv",

    "data/corpus/metadata.json",

    "data/embeddings/faiss_index.bin",
    "data/embeddings/doc_embeddings.npy",

    "data/samples/sample_answers.json",
    "data/samples/sample_audio.wav",

    "api/main.py",
    "api/routes/submit_text.py",
    "api/routes/submit_audio.py",
    "api/routes/reports.py",

    "api/schemas/request_models.py",
    "api/schemas/response_models.py",

    "frontend/streamlit_app.py",
    "frontend/components/question_view.py",
    "frontend/components/answer_input.py",
    "frontend/components/feedback_view.py",
    "frontend/assets/styles.css",

    "experiments/notebooks/dsp_experiments.ipynb",
    "experiments/notebooks/asr_tests.ipynb",
    "experiments/notebooks/sbert_training.ipynb",
    "experiments/notebooks/rag_retrieval_tests.ipynb",

    "experiments/evaluation/metrics.py",
    "experiments/evaluation/fairness_tests.py",

    "tests/test_semantic_scorer.py",
    "tests/test_concept_scorer.py",
    "tests/test_fusion_engine.py",
    "tests/test_orchestrator.py",

    "scripts/build_faiss_index.py",
    "scripts/preprocess_corpus.py",
    "scripts/run_demo.py"
]

# Create base directory
os.makedirs(BASE_DIR, exist_ok=True)

# Create directories
for d in directories:
    os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

# Create files
for f in files:
    file_path = os.path.join(BASE_DIR, f)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, "w") as fp:
            fp.write("")

print("✅ Folder structure created successfully!")
print(f"📁 Root directory: {BASE_DIR}")
