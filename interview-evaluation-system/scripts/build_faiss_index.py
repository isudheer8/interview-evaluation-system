import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# =====================================================
# PATHS (MATCH REPO STRUCTURE)
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = (
    BASE_DIR / "data" / "corpus" / "processed_chunks" / "corpus_chunks.json"
)
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"

FAISS_INDEX_PATH = EMBEDDINGS_DIR / "faiss_index.bin"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "doc_embeddings.npy"

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# EMBEDDING CONFIG (FROZEN)
# =====================================================
EMBEDDING_MODEL_NAME = "multi-qa-mpnet-base-dot-v1"

# =====================================================
# LOAD CORPUS
# =====================================================
def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# =====================================================
# BUILD FAISS INDEX
# =====================================================
def build_faiss_index():
    print("[INFO] Loading corpus chunks...")
    corpus = load_chunks()

    texts = [entry["text"] for entry in corpus]
    print(f"[INFO] Loaded {len(texts)} chunks")

    print("[INFO] Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("[INFO] Computing embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32
    )

    # Normalize for cosine similarity (Inner Product)
    faiss.normalize_L2(embeddings)

    print("[INFO] Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print("[INFO] Saving FAISS index and embeddings...")
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    np.save(EMBEDDINGS_PATH, embeddings)

    print("[DONE] FAISS index built successfully")
    print(f"[INDEX] {FAISS_INDEX_PATH}")
    print(f"[EMB]   {EMBEDDINGS_PATH}")

# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    build_faiss_index()
