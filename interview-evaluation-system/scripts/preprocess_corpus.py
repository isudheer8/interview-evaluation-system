import json
import re
from pathlib import Path
import pdfplumber
from nltk.tokenize import word_tokenize

# =====================================================
# PATHS (MATCH REPO STRUCTURE)
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DOCS_DIR = BASE_DIR / "data" / "corpus" / "raw_docs"
CHUNKS_DIR = BASE_DIR / "data" / "corpus" / "processed_chunks"
METADATA_FILE = BASE_DIR / "data" / "corpus" / "metadata.json"

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# CHUNKING CONFIG (FROZEN)
# =====================================================
CHUNK_SIZE = 250
OVERLAP = 40

# =====================================================
# TEXT UTILITIES
# =====================================================
def clean_text(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_text(pdf_path: Path) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
    return clean_text("\n".join(pages))


def chunk_text(text: str):
    tokens = word_tokenize(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        start = end - OVERLAP

    return chunks

# =====================================================
# MAIN PIPELINE
# =====================================================
def preprocess_corpus():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    all_chunks = []

    for pdf_path in RAW_DOCS_DIR.glob("*.pdf"):
        pdf_name = pdf_path.name
        print(f"[INFO] Processing {pdf_name}")

        if pdf_name not in metadata:
            raise ValueError(f"Metadata missing for {pdf_name}")

        book_meta = metadata[pdf_name]

        # PDF → TEXT
        text = extract_pdf_text(pdf_path)

        # TEXT → CHUNKS
        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{pdf_path.stem}_{idx}",
                "text": chunk,
                "source_book": book_meta["book_name"],
                "authors": book_meta["authors"],
                "domain": book_meta["domain"]
            })

    output_path = CHUNKS_DIR / "corpus_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"[DONE] Created {len(all_chunks)} chunks")
    print(f"[OUTPUT] {output_path}")

# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    import nltk
    nltk.download("punkt")
    preprocess_corpus()
