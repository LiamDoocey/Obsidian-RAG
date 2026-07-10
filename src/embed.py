import os
import hashlib
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

from ingest import ingest_vault

load_dotenv()

VAULT_PATH = os.getenv("OBS_LOC")
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "obsidian_notes"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def make_chunk_id(chunk: dict) -> str:
    headers = chunk.get("headers", {})
    header_path = "|".join(f"{k}:{v}" for k, v in sorted(headers.items()))
    raw = f"{chunk['source']}::{header_path}::{chunk['text'][:50]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def flatten_metadata(chunk: dict) -> dict:
    headers = chunk.get("headers", {})
    header_str = " > ".join(headers.values()) if headers else ""

    return {
        "source": chunk["source"],
        "headers": header_str
    }

def embed_and_store(chunks: list[dict]) -> chromadb.Collection:
    client = chromadb.PersistentClient(path = CHROMA_PATH)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name = EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name = COLLECTION_NAME,
        embedding_function = embedding_fn
    )

    ids = [make_chunk_id(c) for c in chunks]
    docs = [c["text"] for c in chunks]
    metad = [flatten_metadata(c) for c in chunks]

    collection.upsert(
        ids = ids,
        documents = docs,
        metadatas = metad
    )

    return collection

if __name__ == "__main__":
    if not VAULT_PATH:
        raise ValueError("Vault location not set")
    
    print(f"Ingesting vault: {VAULT_PATH}\n")
    chunks = ingest_vault(VAULT_PATH)
    print(f"\nTotal chunks to embed: {len(chunks)}")

    print(f"Embedding with '{EMBEDDING_MODEL}' and storing in '{CHROMA_PATH}'")
    collection = embed_and_store(chunks)

    print(f"\nCollection '{COLLECTION_NAME}' now has {collection.count()} chunks stored.")

    print("\n--- Sanity check query ---")
    test_query = "Whats the difference between a DMZ and Firewall?"
    print(test_query)
    results = collection.query(query_texts = [test_query], n_results = 2)
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\nResult {i + 1} (source: {meta['source']} > {meta['headers']}):")
        print(doc[:150] + "...")
