"""
Walk the obsidian vault, read every .md file and split each note
into chunks along its header structure
"""

import os
import random
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

VAULT_PATH = os.getenv("OBS_LOC")

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3")
]

MAX_CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def find_markdown_files(vault_path: str) -> list[Path]:
    vault = Path(vault_path)
    if not vault.exists():
        raise FileNotFoundError(f"Vault path does not exist: {vault_path}.")
    return sorted(vault.rglob("*.md"))

def chunk_note(text: str, source_name: str) -> list[dict]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on = HEADERS_TO_SPLIT_ON,
        strip_headers = False
    )
    header_chunks = header_splitter.split_text(text)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size = MAX_CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    )

    final_chunks = []

    for doc in header_chunks:
        sub_chunks = char_splitter.split_text(doc.page_content)
        for i, sub in enumerate(sub_chunks):
            final_chunks.append({
                "text": sub,
                "source": source_name,
                "headers": doc.metadata
            })
    return final_chunks

def ingest_vault(vault_path: str) -> list[dict]:
    files = find_markdown_files(vault_path)
    if not files:
        print(f"no .md files found in {vault_path}")
        return []
    
    all_chunks = []
    for f in files:
        text = f.read_text(encoding = "utf-8")
        note_name = f.stem
        chunks = chunk_note(text, note_name)
        all_chunks.extend(chunks)
        print(f"{note_name}: {len(chunks)} chunk(s)")

    return all_chunks

if __name__ == "__main__":
    if not VAULT_PATH:
        raise ValueError("Vault path not set in .env file.")
    
    print(f"Ingesting Vault: {VAULT_PATH}\n")
    chunks = ingest_vault(VAULT_PATH)
    print(f"\nTotal chunks: {len(chunks)}")

    if chunks:
        print("\n--- Example chunk ---")
        example = random.choice(chunks)
        print(f"Source: {example['source']}")
        print(f"Headers: {example['headers']}")
        print(f"Text:\n{example['text']}")
