import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from anthropic import Anthropic

load_dotenv()

CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "obsidian_notes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-5"
TOP_K = 4

SYSTEM_PROMT = """
You are a helpful assistant answering questions using ONLY the context provided below, which is drawn from the user's personal cybersecurity notes.
 
Rules:
- Answer only using information found in the context below.
- If the answer isn't in the context, say clearly that it isn't covered in the notes. Do not guess or use outside knowledge.
- Keep answers concise and direct.
"""

def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path = CHROMA_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name = EMBEDDING_MODEL
    )

    return client.get_collection(name = COLLECTION_NAME, embedding_function = embedding_fn)

def retrieve(collection: chromadb.Collection, question: str, top_k: int = TOP_K) -> list[dict]:
    results = collection.query(query_texts = [question], n_results = top_k)

    chunks = []

    for doc, meta in zip (results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "headers": meta.get("headers", "")
        })
    return chunks

def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []

    for c in chunks:
        label = f"[{c['source']}" + (f" > {c['headers']}]" if c["headers"] else "]")
        context_blocks.append(f"{label}\n{c['text']}")

    context = "\n\n---\n\n".join(context_blocks)

    return f"Context:\n{context}\n\nQuestion: {question}"

def answer_question(question: str, client: Anthropic, collection: chromadb.Collection) -> tuple[str, list[str]]:
    chunks = retrieve(collection, question)
    prompt = build_prompt(question, chunks)

    res = client.messages.create(
        model = CLAUDE_MODEL,
        max_tokens = 500,
        system = SYSTEM_PROMT,
        messages = [{"role": "user", "content": prompt}]
    )

    answer_text = res.content[0].text
    sources = sorted(set(c["source"] for c in chunks))

    return answer_text, sources

if __name__ == "__main__":
    api_key = os.getenv("ANTHROPIC_KEY")

    if not api_key:
        raise ValueError("Claude API key not set in .env file")
    
    client = Anthropic(api_key = api_key)
    collection = get_collection()

    print("Obsidian Rag - Ask any question about your vault! (Ctrl + c to quit)\n>")
    while True:
        try:
            question = input("Q: ").strip()
        except(KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not question:
            continue

        answer, sources = answer_question(question, client, collection)
        print(f"\nA: {answer}")
        print(f"\nSources: {', '.join(sources)}\n")
    

    
    