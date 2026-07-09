# Project Scope - Obsidian RAG

## Purpose

An agentic RAG (Retrieval-Augmented Generation) assistant that answers questions based on notes from my personal Obsidian vault. 

## Goals

- Build a working RAG pipeline over real personal data
- Move beyond a static retrieve then answer chain into an **agentic** flow using LangGraph
- Understand and be able to explain the engineering trade offs involved (RAG vs fine-tuning, local vs API embeddings, data boundary concerns for enterprise deployment)
- Produce something genuinely useful day to day (querying my own notes) rather than another unused project.

## Non-goals

- Not fine-tuning or training a model
- Not building a general-purpose/multi-user product,  single-user, local-first tool
- Not optimising for scale
- Not handling non-Obsidian note formats

## Scope

**In scope:**
- Ingesting `.md` files from an Obsidian vault, chunked in a way that respects note/header structure
- Local embedding + persistent vector storage (Chroma)
- Retrieval + generation via an LLM API (Claude/OpenAI)
- Agentic orchestration via LangGraph (e.g. retrieve -> grade relevance -> answer, or retrieve -> answer -> self-check)
- A minimal API layer (FastAPI/Flask `/ask` endpoint) so it's a service, not just a script
- Explicit testing of failure cases (e.g. asking something not covered in the notes, confirming it says "I don't know" rather than hallucinating)

## Roadmap

1. Ingest - walk vault, chunk `.md` files respecting header structure
2. Embed + store - local embeddings into a persistent Chroma DB
3. Retrieve + answer - working end-to-end script
4. Wrap in LangGraph - make it agentic
5. Expose via API - FastAPI/Flask `/ask` endpoint
