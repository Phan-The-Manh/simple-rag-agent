# Simple History-Aware RAG Chain (LangChain)

This repository implements a **history-aware Retrieval-Augmented Generation (RAG) system**
using **LangChain chains only** (no agents, no LangGraph).

The project is designed for **learning, experimentation, and deep understanding**
of how conversational RAG systems work internally.

---

## 🎯 Project Objectives

### Primary Targets
- Understand how **conversation history** should influence retrieval
- Implement **query rewriting** instead of naive history injection
- Build a **clean, debuggable RAG pipeline** using chains
- Avoid hidden abstractions (agents) to fully control data flow

### Learning Outcomes
After working with this project, you should be able to:
- Explain why history should not be passed directly to a retriever
- Implement conversational RAG via query rewriting
- Understand how LangChain runnables pass and transform data
- Diagnose common RAG failure modes (weak retrieval, schema hallucination)

---

## 🧠 Core Techniques Used

### 1. History-Aware Query Rewriting
Conversation history is used **only** to rewrite the user query into a
standalone, unambiguous search query before retrieval.

> History → Rewrite Query → Retrieve  
> NOT  
> History → Retriever

This keeps retrieval semantic, efficient, and scalable.

---

### 2. Vector-Based Retrieval (FAISS)
- FAISS is used as the vector store
- OpenAI embeddings (`text-embedding-3-small`)
- Top-k similarity search

---

### 3. Structured LLM Output
The final answer is constrained using a `TypedDict` schema to enforce:
- predictable output shape
- easier downstream usage
- clearer debugging

---

### 4. Chain-Only Architecture
This project intentionally avoids:
- Agents
- LangGraph
- Hidden memory abstractions

All logic is explicit and inspectable.

---


