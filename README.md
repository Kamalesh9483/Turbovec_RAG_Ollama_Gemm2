<img width="1852" height="877" alt="image" src="https://github.com/user-attachments/assets/3765e816-7549-4af7-8ec9-22efdbf615ba" />

# ⚡ TurboVec RAG Benchmarking with Ollama + LlamaIndex

## 🧠 Overview

This project benchmarks a Retrieval-Augmented Generation (RAG) system by comparing:

- 🔹 Baseline Vector Store (`SimpleVectorStore`)
- 🔹 TurboVec Optimized Store (4-bit quantized vectors)

It evaluates improvements in:
- Retrieval speed
- LLM inference latency
- End-to-end pipeline performance
- Storage efficiency

The system uses:
- LlamaIndex for RAG pipeline orchestration
- HuggingFace embeddings (`all-MiniLM-L6-v2`)
- Ollama local LLM (`Gemma 2B`)
- TurboVec for compressed vector storage

---

# 🏗️ Architecture

## 🔹 RAG Pipeline Flow
            ┌──────────────────────────┐
            │      Document Input      │
            └────────────┬─────────────┘
                         ↓
            ┌──────────────────────────┐
            │  Embedding Model (384D)  │
            │  MiniLM-L6-v2            │
            └────────────┬─────────────┘
                         ↓
            ┌──────────────────────────┐
            │     Vector Store         │
            └───────┬─────────┬────────┘
                    ↓         ↓
     ┌──────────────────┐  ┌──────────────────┐
     │ Baseline Store   │  │   TurboVec Store │
     │ SimpleVectorStore│  │  4-bit Quantized │
     └────────┬─────────┘  └────────┬─────────┘
              ↓                      ↓
     ┌──────────────────────────────────────┐
     │        Retrieval (Top-K = 3)        │
     └──────────────────┬───────────────────┘
                        ↓
            ┌──────────────────────────┐
            │   Ollama LLM (Gemma 2B)  │
            └────────────┬─────────────┘
                         ↓
            ┌──────────────────────────┐
            │      Final Answer        │
            └──────────────────────────┘


---

# 🚀 Key Features

## 📊 1. Benchmarking System
Compares baseline vs TurboVec on:
- Index build time
- Retrieval latency
- LLM inference time
- Total pipeline latency

---

## ⚡ 2. Speedup Analysis
Automatically computes:
- Retrieval speedup
- End-to-end speedup
- Latency improvement per query

---

## 🧠 3. TurboVec Compression
- 4-bit quantized embeddings
- Reduced memory footprint
- Faster similarity search
- Cache-friendly vector operations

---

## 📦 4. Storage Analysis
- Approximate index memory comparison
- Float32 vs 4-bit compression evaluation

---

# 📁 Project Structure

.
├── app_3.py # Main benchmarking script
├── data/
│ └── Quantum_Computing_Overview.txt
├── README.md
└── requirements.txt


---



