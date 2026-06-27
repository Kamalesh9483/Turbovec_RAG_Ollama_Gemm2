<img width="1852" height="877" alt="image" src="https://github.com/user-attachments/assets/3765e816-7549-4af7-8ec9-22efdbf615ba" />

```markdown
# ⚡ TurboVec RAG Benchmarking with Ollama + LlamaIndex

## 🧠 Overview

This project benchmarks the performance of a Retrieval-Augmented Generation (RAG) pipeline using:

- **Baseline vector store (SimpleVectorStore)**
- **TurboVec compressed vector store (4-bit quantization)**

It evaluates improvements in:
- Retrieval speed
- LLM inference latency
- End-to-end pipeline performance
- Storage efficiency

The system uses:
- LlamaIndex for RAG orchestration
- HuggingFace embeddings (`all-MiniLM-L6-v2`)
- Ollama local LLM (`Gemma 2B`)
- TurboVec for compressed vector search

---

# 🏗️ Architecture

```

Document
↓
Embedding Model (MiniLM)
↓
Vector Store
├── Baseline (SimpleVectorStore)
└── TurboVec (4-bit compressed index)
↓
Retrieval (Top-K)
↓
LLM (Ollama Gemma 2B)
↓
Final Answer

```

---

# 🚀 Key Features

## 📊 1. Performance Benchmarking
Compares baseline vs TurboVec on:
- Index build time
- Retrieval latency
- LLM inference time
- Total pipeline latency

---

## 🧠 2. TurboVec Compression
- 4-bit quantized embeddings
- Memory-efficient vector storage
- Faster similarity search due to reduced memory bandwidth

---

## ⚡ 3. Speedup Analysis
Automatically calculates:
- Retrieval speedup
- End-to-end pipeline speedup
- Latency improvement per query

---

## 📦 4. Storage Estimation
- Approximate index memory footprint comparison
- Compression impact analysis (float32 vs 4-bit)

---

# 📁 Project Structure

```

.
├── app_3.py              # Main benchmarking script (baseline vs TurboVec)
├── data/
│   └── Quantum_Computing_Overview.txt
├── README.md
└── requirements.txt

````

---

# ⚙️ Installation

```bash
pip install llama-index
pip install llama-index-core
pip install sentence-transformers
pip install ollama
pip install turbovec
````

---

# ▶️ How to Run

## 1. Start Ollama model

```bash
ollama run gemma:2b
```

> Ensure CPU mode if needed:

```python
"options": {"num_gpu": 0}
```

---

## 2. Run Benchmark

```bash
python app_3.py
```

---


# 🧠 Why TurboVec Improves Performance

## 1. Memory Compression

* 4-bit quantization reduces embedding size significantly
* Better CPU cache utilization

## 2. Faster Retrieval

* Lower memory bandwidth usage
* Faster similarity search

## 3. Efficient Context Usage

* More relevant retrieval results
* Smaller context passed to LLM
* Lower token inference cost

---

