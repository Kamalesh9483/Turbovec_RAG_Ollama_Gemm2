import time
import numpy as np

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.vector_stores.simple import SimpleVectorStore
from llama_index.core.vector_stores import SimpleVectorStore

from turbovec import IdMapIndex
from turbovec.llama_index import TurboQuantVectorStore
import ollama


# -----------------------------
# LLM SETUP (CPU ONLY)
# -----------------------------
Settings.llm = Ollama(
    model="gemma:2b",
    request_timeout=120.0,
    additional_kwargs={"options": {"num_gpu": 0}}
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading document...")
documents = SimpleDirectoryReader(
    input_files=[r"data\Quantum_Computing_Overview.txt"]
).load_data()

questions = [
    "What is a qubit and how is it different from a classical bit?",
    "What is quantum superposition?",
]


# =========================================================
# 1. BASELINE INDEX (NO TURBOVEC)
# =========================================================
print("\n================ BASELINE (Standard Vector Store) ================\n")

baseline_storage = StorageContext.from_defaults(
    vector_store=SimpleVectorStore()
)

t0 = time.time()

baseline_index = VectorStoreIndex.from_documents(
    documents,
    storage_context=baseline_storage
)

baseline_build_time = time.time() - t0

baseline_retriever = baseline_index.as_retriever(similarity_top_k=3)

baseline_retrieval_times = []
baseline_llm_times = []
baseline_total_times = []

for q in questions:
    print("=" * 80)
    print(f"[BASELINE] Q: {q}")

    t1 = time.time()
    nodes = baseline_retriever.retrieve(q)
    t2 = time.time()

    context = "\n\n".join([n.text for n in nodes])

    t3 = time.time()
    response = ollama.chat(
        model="gemma:2b",
        options={"num_gpu": 0},
        messages=[{
            "role": "user",
            "content": f"Answer using context:\n{context}\n\nQ: {q}"
        }]
    )
    t4 = time.time()

    print(response["message"]["content"])

    baseline_retrieval_times.append(t2 - t1)
    baseline_llm_times.append(t4 - t3)
    baseline_total_times.append(t4 - t1)


# =========================================================
# 2. TURBOVEC INDEX (COMPRESSED)
# =========================================================
print("\n================ TURBOVEC (4-bit Compression) ================\n")

tq_index = IdMapIndex(dim=384, bit_width=4)

vector_store = TurboQuantVectorStore(index=tq_index)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

t0 = time.time()

tq_index_llama = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

turbovec_build_time = time.time() - t0

retriever = tq_index_llama.as_retriever(similarity_top_k=3)

turbovec_retrieval_times = []
turbovec_llm_times = []
turbovec_total_times = []

for q in questions:
    print("=" * 80)
    print(f"[TURBOVEC] Q: {q}")

    t1 = time.time()
    nodes = retriever.retrieve(q)
    t2 = time.time()

    context = "\n\n".join([n.text for n in nodes])

    t3 = time.time()
    response = ollama.chat(
        model="gemma:2b",
        options={"num_gpu": 0},
        messages=[{
            "role": "user",
            "content": f"Answer using context:\n{context}\n\nQ: {q}"
        }]
    )
    t4 = time.time()

    print(response["message"]["content"])

    turbovec_retrieval_times.append(t2 - t1)
    turbovec_llm_times.append(t4 - t3)
    turbovec_total_times.append(t4 - t1)


# =========================================================
# 3. METRICS COMPARISON
# =========================================================
print("\n\n================ FINAL COMPARISON REPORT ================\n")

def summarize(name, arr):
    return {
        "avg": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

baseline_stats = {
    "retrieval": summarize("baseline", baseline_retrieval_times),
    "llm": summarize("baseline", baseline_llm_times),
    "total": summarize("baseline", baseline_total_times),
}

turbovec_stats = {
    "retrieval": summarize("turbovec", turbovec_retrieval_times),
    "llm": summarize("turbovec", turbovec_llm_times),
    "total": summarize("turbovec", turbovec_total_times),
}

print(" BUILD TIME")
print(f"Baseline build time : {baseline_build_time:.4f} sec")
print(f"TurboVec build time : {turbovec_build_time:.4f} sec\n")

print(" RETRIEVAL TIME (avg)")
print(f"Baseline  : {baseline_stats['retrieval']['avg']:.4f} sec")
print(f"TurboVec  : {turbovec_stats['retrieval']['avg']:.4f} sec\n")

print(" LLM TIME (avg)")
print(f"Baseline  : {baseline_stats['llm']['avg']:.4f} sec")
print(f"TurboVec  : {turbovec_stats['llm']['avg']:.4f} sec\n")

print(" TOTAL PIPELINE TIME (avg)")
print(f"Baseline  : {baseline_stats['total']['avg']:.4f} sec")
print(f"TurboVec  : {turbovec_stats['total']['avg']:.4f} sec\n")


# =========================================================
# 4. STORAGE APPROXIMATION
# =========================================================
def estimate_memory(index, label):
    try:
        size = len(str(index)) / (1024 * 1024)  # rough proxy
    except:
        size = 0
    print(f"{label} approx index memory footprint: {size:.4f} MB")

print("\n STORAGE ESTIMATION")
estimate_memory(baseline_index, "Baseline")
estimate_memory(tq_index_llama, "TurboVec")


# =========================================================
# 5. SPEEDUP SUMMARY
# =========================================================
print("\n SPEEDUP SUMMARY")

retr_speedup = baseline_stats['retrieval']['avg'] / turbovec_stats['retrieval']['avg']
total_speedup = baseline_stats['total']['avg'] / turbovec_stats['total']['avg']

print(f"Retrieval speedup : {retr_speedup:.2f}x")
print(f"Total pipeline speedup : {total_speedup:.2f}x")