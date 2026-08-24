# SpatialSearch — Tweet RAG Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) system built over 110,000+ tweets. The pipeline retrieves semantically relevant tweets for a query, reranks them with a cross-encoder, and generates a natural language summary using FLAN-T5.

## Pipeline

```
Query
  │
  ▼
[MiniLM bi-encoder] ──► FAISS index (cosine similarity) ──► top-50 candidates
                                                                    │
                                                                    ▼
                                                     [Cross-encoder reranker] ──► top-10
                                                                                      │
                                                                                      ▼
                                                                              [FLAN-T5-large] ──► Summary
```

## Notebooks (run in order)

| Notebook | What it does |
|---|---|
| `01_build_index.ipynb` | Encode 110k tweets with GloVe and MiniLM; build and save FAISS indexes |
| `02_retrieval_pipeline.ipynb` | Full retrieve → rerank → summarize pipeline with demo queries |
| `03_evaluation.ipynb` | Benchmark GloVe vs MiniLM on Precision@k and MRR |
| `04_visualization.ipynb` | UMAP projection + K-Means clustering of tweet embeddings |
| `05_gradio_demo.ipynb` | Interactive web UI deployable to HuggingFace Spaces |

## Tech Stack

| Component | Library / Model |
|---|---|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2`, `average_word_embeddings_glove.840B.300d` |
| Vector index | `faiss-cpu` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Summarizer | `google/flan-t5-large` |
| Visualization | `umap-learn`, `scikit-learn` (K-Means), `matplotlib` |
| Demo UI | `gradio` |

## Setup

All notebooks are designed for **Google Colab** (free tier). Each notebook installs its own dependencies via `!pip install`. The only external file needed is `tweets-utf-8.json` — place it in the same directory or at `/content/` on Colab.

Run `01_build_index.ipynb` first — it generates the FAISS indexes and embedding files that all other notebooks load from disk.
