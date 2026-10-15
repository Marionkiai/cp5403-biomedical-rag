# Evidence-grounded biomedical question answering with hybrid RAG

CP5403 Assessment 2 (LLM project).

A retrieval augmented generation (RAG) system that answers PubMedQA research questions (yes / no / maybe) with an open-weights LLM (Qwen2.5-3B-Instruct), grounded in retrieved PubMed abstracts.

## Pipeline
1. **Corpus:** 1,000 labelled PubMedQA abstracts plus 5,000 distractor abstracts.
2. **Retrieval:** BM25 and dense embeddings (bge-small-en-v1.5), fused with Reciprocal Rank Fusion, then reranked by a cross-encoder (ms-marco-MiniLM-L-6-v2).
3. **Generation:** Qwen2.5-3B-Instruct, greedy decoding, one-word answer.
4. **Evaluation:** closed-book, BM25, dense, hybrid, hybrid + rerank and an oracle upper bound, on 200 questions. Metrics: Recall@k, MRR, accuracy, macro-F1, invalid-answer rate, latency.

## Run it
Open `rag_pubmedqa.ipynb` in Google Colab, choose Runtime > Change runtime type > T4 GPU, then Runtime > Run all (about 20-30 minutes).

Outputs are written to `results/` (CSV tables, chart, predictions).

## Data and models
* PubMedQA (Jin et al., 2019): https://huggingface.co/datasets/qiaojin/PubMedQA
* Models are downloaded from Hugging Face at run time.

*Research prototype only. Not medical advice.*
