# RAG System for Question Answering: Technical Documentation

**Final Kaggle Score: 32.80%**  
**Rank: Top 10 (8th-10th place)**  
**Baseline Score: 11.6% → Final: 32.80% (2.8x improvement)**

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Key Components](#key-components)
4. [Retrieval Strategy](#retrieval-strategy)
5. [Dynamic Few-Shot Prompting](#dynamic-few-shot-prompting)
6. [Answer Generation & Post-Processing](#answer-generation--post-processing)
7. [Iterative Improvements](#iterative-improvements)
8. [Final Configuration](#final-configuration)
9. [Results & Analysis](#results--analysis)
10. [Key Insights](#key-insights)

---

## System Overview

This system implements an advanced Retrieval-Augmented Generation (RAG) pipeline for answering factual questions using the Wikipedia-KILT corpus. The approach combines:

- **Multi-strategy document retrieval** (BM25 with multiple configurations)
- **Dynamic few-shot prompting** (adaptive example selection)
- **Aggressive answer post-processing** (format normalization)
- **Large language model** (Llama-3.2-1B-Instruct)

The system achieved **32.80% F1 score** on the Kaggle leaderboard, representing a **2.8x improvement** over the baseline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Question                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Query Preprocessing                             │
│  - Remove question marks                                     │
│  - Extract entity terms (capitalized words)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          Multi-Strategy Document Retrieval                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. BM25 with Multiple Mu Values (500, 1000, 2000)   │   │
│  │ 2. Pseudo-Relevance Feedback (PRF)                  │   │
│  │ 3. Entity-Focused Queries                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│             Reciprocal Rank Fusion (RRF)                     │
│                           │                                  │
│                           ▼                                  │
│              Document Reranking (Similarity)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          Dynamic Few-Shot Example Selection                  │
│  - Find 5 most similar questions from train.csv             │
│  - Use Jaccard similarity + question type matching          │
│  - Adaptive to current question                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Prompt Construction                             │
│  - Few-shot examples (5 adaptive)                           │
│  - Document context (5 docs × 1000 chars)                   │
│  - Ultra-strict instructions                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         LLM Generation (Llama-3.2-1B-Instruct)              │
│  - Temperature: 0.01 (highly deterministic)                 │
│  - Max tokens: 30                                           │
│  - Top-p sampling: 0.95                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         Aggressive Answer Post-Processing                    │
│  - Remove list markers                                       │
│  - Remove verbose prefixes                                   │
│  - Extract entities from verbose responses                   │
│  - Truncate to 1-5 words                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Final Answer                                │
└─────────────────────────────────────────────────────────────┘
```

