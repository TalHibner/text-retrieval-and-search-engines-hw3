# Text Information Retrieval - HW3: RAG Question Answering System

## Overview

This repository contains an advanced Retrieval-Augmented Generation (RAG) system for answering questions using Wikipedia articles. The system retrieves relevant passages from a Wikipedia corpus and uses `Llama-3.2-1B-Instruct` to extract concise answers. Through systematic experimentation and innovative techniques like dynamic few-shot prompting, the system achieved a **32.80% F1 score**, placing in the **top 10** on the Kaggle leaderboard (a 2.8x improvement over the baseline of 11.6%).

**Student:** Tal Hibner  
**Course:** Text Information Retrieval, Reichman University  

## Key Innovation: Dynamic Few-Shot Prompting

The primary breakthrough was implementing **dynamic few-shot prompting** - an adaptive approach that selects the 5 most relevant examples from training data (using Jaccard similarity) for each test question and includes them in the prompt, leading to a significant performance boost.

## Files

- **`TemplateRAGAssignment_Upload.ipynb`**: Original baseline system (~11.6% F1)
- **`advanced_retrieval_methods.ipynb`**: Complete implementation of the enhanced system
- **`EXECUTIVE_SUMMARY.md`**: High-level summary of the approach, results, and insights
- **`FINAL_APPROACH_DOCUMENTATION.md`**: Detailed technical documentation of the architecture
- **`README.md`**: This file
- **`train.csv`**: Training data (3,778 questions with answers)
- **`test.csv`**: Test data (2,032 questions, no answers)
- **`advanced_retrieval_predictions.csv`**: Final Kaggle submission (32.80% score)

## Quick Start

### Option 1: Google Colab (Recommended)

1. Open `advanced_retrieval_methods.ipynb` in Google Colab
2. Update the Hugging Face token:
   ```python
   hugging_face_token = "YOUR_TOKEN_HERE"
   ```
3. Update file paths in the data loading cell:
   ```python
   train_path = "/content/drive/MyDrive/YOUR_PATH/train.csv"
   test_path = "/content/drive/MyDrive/YOUR_PATH/test.csv"
   ```
4. Run all cells
5. Download `advanced_retrieval_predictions.csv` from your Drive

### Option 2: Local Execution

Requirements:
- Python 3.8+
- CUDA GPU (recommended)
- 16GB+ RAM
- Java 11+ (for Pyserini)

```bash
# Install dependencies
pip install pyserini==0.36.0 torch transformers pandas sentence-transformers

# Install Java (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install openjdk-21-jdk

# Run notebook
jupyter notebook advanced_retrieval_methods.ipynb
```

## Key Components Over Baseline

1. **Query Processing & Multi-Strategy Retrieval**
   - BM25 with multiple configurations (mu=500, 1000, 2000)
   - Pseudo-Relevance Feedback and entity-focused queries
   - Reciprocal Rank Fusion to combine rankings

2. **Large Context & Reranking**
   - 5 documents × 1000 characters = 5,000 chars total context
   - Final reranking by query-document similarity

3. **Prompt Engineering & Dynamic Few-Shot**
   - Dynamic selection of 5 most similar training questions
   - Ultra-strict prompt instructions for the 1B model
   - Highly deterministic generation (temperature=0.01)

4. **Answer Post-Processing**
   - 90+ regex rules to clean LLM outputs (removing lists, verbose prefixes, extracting entities)
   - Truncation to 1-5 words

## Final Configuration

Optimal settings discovered through systematic testing:
```python
{
    'k': 15,               # Initial documents to retrieve before reranking/fusion
    'context_docs': 5,     # Final documents fed to LLM
    'chars_per_doc': 1000, # Characters per document
    'temperature': 0.01,   # LLM sampling temperature
    'max_tokens': 30       # Maximum answer length
}
```

## System Architecture

```
Input Question
    ↓
Query Preprocessing (Extract Entities)
    ↓
Multi-Strategy Retrieval (BM25, PRF)
    ↓
Document Fusion & Ranking (RRF)
    ↓
Dynamic Few-Shot Example Selection
    ↓
Prompt Construction
    ↓
LLM Generation (Llama-3.2-1B-Instruct)
    ↓
Aggressive Answer Post-Processing
    ↓
Final Answer
```

## Submission Format

The output CSV uses this format (with JSON lists containing a single answer string):
```csv
id,prediction
1,"[""Jamaican Patois""]"
2,"[""Speaker of the Tennessee House of Representatives""]"
```

### Critical Formatting Steps

**1. Loading train.csv (with answers):**
```python
import pandas as pd
import json

# MUST use converters to parse answers column as JSON
df_train = pd.read_csv("train.csv", converters={"answers": json.loads})
```

**2. Before saving predictions:**
```python
# MUST format predictions as JSON strings with ensure_ascii=False
df_prediction["prediction"] = df_prediction["prediction"].apply(
    lambda x: json.dumps([x], ensure_ascii=False)
)

# Then save to CSV
df_prediction.to_csv("advanced_retrieval_predictions.csv", index=False)
```

## Troubleshooting

### Java/Pyserini Errors
- Ensure Java 11+ is installed: `java -version`
- Set JAVA_HOME environment variable
- Reinstall pyserini: `pip install --force-reinstall pyserini==0.36.0`

## Evaluation Metrics

The competition uses token-level F1 score. 
- **Baseline F1**: ~11.6%
- **Final Kaggle F1**: 32.80% (Top 10)

## Resources

- [Pyserini Documentation](https://github.com/castorini/pyserini)
- [Llama Model Card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [KILT Benchmark](https://ai.facebook.com/tools/kilt/)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)

## License

Educational use only - Reichman University, Text Information Retrieval Course
