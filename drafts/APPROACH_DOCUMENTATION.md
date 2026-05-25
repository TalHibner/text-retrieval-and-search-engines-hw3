# Improved RAG System - Technical Documentation

## Executive Summary

This document describes the enhanced Retrieval-Augmented Generation (RAG) system developed for the question-answering competition. The baseline system achieved an F1 score of ~11.6%. Through systematic improvements in retrieval, prompting, and post-processing, this solution aims to significantly boost performance.

## Problem Analysis

### Baseline System Issues
1. **Poor Retrieval Quality**: Basic BM25 with default parameters retrieved irrelevant documents
2. **Suboptimal Queries**: Question format queries (e.g., "what is...") don't match document text well
3. **Weak Prompt Design**: Generic prompts led to verbose, imprecise answers
4. **No Answer Normalization**: Raw LLM outputs contained unnecessary text
5. **Limited Context**: Only 5 documents @ 300 chars each = insufficient information

### Key Constraints
- **Fixed LLM**: Llama-3.2-1B-Instruct (cannot be changed)
- **Fixed Corpus**: Wikipedia-KILT (cannot be modified)
- **Fixed Index**: Pyserini prebuilt index (cannot be changed)
- **Allowed Modifications**: Retrieval strategy, prompt engineering, post-processing

## Solution Architecture

### 1. Query Processing & Expansion

#### Query Preprocessing
```python
def preprocess_query(query):
    # Remove verbose question patterns
    # "what is the name of X" → "name of X"
    # "where is X located" → "location of X"
```

**Rationale**: Wikipedia articles use declarative statements, not questions. Converting queries to declarative form improves lexical matching.

#### Query Expansion
```python
def expand_query(query):
    # Generate multiple query variations:
    # 1. Original query
    # 2. Keyword-only version (remove question words)
```

**Example**:
- Original: "what is the name of justin bieber brother?"
- Expanded: "name justin bieber brother"

**Benefit**: Retrieves documents using multiple search strategies, increasing recall.

### 2. Advanced Retrieval Strategy

#### Multi-Query Retrieval with Fusion
```python
def get_context_improved(query, k=10, mu=1000):
    # For each query variation:
    #   1. Retrieve top-k documents
    #   2. Deduplicate by document ID
    #   3. Boost scores for docs appearing in multiple retrievals
    # Return top-k by final score
```

**Key Features**:
- **Query Fusion**: Documents matching multiple query variations rank higher
- **Score Aggregation**: Primary score + 0.5 × secondary scores
- **Deduplication**: Prevents redundant context
- **Title Inclusion**: Prepends document title to content for better context

**Parameter Tuning**:
- `k=10`: Retrieve more documents than baseline (5) for better coverage
- `mu=1000`: Dirichlet smoothing parameter (tested: 500, 1000, 2000)

#### Context Formatting
- Include document titles: `"Title: content"`
- Extend content length: 300 chars → 500 chars
- Clear document separation with delimiters

### 3. Prompt Engineering

#### Strategy 1: Direct Extraction (Default)
```
System: You are a precise question-answering assistant.
        Return ONLY the answer, nothing else.
        Do not add explanations or extra words.

User: Documents:
      [context]

      Question: [query]
      Answer (only the answer, nothing else):
```

**Rationale**: Llama-3.2-1B-Instruct is a small model. Simple, directive prompts work better than complex instructions.

#### Strategy 2: Entity-Focused
```
System: Extract the specific entity, name, date, or fact.
        Return only that entity with no additional text.
```

**Use case**: Better for factoid questions (names, dates, places)

#### Strategy 3: Few-Shot
```
System: Examples:
        Q: What is the capital of France? A: Paris
        Q: Who wrote Hamlet? A: William Shakespeare
```

**Use case**: Helps model understand desired answer format

#### Optimization Parameters
- **Temperature**: 0.1-0.5 (lower = more focused, tested 0.1, 0.3, 0.5)
- **Max Tokens**: 100 (prevent overly long answers)
- **Top-p**: 0.9 (nucleus sampling for diversity)

### 4. Answer Post-Processing

```python
def post_process_answer(answer, question):
    # 1. Remove common prefixes ("the answer is", "according to")
    # 2. Remove quotation marks
    # 3. Remove trailing punctuation (except abbreviations)
    # 4. Handle "unknown" responses
    # 5. Truncate overly long answers (>15 words)
```

**Examples**:
- Input: "The answer is Paris."
- Output: "Paris"

- Input: "According to the document, William Shakespeare"
- Output: "William Shakespeare"

**Benefit**: Matches evaluation format, improves F1 score by reducing spurious tokens.

### 5. Hyperparameter Tuning

#### Grid Search Strategy
Test on 100-sample training subset to find optimal:
- `k` (documents to retrieve): [5, 10, 15]
- `mu` (BM25 smoothing): [500, 1000, 2000]
- `prompt_version`: [1, 2, 3]
- `temperature`: [0.1, 0.3, 0.5]

#### Recommended Starting Configuration
```python
{
    'k': 10,
    'mu': 1000,
    'prompt_version': 1,
    'temperature': 0.3
}
```

**Justification**: Balances retrieval recall (k=10) with precision, standard BM25 smoothing, direct prompting, and moderate temperature.

## Implementation Details

### Workflow

1. **Input**: Question from test set
2. **Query Processing**:
   - Preprocess query
   - Generate variations
3. **Retrieval**:
   - Search with each query variation
   - Aggregate and deduplicate results
   - Select top-k by combined score
4. **Context Preparation**:
   - Format documents with titles
   - Truncate to appropriate length
5. **Prompt Construction**:
   - Select prompt template
   - Insert context and question
6. **LLM Generation**:
   - Generate answer with optimized parameters
7. **Post-Processing**:
   - Clean and normalize answer
8. **Output**: Final answer

### Evaluation Metrics

**F1 Score Calculation**:
```python
def f1_score(prediction, ground_truth):
    # 1. Normalize: lowercase, remove punctuation, remove articles
    # 2. Tokenize
    # 3. Calculate precision = common_tokens / pred_tokens
    # 4. Calculate recall = common_tokens / gt_tokens
    # 5. F1 = 2 * (precision * recall) / (precision + recall)
```

**Final Score**: Average F1 over all test questions (max F1 if multiple ground truths)

## Expected Improvements

### Baseline vs. Improved

| Component | Baseline | Improved | Expected Gain |
|-----------|----------|----------|---------------|
| Query Processing | Raw question | Preprocessed + expanded | +5-10% F1 |
| Retrieval | Single query, k=5 | Multi-query fusion, k=10 | +10-15% F1 |
| Prompt Design | Generic | Optimized directive | +5-10% F1 |
| Answer Processing | Raw output | Cleaned & normalized | +5-8% F1 |
| Context Length | 300 chars | 500 chars | +3-5% F1 |

**Estimated Total**: 28-48% F1 improvement → **40-60% F1 score**

### Why These Improvements Work

1. **Query Expansion**: Addresses vocabulary mismatch between questions and documents
2. **Score Fusion**: Documents relevant to multiple query aspects rank higher
3. **Direct Prompting**: Small LLMs (1B params) perform better with simple instructions
4. **Post-Processing**: Removes tokens that hurt F1 without losing information
5. **More Context**: More documents + longer excerpts = higher answer coverage

## Advanced Techniques (Optional Enhancements)

### If Time/Resources Permit:

#### 1. Pseudo-Relevance Feedback
```python
# Use top-1 document to expand query with relevant terms
def prf_expand(query, top_doc):
    # Extract key terms from top document
    # Add to query for second retrieval round
```

#### 2. Answer Re-ranking
```python
# Generate 3 answers with different prompts/temps
# Select best by self-consistency or confidence
```

#### 3. Dense Retrieval (Hybrid)
```python
# Combine BM25 with dense retrieval (SBERT)
# Requires loading additional model - check memory
```

#### 4. Document Passage Splitting
```python
# Split long documents into passages
# Retrieve at passage level for more precise context
```

#### 5. Query Classification
```python
# Classify question type (who/what/when/where)
# Use type-specific prompts
```

## Usage Instructions

### Setup (Google Colab)
1. Upload notebook to Colab
2. Mount Google Drive
3. Place train.csv and test.csv in Drive
4. Add Hugging Face token
5. Run all cells

### Local Execution
```bash
# Install dependencies
pip install pyserini==0.36.0 torch transformers pandas

# Java required for Pyserini
apt-get install openjdk-21-jdk

# Run notebook
jupyter notebook improved_rag_system.ipynb
```

### Adjusting Hyperparameters
Modify `final_params` dictionary:
```python
final_params = {
    'k': 10,              # Number of documents
    'mu': 1000,           # BM25 smoothing
    'prompt_version': 1,  # Prompt strategy
    'temperature': 0.3,   # LLM temperature
    'max_tokens': 100     # Max answer length
}
```

## Submission Format

Output CSV with columns:
- `id`: Question ID (integer)
- `prediction`: JSON string list with single answer (e.g., `["Paris"]`)

Example:
```csv
id,prediction
1,"[""Jamaican Patois""]"
2,"[""Speaker of the Tennessee House of Representatives""]"
```

## Key Insights & Lessons

1. **Small LLMs Need Simple Prompts**: Complex chain-of-thought doesn't work well with 1B models
2. **Retrieval Matters Most**: Better documents > better prompts
3. **Answer Format Critical**: Cleaning outputs significantly improves F1
4. **Query Understanding**: Questions ≠ documents; bridge the gap
5. **Hyperparameter Sensitivity**: BM25 mu and k have large impact

## References & Inspiration

- **Pyserini**: Anserini toolkit for information retrieval
- **KILT Benchmark**: Knowledge-Intensive Language Tasks
- **BM25**: Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond"
- **Query Expansion**: Carpineto & Romano, "A Survey of Automatic Query Expansion"
- **RAG**: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- **Prompt Engineering**: OpenAI, "GPT Best Practices"

## Contact & Questions

For questions about this implementation:
1. Check code comments in notebook
2. Review function docstrings
3. Experiment with hyperparameters on small sample

## License

Educational use only - HW3 Assignment, Reichman University
