"""
Quick Diagnostic Script for Low F1 Scores

Run this to understand where your system is failing.
"""

import pandas as pd
import json
from pyserini.search import SimpleSearcher

# Configuration
TRAIN_PATH = "train.csv"  # Update this path
NUM_SAMPLES = 20

def diagnose_retrieval(train_path, n_samples=20, k=15, mu=1000):
    """
    Diagnose retrieval quality by checking if answers are in retrieved documents.
    """
    print("=" * 80)
    print("RETRIEVAL DIAGNOSIS")
    print("=" * 80)

    # Load data
    df_train = pd.read_csv(train_path, converters={"answers": json.loads})

    # Load searcher
    searcher = SimpleSearcher.from_prebuilt_index('wikipedia-kilt-doc')
    searcher.set_qld(mu=mu)

    # Sample questions
    sample = df_train.sample(n=min(n_samples, len(df_train)), random_state=42)

    # Statistics
    stats = {
        'total': len(sample),
        'answer_in_docs': 0,
        'answer_not_in_docs': 0,
        'avg_docs_to_find': [],
    }

    print(f"\nTesting {len(sample)} questions with k={k}, mu={mu}\n")

    for idx, row in sample.iterrows():
        question = row['question']
        answers = row['answers']

        print(f"\nQ{idx}: {question}")
        print(f"Expected: {answers}")

        # Simple query (just remove ?)
        query = question.rstrip('?')

        # Retrieve
        hits = searcher.search(query, k)

        # Check if any answer is in any document
        found = False
        found_at = -1

        for i, hit in enumerate(hits):
            doc = searcher.doc(hit.docid)
            data = json.loads(doc.raw())
            content = data['contents'].lower()
            title = data['title'].lower()

            # Check if any answer appears
            for ans in answers:
                ans_lower = ans.lower()
                if ans_lower in content or ans_lower in title:
                    found = True
                    found_at = i + 1
                    print(f"  ✓ Found '{ans}' in doc #{found_at}: {data['title']}")
                    break

            if found:
                break

        if found:
            stats['answer_in_docs'] += 1
            stats['avg_docs_to_find'].append(found_at)
        else:
            stats['answer_not_in_docs'] += 1
            print(f"  ✗ Answer NOT in top {k} documents!")
            print(f"    → First doc: {data['title'][:60]}...")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = stats['total']
    found = stats['answer_in_docs']
    not_found = stats['answer_not_in_docs']

    print(f"\nTotal questions: {total}")
    print(f"Answer found in documents: {found} ({found/total*100:.1f}%)")
    print(f"Answer NOT found: {not_found} ({not_found/total*100:.1f}%)")

    if stats['avg_docs_to_find']:
        avg_pos = sum(stats['avg_docs_to_find']) / len(stats['avg_docs_to_find'])
        print(f"Average position when found: {avg_pos:.1f}")

    print("\nINTERPRETATION:")

    retrieval_rate = found / total * 100

    if retrieval_rate < 40:
        print("  ⚠️  CRITICAL: Only {:.0f}% of answers are in retrieved documents!".format(retrieval_rate))
        print("  → Problem: RETRIEVAL is the main issue")
        print("  → Solution:")
        print("     1. Increase k (try k=25 or k=30)")
        print("     2. Use less aggressive query preprocessing")
        print("     3. Try different mu values (500, 1500, 2000)")

    elif retrieval_rate < 60:
        print("  ⚠️  MODERATE: {:.0f}% of answers in documents".format(retrieval_rate))
        print("  → Retrieval needs improvement")
        print("  → Solution: Increase k and optimize query processing")

    elif retrieval_rate < 80:
        print("  ✓ GOOD: {:.0f}% of answers in documents".format(retrieval_rate))
        print("  → Retrieval is working reasonably well")
        print("  → Focus on improving EXTRACTION (prompts, LLM)")

    else:
        print("  ✓✓ EXCELLENT: {:.0f}% of answers in documents".format(retrieval_rate))
        print("  → Retrieval is working very well!")
        print("  → Focus on EXTRACTION and POST-PROCESSING")

    print("\nRECOMMENDED NEXT STEPS:")

    if retrieval_rate < 60:
        print("  1. Increase k to 25-30")
        print("  2. Keep original query (less preprocessing)")
        print("  3. Test different mu values")
        print("  4. Expected F1 improvement: +10-20%")
    else:
        print("  1. Optimize prompts for better extraction")
        print("  2. Lower temperature (try 0.1)")
        print("  3. Increase context length per document")
        print("  4. Expected F1 improvement: +5-15%")

    return stats


def check_query_variations(question, k=10):
    """
    Show what different query preprocessing strategies retrieve.
    """
    import re

    print("=" * 80)
    print("QUERY VARIATION ANALYSIS")
    print("=" * 80)
    print(f"\nOriginal: {question}\n")

    searcher = SimpleSearcher.from_prebuilt_index('wikipedia-kilt-doc')
    searcher.set_qld(mu=1000)

    # Different query strategies
    queries = {
        'original': question.rstrip('?'),
        'lowercase': question.lower().rstrip('?'),
        'no_question_words': None,
        'keywords_only': None,
    }

    # No question words
    pattern = r'^(what|where|when|who|which|how)\s+(is|are|was|were|does|do|did)\s+(the\s+)?(.+)$'
    match = re.match(pattern, question.lower())
    if match:
        queries['no_question_words'] = match.group(4).strip()

    # Keywords only
    stop_words = {'what', 'where', 'when', 'who', 'which', 'how', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'the', 'a', 'an', 'of'}
    tokens = question.lower().rstrip('?').split()
    keywords = [t for t in tokens if t not in stop_words]
    queries['keywords_only'] = ' '.join(keywords)

    # Remove None values
    queries = {k: v for k, v in queries.items() if v}

    # Test each query
    for name, query in queries.items():
        print(f"\n{name.upper()}: '{query}'")
        print("-" * 80)

        hits = searcher.search(query, k)

        for i, hit in enumerate(hits[:3], 1):  # Show top 3
            doc = searcher.doc(hit.docid)
            data = json.loads(doc.raw())
            title = data['title']
            content = data['contents'][:150]

            print(f"{i}. {title} (score: {hit.score:.2f})")
            print(f"   {content}...")

    print("\n" + "=" * 80)
    print("Which query strategy looks best?")
    print("Use that strategy in your expand_query function!")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python diagnose_retrieval.py <train.csv> [k] [mu]")
        print()
        print("Example:")
        print("  python diagnose_retrieval.py train.csv")
        print("  python diagnose_retrieval.py train.csv 20 1000")
        print()
        print("This will:")
        print("  1. Check if answers are in retrieved documents")
        print("  2. Identify whether retrieval or extraction is the problem")
        print("  3. Suggest specific improvements")
        sys.exit(1)

    train_path = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    mu = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    # Run diagnosis
    stats = diagnose_retrieval(train_path, n_samples=20, k=k, mu=mu)

    # Optional: Test different query strategies on one example
    print("\n\n")
    response = input("Test query variations on an example? (y/n): ")
    if response.lower() == 'y':
        df = pd.read_csv(train_path, converters={"answers": json.loads})
        example = df.sample(1, random_state=42).iloc[0]
        check_query_variations(example['question'], k=10)
