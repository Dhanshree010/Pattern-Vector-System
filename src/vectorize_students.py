import os
import sys

# Ensure root directory is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import numpy as np
from src.vectorizers.tabular_vectorizer import TabularVectorizer
from src.metrics.distance_metrics import DistanceMetrics
from src.metrics.similarity_engine import SimilarityRetrievalEngine

# Legacy compatibility wrapper class
class StudentVectorizer:
    """Backward-compatible wrapper around TabularVectorizer."""
    def __init__(self):
        self._vectorizer = TabularVectorizer(
            id_col='Student_ID',
            scaler_type='minmax'
        )

    def fit_transform(self, df):
        return self._vectorizer.fit_transform(df)

    def get_feature_names(self):
        return self._vectorizer.get_feature_names()

def find_similar_students(vectors, student_ids, query_index, top_n=3, metric='cosine'):
    """Backward-compatible helper function for student similarity search."""
    engine = SimilarityRetrievalEngine(vectors, identifiers=list(student_ids))
    metric_key = 'cosine_sim' if metric == 'cosine' else 'euclidean'
    raw_results = engine.query_top_k(vectors[query_index], top_k=top_n, metric=metric_key, exclude_index=query_index)
    return [{'Student_ID': r['identifier'], 'Score': r['score']} for r in raw_results]

if __name__ == "__main__":
    data_path = os.path.join(ROOT_DIR, 'data', 'student_records.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}")
        exit(1)
        
    df = pd.read_csv(data_path)
    print("=================================================================")
    print("           DAY 1: TABULAR STUDENT PATTERN VECTORIZATION           ")
    print("=================================================================")
    print(f"Loaded {len(df)} Student Records.")
    print(df.head(5))
    print("-" * 65)
    
    vectorizer = StudentVectorizer()
    vectors = vectorizer.fit_transform(df)
    
    print(f"Vector Space Shape : {vectors.shape} ({vectors.shape[0]} samples x {vectors.shape[1]} features)")
    print("Feature Dimensions :")
    for i, f in enumerate(vectorizer.get_feature_names()):
        print(f"  [{i:02d}] {f}")
        
    print(f"\nSample Vector for Student '{df['Student_ID'].iloc[0]}':")
    print(np.round(vectors[0], 4))
    print("-" * 65)
    
    query_idx = 0
    query_student = df.iloc[query_idx]
    print(f"Query Student: {query_student['Student_ID']} (Major: {query_student['Major']}, Grade: {query_student['Grade']})")
    
    print("\n--- Top 3 Similar Students by Cosine Similarity (Higher = Closer) ---")
    cos_matches = find_similar_students(vectors, df['Student_ID'].values, query_idx, top_n=3, metric='cosine')
    for m in cos_matches:
        rec = df[df['Student_ID'] == m['Student_ID']].iloc[0]
        print(f"  * {m['Student_ID']} | Major: {rec['Major']:<22} | Grade: {rec['Grade']:<3} | Cosine Sim: {m['Score']:.4f}")
        
    print("\n--- Top 3 Similar Students by Euclidean Distance (Lower = Closer) ---")
    euc_matches = find_similar_students(vectors, df['Student_ID'].values, query_idx, top_n=3, metric='euclidean')
    for m in euc_matches:
        rec = df[df['Student_ID'] == m['Student_ID']].iloc[0]
        print(f"  * {m['Student_ID']} | Major: {rec['Major']:<22} | Grade: {rec['Grade']:<3} | Euclidean Dist: {m['Score']:.4f}")
    print("=================================================================\n")
