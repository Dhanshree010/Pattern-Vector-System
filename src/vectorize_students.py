import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import os

class StudentVectorizer:
    def __init__(self):
        # Setup column transformer for preprocessing
        # Age, Grade, Attendance_Rate -> MinMaxScale
        # Major -> OneHotEncode
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), ['Age', 'Grade', 'Attendance_Rate']),
                ('cat', OneHotEncoder(sparse_output=False), ['Major'])
            ])
        self.student_ids = None

    def fit_transform(self, df):
        """Fits the preprocessor and transforms the data into vectors."""
        self.student_ids = df['Student_ID'].values
        # Drop Student_ID for vectorization
        features_df = df.drop(columns=['Student_ID'])
        vectors = self.preprocessor.fit_transform(features_df)
        return vectors

    def get_feature_names(self):
        """Returns the names of the features in the vector."""
        return self.preprocessor.get_feature_names_out()

def find_similar_students(vectors, student_ids, query_index, top_n=3, metric='cosine'):
    """
    Finds the most similar students to the query_index using the specified metric.
    metric: 'cosine' or 'euclidean'
    """
    query_vector = vectors[query_index].reshape(1, -1)
    
    if metric == 'cosine':
        # Cosine similarity returns values between -1 and 1 (1 is most similar)
        similarities = cosine_similarity(query_vector, vectors)[0]
        # Sort indices descending
        best_indices = np.argsort(similarities)[::-1]
    elif metric == 'euclidean':
        # Euclidean distance returns values >= 0 (0 is most similar)
        distances = euclidean_distances(query_vector, vectors)[0]
        # Sort indices ascending
        best_indices = np.argsort(distances)
        similarities = distances # Just for return variable naming
    else:
        raise ValueError("Unsupported metric. Choose 'cosine' or 'euclidean'.")

    # Filter out the query index itself if it's the top one (it should be)
    results = []
    for idx in best_indices:
        if idx != query_index:
            results.append({
                'Student_ID': student_ids[idx],
                'Score': similarities[idx]
            })
            if len(results) >= top_n:
                break
    return results

if __name__ == "__main__":
    # Ensure correct working directory context
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'student_records.csv')
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}.")
        exit(1)

    print("--- Loaded Student Records ---")
    print(df.head())
    print("\n")

    # Vectorize
    vectorizer = StudentVectorizer()
    vectors = vectorizer.fit_transform(df)
    
    print("--- Vector Representations ---")
    print(f"Shape of vector space: {vectors.shape}")
    print(f"Features: {vectorizer.get_feature_names()}")
    print(f"Vector for {df['Student_ID'].iloc[0]}:\n{vectors[0]}")
    print("\n")

    # Similarity Analysis
    query_idx = 0 # Let's find students similar to the first student (S001)
    query_student = df.iloc[query_idx]
    print(f"--- Finding students similar to: {query_student['Student_ID']} ({query_student['Major']}, Grade: {query_student['Grade']}) ---")

    print("\nTop 3 by Cosine Similarity:")
    cosine_results = find_similar_students(vectors, df['Student_ID'].values, query_idx, top_n=3, metric='cosine')
    for res in cosine_results:
        # Fetch original record for display
        record = df[df['Student_ID'] == res['Student_ID']].iloc[0]
        print(f"- {res['Student_ID']} ({record['Major']}, Grade: {record['Grade']}) -> Score: {res['Score']:.4f}")

    print("\nTop 3 by Euclidean Distance (lower is more similar):")
    euclidean_results = find_similar_students(vectors, df['Student_ID'].values, query_idx, top_n=3, metric='euclidean')
    for res in euclidean_results:
        record = df[df['Student_ID'] == res['Student_ID']].iloc[0]
        print(f"- {res['Student_ID']} ({record['Major']}, Grade: {record['Grade']}) -> Distance: {res['Score']:.4f}")
