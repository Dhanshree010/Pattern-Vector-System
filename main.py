import os
import sys
import argparse
import subprocess

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import numpy as np
from src.vectorizers.tabular_vectorizer import TabularVectorizer
from src.vectorizers.image_vectorizer import ImageVectorizer
from src.vectorizers.text_vectorizer import TextVectorizer
from src.vectorizers.signal_vectorizer import SignalVectorizer
from src.metrics.distance_metrics import DistanceMetrics
from src.metrics.similarity_engine import SimilarityRetrievalEngine
from src.models.classifiers import PatternClassifierSuite
from src.models.clustering import PatternClusterSuite
from src.models.dimensionality import DimensionalityReducer
from src.generate_images import generate_all_images
from src.run_experiments import run_all_experiments

def run_multi_modal_demo():
    """Runs a complete end-to-end multi-modal pattern vectorization and similarity retrieval demo."""
    print("="*75)
    print("      OMNIPATTERN: MULTI-MODAL PATTERN RECOGNITION SYSTEM DEMO     ")
    print("="*75)
    
    # 1. Tabular Modality
    print("\n[MODALITY 1: TABULAR PATTERNS (Student Academic Profiles)]")
    df_students = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'student_records.csv'))
    tab_vec = TabularVectorizer(id_col='Student_ID', scaler_type='minmax')
    tab_vectors = tab_vec.fit_transform(df_students)
    print(f"Loaded {len(df_students)} student records -> Vectorized into {tab_vectors.shape[1]}-D space.")
    
    tab_engine = SimilarityRetrievalEngine(tab_vectors, identifiers=list(df_students['Student_ID']), metadata_df=df_students)
    top_students = tab_engine.query_top_k(tab_vectors[0], top_k=3, metric='cosine_sim', exclude_index=0)
    print(f"Top 3 Similar Students to '{df_students['Student_ID'].iloc[0]}' ({df_students['Major'].iloc[0]}):")
    for s in top_students:
        m = s['metadata']
        print(f"  * {s['identifier']} ({m['Major']}) | Grade: {m['Grade']} | Cosine Similarity: {s['score']:.4f}")

    # 2. Visual Modality
    print("\n[MODALITY 2: VISUAL PATTERNS (Geometric & Color Features)]")
    image_dir = os.path.join(ROOT_DIR, 'data', 'images')
    img_vec = ImageVectorizer(image_size=(64, 64), extraction_mode='hybrid')
    img_vectors = img_vec.transform(image_dir)
    img_names = [os.path.basename(p) for p in img_vec.image_paths_]
    print(f"Loaded {len(img_vectors)} images -> Extracted {img_vectors.shape[1]}-D Hybrid feature vectors.")
    
    img_engine = SimilarityRetrievalEngine(img_vectors, identifiers=img_names)
    top_images = img_engine.query_top_k(img_vectors[0], top_k=3, metric='cosine_sim', exclude_index=0)
    print(f"Top 3 Visual Matches to '{img_names[0]}':")
    for img in top_images:
        print(f"  * {img['identifier']:<24} -> Cosine Similarity: {img['score']:.4f}")

    # 3. Text Modality
    print("\n[MODALITY 3: TEXT & NLP PATTERNS (Course Syllabus & Abstracts)]")
    df_texts = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'academic_texts.csv'))
    txt_vec = TextVectorizer(mode='tfidf', ngram_range=(1, 2), max_features=40)
    txt_vectors = txt_vec.fit_transform(df_texts['Text'].tolist())
    print(f"Loaded {len(df_texts)} academic documents -> Extracted {txt_vectors.shape[1]}-D TF-IDF space.")
    
    sample_query = "neural networks, computer vision, and visual feature extraction"
    q_vec = txt_vec.transform(sample_query)[0]
    txt_engine = SimilarityRetrievalEngine(txt_vectors, identifiers=df_texts['Title'].tolist(), metadata_df=df_texts)
    top_texts = txt_engine.query_top_k(q_vec, top_k=2, metric='cosine_sim')
    print(f"Query: \"{sample_query}\"")
    for t in top_texts:
        print(f"  * {t['identifier']} ({t['metadata']['Domain']}) -> Score: {t['score']:.4f}")

    # 4. 1D Signal Modality
    print("\n[MODALITY 4: 1D TEMPORAL SIGNAL PATTERNS (Time & Frequency Features)]")
    sig_vec = SignalVectorizer(sample_rate=1000, n_fft_bins=8)
    _, sig_sine = SignalVectorizer.generate_synthetic_signal('sine', freq=20.0)
    sig_features = sig_vec.transform(sig_sine)[0]
    print(f"Extracted {len(sig_features)}-D Time-Frequency Vector from 20Hz Sine Wave.")
    print(f"Time-Domain Moments (Mean, Std, Skewness, Kurtosis): {np.round(sig_features[:4], 3)}")

    # 5. Multi-Metric Correlation Matrix
    print("\n[METRIC EVALUATION: Spearman Rank Correlation Matrix (Tabular Space)]")
    corr_df = tab_engine.compute_metric_correlation_matrix(tab_vectors[0])
    print(corr_df.round(3).to_string())

    print("\n" + "="*75)
    print("  Demo completed successfully! Run 'python main.py web' to launch Web Studio.")
    print("="*75 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="OmniPattern: Multi-Modal Pattern Recognition System (7th Sem PR Final Project)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Subcommands')
    
    # Demo command
    subparsers.add_parser('demo', help='Run end-to-end multi-modal demonstration')
    
    # Generate data command
    subparsers.add_parser('generate-data', help='Generate synthetic pattern images and datasets')
    
    # Experiments command
    subparsers.add_parser('experiments', help='Run full benchmark experiment suite and generate visualization plots')
    
    # Evaluate command
    subparsers.add_parser('evaluate', help='Run cross-validation classifier benchmarks')
    
    # Web command
    subparsers.add_parser('web', help='Launch Streamlit interactive web dashboard')
    
    # Test command
    subparsers.add_parser('test', help='Run unittest test suite')

    args = parser.parse_args()
    
    if args.command == 'demo' or args.command is None:
        run_multi_modal_demo()
    elif args.command == 'generate-data':
        generate_all_images()
    elif args.command == 'experiments':
        run_all_experiments()
    elif args.command == 'evaluate':
        df_students = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'student_records.csv'))
        tab_vec = TabularVectorizer(id_col='Student_ID')
        vectors = tab_vec.fit_transform(df_students)
        suite = PatternClassifierSuite()
        res = suite.cross_validate_all(vectors, df_students['Major'].values, n_splits=3)
        print("\n--- Pattern Classifier Cross-Validation Benchmark ---")
        print(res.to_string(index=False))
    elif args.command == 'web':
        app_path = os.path.join(ROOT_DIR, 'src', 'web', 'app.py')
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
    elif args.command == 'test':
        subprocess.run([sys.executable, "-m", "unittest", "discover", "tests"])
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
