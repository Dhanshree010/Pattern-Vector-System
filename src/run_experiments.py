import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.vectorizers.tabular_vectorizer import TabularVectorizer
from src.vectorizers.image_vectorizer import ImageVectorizer
from src.vectorizers.text_vectorizer import TextVectorizer
from src.vectorizers.signal_vectorizer import SignalVectorizer
from src.metrics.distance_metrics import DistanceMetrics
from src.metrics.similarity_engine import SimilarityRetrievalEngine
from src.models.classifiers import PatternClassifierSuite
from src.models.clustering import PatternClusterSuite
from src.models.dimensionality import DimensionalityReducer

def run_all_experiments():
    """
    Automated Benchmark & Visualization Experiment Suite for OmniPattern.
    Executes multi-modal feature vectorization, metric correlation analysis,
    classification benchmarks, and clustering projections, saving all artifacts.
    """
    output_dir = os.path.join(ROOT_DIR, 'data', 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    print("\n" + "="*70, flush=True)
    print("      OMNIPATTERN: MULTI-MODAL EXPERIMENT & BENCHMARK SUITE       ", flush=True)
    print("="*70, flush=True)
    
    # -------------------------------------------------------------
    # 1. TABULAR PATTERN EXPERIMENTS (Student Profiles)
    # -------------------------------------------------------------
    print("\n[1/5] Running Tabular Pattern Experiments...", flush=True)
    student_csv = os.path.join(ROOT_DIR, 'data', 'student_records.csv')
    df_students = pd.read_csv(student_csv)
    
    tab_vec = TabularVectorizer(id_col='Student_ID', scaler_type='minmax')
    tab_vectors = tab_vec.fit_transform(df_students)
    print(f"  -> Vectorized {tab_vectors.shape[0]} student profiles into {tab_vectors.shape[1]}-D space.", flush=True)
    
    # Tabular Metric Rank Correlation
    tab_engine = SimilarityRetrievalEngine(tab_vectors, identifiers=list(df_students['Student_ID']))
    tab_corr = tab_engine.compute_metric_correlation_matrix(tab_vectors[0])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(tab_corr, annot=True, cmap="YlGnBu", fmt=".3f", cbar=True, ax=ax, linewidths=.5)
    ax.set_title("Metric Spearman Rank Correlation Matrix (Tabular Feature Space)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    corr_path = os.path.join(output_dir, "metric_correlation_heatmap.png")
    fig.savefig(corr_path, dpi=150)
    plt.close(fig)
    print(f"  -> Saved: {corr_path}", flush=True)
    
    # Tabular Classification Benchmark (Predicting Major from Numerical Scores)
    num_cols = ['Age', 'Grade', 'Attendance_Rate', 'Study_Hours_Per_Week', 'Assignment_Score', 'Midterm_Score', 'Final_Score', 'Extracurricular_Hours', 'Project_Score']
    tab_vec_num = TabularVectorizer(numerical_cols=num_cols, categorical_cols=[], id_col='Student_ID')
    X_academic = tab_vec_num.fit_transform(df_students)
    y_tab = df_students['Major'].values
    
    clf_suite = PatternClassifierSuite()
    cv_results = clf_suite.cross_validate_all(X_academic, y_tab, n_splits=3)
    cv_csv_path = os.path.join(output_dir, "classifier_benchmark.csv")
    cv_results.to_csv(cv_csv_path, index=False)
    
    # Bar Chart for Classifier Benchmark
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=cv_results, x='Classifier', y='CV Mean F1', hue='Classifier', palette="crest", legend=False, ax=ax)
    ax.set_title("Cross-Validated F1-Score Benchmark Across Pattern Classifiers", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis='x', rotation=25)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    clf_bar_path = os.path.join(output_dir, "classifier_performance_benchmark.png")
    fig.savefig(clf_bar_path, dpi=150)
    plt.close(fig)
    print(f"  -> Saved: {clf_bar_path}", flush=True)

    # -------------------------------------------------------------
    # 2. IMAGE PATTERN EXPERIMENTS (Visual Shape & Color Descriptors)
    # -------------------------------------------------------------
    print("\n[2/5] Running Visual Image Pattern Experiments...", flush=True)
    image_dir = os.path.join(ROOT_DIR, 'data', 'images')
    img_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    
    modes = ['flattened_spatial', 'color_histogram', 'edge_gradient', 'hybrid']
    img_vectors_by_mode = {}
    for mode in modes:
        ivec = ImageVectorizer(image_size=(64, 64), extraction_mode=mode)
        vecs = ivec.transform(image_dir)
        img_vectors_by_mode[mode] = (ivec, vecs)
        print(f"  -> Extracted '{mode}': shape {vecs.shape}", flush=True)

    # -------------------------------------------------------------
    # 3. TEXT PATTERN EXPERIMENTS (Academic Syllabus & Abstracts)
    # -------------------------------------------------------------
    print("\n[3/5] Running Text Pattern NLP Experiments...", flush=True)
    text_csv = os.path.join(ROOT_DIR, 'data', 'academic_texts.csv')
    df_texts = pd.read_csv(text_csv)
    
    txt_vec = TextVectorizer(mode='tfidf', ngram_range=(1, 2), max_features=50)
    txt_vectors = txt_vec.fit_transform(df_texts['Text'].tolist())
    print(f"  -> Vectorized {txt_vectors.shape[0]} documents into {txt_vectors.shape[1]}-D TF-IDF space.", flush=True)

    # -------------------------------------------------------------
    # 4. CLUSTERING & ELBOW CURVE ANALYSIS
    # -------------------------------------------------------------
    print("\n[4/5] Running Clustering & Silhouette Analysis...", flush=True)
    hybrid_vecs = img_vectors_by_mode['hybrid'][1]
    cluster_suite = PatternClusterSuite(hybrid_vecs)
    elbow_df = cluster_suite.compute_elbow_curve(k_range=range(2, 9))
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:red'
    ax1.set_xlabel('Number of Clusters (K)', fontweight='bold')
    ax1.set_ylabel('Inertia (Sum of Squared Distances)', color=color, fontweight='bold')
    ax1.plot(elbow_df['k'], elbow_df['Inertia'], 'o-', color=color, linewidth=2, markersize=7)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color, fontweight='bold')
    ax2.plot(elbow_df['k'], elbow_df['Silhouette Score'], 's--', color=color, linewidth=2, markersize=7)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("K-Means Elbow Inertia & Silhouette Curve (Visual Hybrid Feature Space)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    cluster_plot_path = os.path.join(output_dir, "clustering_inertia_silhouette.png")
    fig.savefig(cluster_plot_path, dpi=150)
    plt.close(fig)
    print(f"  -> Saved: {cluster_plot_path}", flush=True)

    # -------------------------------------------------------------
    # 5. MULTI-MODAL 2D PROJECTION ATLAS (PCA vs t-SNE)
    # -------------------------------------------------------------
    print("\n[5/5] Generating Multi-Modal PCA & t-SNE 2D Projection Atlas...", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    
    # Subplot 1: Tabular PCA
    pca_tab = DimensionalityReducer(n_components=2, method='pca')
    tab_pca_coords = pca_tab.fit_transform(tab_vectors)
    sns.scatterplot(
        x=tab_pca_coords[:, 0], y=tab_pca_coords[:, 1],
        hue=df_students['Major'], style=df_students['Major'],
        s=90, ax=axes[0, 0], palette="Set2"
    )
    axes[0, 0].set_title(f"Tabular Student Space (PCA: {sum(pca_tab.explained_variance_ratio_)*100:.1f}% var)", fontweight='bold')
    axes[0, 0].legend(fontsize=8, loc='best')
    
    # Subplot 2: Tabular t-SNE
    tsne_tab = DimensionalityReducer(n_components=2, method='tsne')
    tab_tsne_coords = tsne_tab.fit_transform(tab_vectors, perplexity=10)
    sns.scatterplot(
        x=tab_tsne_coords[:, 0], y=tab_tsne_coords[:, 1],
        hue=df_students['Major'], style=df_students['Major'],
        s=90, ax=axes[0, 1], palette="Set2"
    )
    axes[0, 1].set_title("Tabular Student Space (t-SNE Manifold)", fontweight='bold')
    axes[0, 1].legend(fontsize=8, loc='best')

    # Subplot 3: Visual Image Hybrid Space (PCA)
    pca_img = DimensionalityReducer(n_components=2, method='pca')
    img_pca_coords = pca_img.fit_transform(hybrid_vecs)
    shape_labels = [os.path.basename(p).split('_')[0] for p in img_files]
    sns.scatterplot(
        x=img_pca_coords[:, 0], y=img_pca_coords[:, 1],
        hue=shape_labels, style=shape_labels,
        s=90, ax=axes[1, 0], palette="tab10"
    )
    axes[1, 0].set_title(f"Visual Hybrid Space (PCA: {sum(pca_img.explained_variance_ratio_)*100:.1f}% var)", fontweight='bold')
    axes[1, 0].legend(fontsize=7, loc='best', ncol=2)

    # Subplot 4: Text TF-IDF Space (PCA)
    pca_txt = DimensionalityReducer(n_components=2, method='pca')
    txt_pca_coords = pca_txt.fit_transform(txt_vectors)
    sns.scatterplot(
        x=txt_pca_coords[:, 0], y=txt_pca_coords[:, 1],
        hue=df_texts['Domain'], style=df_texts['Domain'],
        s=90, ax=axes[1, 1], palette="Dark2"
    )
    axes[1, 1].set_title(f"Text TF-IDF Space (PCA: {sum(pca_txt.explained_variance_ratio_)*100:.1f}% var)", fontweight='bold')
    axes[1, 1].legend(fontsize=8, loc='best')

    plt.suptitle("Multi-Modal Pattern Vector Space Projections (PCA vs t-SNE)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    proj_path = os.path.join(output_dir, "pca_tsne_multimodal_projections.png")
    fig.savefig(proj_path, dpi=150)
    plt.close(fig)
    print(f"  -> Saved: {proj_path}", flush=True)

    print("\n" + "="*70, flush=True)
    print(f"ALL EXPERIMENTS COMPLETED SUCCESSFULLY! Artifacts saved in: {output_dir}", flush=True)
    print("="*70 + "\n", flush=True)

if __name__ == "__main__":
    run_all_experiments()
