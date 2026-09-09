# OmniPattern: Multi-Modal Pattern Recognition & Vector Space Retrieval System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit%201.30+-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

An academic-grade, production-ready **Pattern Recognition (PR)** system implementing multi-modal pattern vectorization, distance metric benchmarking, neighborhood retrieval, classification, clustering, dimensionality reduction, and interactive exploration.

---

## 📑 Table of Contents
1. [Overview & Architecture](#overview--architecture)
2. [Supported Pattern Modalities](#supported-pattern-modalities)
3. [Distance & Similarity Metrics Suite](#distance--similarity-metrics-suite)
4. [Pattern Classifiers & Clustering Suite](#pattern-classifiers--clustering-suite)
5. [Interactive Web Studio](#interactive-web-studio)
6. [Installation & Setup](#installation--setup)
7. [Command-Line Interface (CLI)](#command-line-interface-cli)
8. [Automated Testing](#automated-testing)
9. [Project Directory Structure](#project-directory-structure)

---

## 🔮 Overview & Architecture

**OmniPattern** formalizes the mapping of heterogeneous raw phenomena into continuous, normalized vector spaces $\mathbb{R}^D$, allowing rigorous pattern analysis, geometric neighborhood discovery, and decision boundary modeling.

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 RAW PATTERN OBSERVATIONS                │
                  │  [Tabular Records]  [Images]  [Text NLP]  [1D Signals]  │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │          MULTI-MODAL VECTORIZATION PIPELINE             │
                  │  * MinMax / One-Hot Scaling   * Sobel & Color Hists     │
                  │  * TF-IDF N-Grams             * FFT Spectral Energy     │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │             CONTINUOUS VECTOR SPACE R^D                 │
                  └───────┬─────────────────────┬───────────────────┬───────┘
                          │                     │                   │
                          ▼                     ▼                   ▼
            ┌──────────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
            │   SIMILARITY ENGINE  │  │   CLASSIFIERS    │  │  MANIFOLD PROJECTION │
            │  * Cosine / L1 / L2  │  │  * k-NN          │  │  * PCA 2D/3D         │
            │  * Chebyshev / Mink. │  │  * Min-Centroid  │  │  * t-SNE Embeddings  │
            │  * Mahalanobis Cov.  │  │  * SVM & Bayes   │  │  * K-Means Clusters  │
            └──────────────────────┘  └──────────────────┘  └──────────────────────┘
```

---

## 🎯 Supported Pattern Modalities

| Modality | Vectorizer Class | Feature Extraction Method | Output Dimension |
|---|---|---|---|
| **Tabular (Students)** | [`TabularVectorizer`](file:///c:/Users/Dhanu/OneDrive/Desktop/work/7th_sem/pr/TAE1_PR/src/vectorizers/tabular_vectorizer.py) | MinMax/Standard Scaling + One-Hot Encoding | 14-D Continuous |
| **Visual (Images)** | [`ImageVectorizer`](file:///c:/Users/Dhanu/OneDrive/Desktop/work/7th_sem/pr/TAE1_PR/src/vectorizers/image_vectorizer.py) | Raw Pixels, RGB Histograms, Sobel Gradients, Hybrid | 24-D to 12,288-D |
| **Text (NLP)** | [`TextVectorizer`](file:///c:/Users/Dhanu/OneDrive/Desktop/work/7th_sem/pr/TAE1_PR/src/vectorizers/text_vectorizer.py) | Sublinear TF-IDF + Unigram/Bigram Vocabulary | 50-D to 100-D |
| **Signals (1D)** | [`SignalVectorizer`](file:///c:/Users/Dhanu/OneDrive/Desktop/work/7th_sem/pr/TAE1_PR/src/vectorizers/signal_vectorizer.py) | Statistical Moments (Mean, Kurtosis, etc.) + FFT Spectrum | 24-D Combined |

---

## 📐 Distance & Similarity Metrics Suite

OmniPattern implements a comprehensive suite of mathematical distance functions in [`DistanceMetrics`](file:///c:/Users/Dhanu/OneDrive/Desktop/work/7th_sem/pr/TAE1_PR/src/metrics/distance_metrics.py):

- **Euclidean ($L_2$) Distance**: $d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum (u_i - v_i)^2}$
- **Manhattan ($L_1$) Distance**: $d(\mathbf{u}, \mathbf{v}) = \sum |u_i - v_i|$
- **Cosine Similarity & Distance**: $S(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$
- **Chebyshev ($L_\infty$) Distance**: $d(\mathbf{u}, \mathbf{v}) = \max_i |u_i - v_i|$
- **Minkowski ($L_p$) Distance**: $d(\mathbf{u}, \mathbf{v}) = (\sum |u_i - v_i|^p)^{1/p}$
- **Mahalanobis Distance**: $d(\mathbf{u}, \mathbf{v}) = \sqrt{(\mathbf{u}-\mathbf{v})^T \mathbf{\Sigma}^{-1} (\mathbf{u}-\mathbf{v})}$

---

## 🔬 Pattern Classifiers & Clustering Suite

- **Supervised Classifiers**:
  - $k$-Nearest Neighbors ($k$-NN) with Euclidean, Manhattan, and Cosine metrics
  - Nearest Centroid / Minimum Distance Classifier
  - Support Vector Machines (Linear & RBF Kernels)
  - Gaussian Naive Bayes
- **Unsupervised Clustering**:
  - $K$-Means with automated Inertia & Silhouette score Elbow analysis
  - Agglomerative Hierarchical Clustering with linkage strategies
  - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- **Dimensionality Reduction**:
  - Principal Component Analysis (PCA) with explained variance ratios
  - t-Distributed Stochastic Neighbor Embedding (t-SNE) for 2D/3D manifold projections

---

## 🚀 Installation & Setup

1. **Clone and Navigate**:
   ```bash
   git clone https://github.com/Dhanshree010/Pattern-Vector-System.git
   cd TAE1_PR
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**:
   ```bash
   python -m unittest discover tests
   ```

---

## 💻 Command-Line Interface (CLI)

Run the unified CLI entrypoint [`main.py`](file:///c:/Users/Dhanu/OneDrive/Desktop/work/7th_sem/pr/TAE1_PR/main.py):

```bash
# 1. Run multi-modal end-to-end demo across all 4 pattern modalities
python main.py demo

# 2. Generate 64 synthetic geometric and multi-color pattern images
python main.py generate-data

# 3. Run full benchmark experiments and generate visualization plots
python main.py experiments

# 4. Evaluate classifiers using Stratified K-Fold Cross-Validation
python main.py evaluate

# 5. Launch the interactive Streamlit Web Studio
python main.py web

# 6. Run automated test suite
python main.py test
```

---

## 🎨 Interactive Web Studio

Launch the full interactive studio in your browser:
```bash
streamlit run src/web/app.py
```

### Web Studio Modules:
1. **🏠 Overview & Foundations**: Mathematical formulations and vector pipeline.
2. **🎨 Visual Pattern Studio**: Live image similarity search, feature histogram breakdown, and multi-metric comparison cards.
3. **📊 Tabular Profiler**: Student search, real-time dynamic slider vectorizer, and peer cohort retrieval.
4. **📝 Text & NLP Engine**: TF-IDF semantic document search and salient keyword badge display.
5. **📡 1D Signal Analyzer**: Waveform synthesizer, FFT frequency spectrum, and statistical moments.
6. **🌐 2D/3D Vector Space Explorer**: Interactive Plotly scatter plot with PCA/t-SNE projections and live sample query injection.
7. **🔬 Metric Benchmark Hub**: Spearman rank correlation heatmap, cross-validation bar charts, and K-Means elbow curves.

---

## 📂 Project Directory Structure

```
TAE1_PR/
├── data/
│   ├── student_records.csv          # Tabular student profile dataset (40 records)
│   ├── academic_texts.csv           # NLP course abstracts dataset (30 documents)
│   ├── images/                      # 64 Geometric & color pattern images
│   ├── similarity_results.png       # Image retrieval comparison plot
│   └── visualizations/              # Generated benchmark plots & heatmaps
│
├── src/
│   ├── vectorizers/                 # Modality feature extractors (Tabular, Image, Text, Signal)
│   ├── metrics/                     # Distance metrics & similarity retrieval engine
│   ├── models/                      # Classifiers, clustering & dimensionality reduction
│   ├── web/                         # Streamlit interactive Web Studio
│   ├── generate_images.py           # Synthetic shape generator
│   ├── vectorize_students.py        # Modular Day 1 entrypoint
│   ├── vectorize_images.py          # Modular Day 2 entrypoint
│   └── run_experiments.py          # Automated experiment & benchmark generator
│
├── tests/                           # Unit & integration test suite
├── main.py                          # Unified CLI entrypoint
├── requirements.txt                 # Project dependencies
├── README.md                        # Project documentation
└── PROJECT_REPORT.md                # 7th Semester university academic report
```
