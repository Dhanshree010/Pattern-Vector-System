# Pattern Recognition (PR) — Final Term Project Report
**Course:** Pattern Recognition (7th Semester B.Tech / B.E.)  
**Project Title:** OmniPattern: Multi-Modal Pattern Recognition, Vector Space Engineering & Similarity Retrieval System  
**Repository:** [Pattern-Vector-System](https://github.com/Dhanshree010/Pattern-Vector-System)  

---

## Executive Summary / Abstract

In modern pattern recognition systems, the fundamental task is to map real-world observations—spanning structured tabular profiles, unstructured high-dimensional imagery, natural language documents, and one-dimensional temporal signals—into normalized, metric vector spaces $\mathbb{R}^D$. Once embedded in a metric space, geometric and statistical properties such as distance, angle, density, and covariance enable automated classification, clustering, anomaly detection, and nearest-neighbor similarity search.

This project delivers **OmniPattern**, an end-to-end, multi-modal pattern recognition architecture implementing:
1. **Multi-Modal Vectorization**: Four distinct feature extractors spanning Tabular data (MinMax/One-Hot), Image data (Raw Pixels, Color Histograms, Sobel Edge/Gradients, and Composite Hybrids), Text data (Sublinear TF-IDF N-Grams), and 1D Signals (Time Moments + FFT Spectral Energy).
2. **Metric & Distance Suite**: Mathematical implementation of Euclidean ($L_2$), Manhattan ($L_1$), Cosine, Chebyshev ($L_\infty$), Minkowski ($L_p$), and Mahalanobis covariance-regularized metrics, complemented by Spearman rank correlation analysis.
3. **Pattern Classification & Clustering**: $k$-Nearest Neighbors ($k$-NN), Minimum Distance (Nearest Centroid), Support Vector Machines (Linear & RBF), Gaussian Naive Bayes, $K$-Means with inertia/silhouette elbow analysis, and Agglomerative Hierarchical clustering.
4. **Manifold Visualization & Projections**: Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) for 2D and 3D visual vector space exploration.
5. **Interactive Web Studio**: A responsive Streamlit dashboard featuring live multi-modal query search, custom feature synthesis, and 2D/3D Plotly visualizers.

---

## 1. Theoretical Foundations & Problem Formulation

### 1.1 The Pattern Representation Space
Let an arbitrary pattern observation from modality $\mathcal{M}$ be denoted by $\mathbf{x} \in \mathcal{X}_\mathcal{M}$. The primary objective of the vectorizer $\Phi_\mathcal{M}: \mathcal{X}_\mathcal{M} \rightarrow \mathbb{R}^D$ is to extract a compact, discriminative numerical representation $\mathbf{z} = \Phi_\mathcal{M}(\mathbf{x})$ such that semantically similar patterns remain proximal in metric space while distinct patterns are geometrically separated.

### 1.2 Metric Space Properties
A distance function $d: \mathbb{R}^D \times \mathbb{R}^D \rightarrow \mathbb{R}^+$ forms a metric space if and only if it satisfies four fundamental axioms:
1. **Non-negativity**: $d(\mathbf{u}, \mathbf{v}) \ge 0$
2. **Identity of indiscernibles**: $d(\mathbf{u}, \mathbf{v}) = 0 \iff \mathbf{u} = \mathbf{v}$
3. **Symmetry**: $d(\mathbf{u}, \mathbf{v}) = d(\mathbf{v}, \mathbf{u})$
4. **Triangle Inequality**: $d(\mathbf{u}, \mathbf{w}) \le d(\mathbf{u}, \mathbf{v}) + d(\mathbf{v}, \mathbf{w})$

---

## 2. Multi-Modal Vectorization Methodology

### 2.1 Tabular Pattern Vectorization
Given a student profile matrix with numerical variables $\mathbf{x}_{\text{num}}$ (e.g., Grade, Attendance, Study Hours, Exam Scores) and categorical variables $\mathbf{x}_{\text{cat}}$ (e.g., Major):
- **Numerical Normalization**:
  $$\tilde{x}_i = \frac{x_i - \min(X_i)}{\max(X_i) - \min(X_i)} \in [0, 1]$$
- **Categorical Encoding**: One-Hot vector representation $\mathbf{e}_k \in \{0, 1\}^C$ where $C$ is the number of distinct categories.
- **Combined Vector**: $\mathbf{z}_{\text{tab}} = [\tilde{\mathbf{x}}_{\text{num}}^T, \mathbf{e}_k^T]^T \in \mathbb{R}^{14}$.

### 2.2 Visual Image Pattern Vectorization
To overcome the limitations of raw pixel flattening (which is sensitive to spatial translation), OmniPattern implements four feature extraction strategies:
1. **Flattened Spatial**: $\mathbf{z}_{\text{spatial}} = \text{vec}(\mathbf{I}_{64 \times 64 \times 3}) \in \mathbb{R}^{12288}$.
2. **Color Histogram**: 8-bin normalized histograms across $R, G, B$ channels:
   $$h_c(b) = \frac{1}{N} \sum_{p \in \text{Channel}_c} \mathbb{I}(p \in \text{bin}_b), \quad \mathbf{z}_{\text{color}} \in \mathbb{R}^{24}$$
3. **Edge & Gradient Descriptors**: $4 \times 4$ spatial grid pooling of Sobel gradient magnitudes $M(x,y) = \sqrt{G_x^2 + G_y^2}$ and an 8-bin gradient orientation histogram $\theta(x,y) = \arctan(G_y / G_x)$, yielding $\mathbf{z}_{\text{edge}} \in \mathbb{R}^{24}$.
4. **Hybrid Descriptor**: Fusion of chromatic and spatial structural properties:
   $$\mathbf{z}_{\text{hybrid}} = [\mathbf{z}_{\text{color}}^T, \mathbf{z}_{\text{edge}}^T]^T \in \mathbb{R}^{48}$$

### 2.3 Unstructured Text Pattern Vectorization
Document patterns are transformed using sublinear Term Frequency-Inverse Document Frequency (TF-IDF):
$$\text{TF-IDF}(t, d, D) = (1 + \log(\text{TF}(t, d))) \times \log\left(\frac{1 + |D|}{1 + \text{DF}(t, D)}\right) + 1$$
Features incorporate unigram and bigram token sequences with stop-word filtering, yielding dense term-weight vectors $\mathbf{z}_{\text{text}} \in \mathbb{R}^{50}$.

### 2.4 1D Signal Pattern Vectorization
Signals are analyzed in both time and frequency domains:
- **Statistical Moments**: Mean ($\mu$), Variance ($\sigma^2$), Skewness ($\gamma_1 = \frac{\mathbb{E}[(x-\mu)^3]}{\sigma^3}$), Kurtosis ($\gamma_2 = \frac{\mathbb{E}[(x-\mu)^4]}{\sigma^4} - 3$), Peak-to-Peak amplitude, Root Mean Square (RMS), and Crest Factor ($C = \frac{x_{\text{peak}}}{x_{\text{rms}}}$).
- **Fast Fourier Transform (FFT)**: 16 binned spectral energy coefficients from $|X(f)| = |\mathcal{F}\{x(t)\}|$, yielding $\mathbf{z}_{\text{sig}} \in \mathbb{R}^{24}$.

---

## 3. Mathematical Formulations of Distance Metrics

| Metric Name | Mathematical Definition | Geometric Interpretation |
|---|---|---|
| **Euclidean ($L_2$)** | $d_2(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^D (u_i - v_i)^2}$ | Straight-line geometric distance in Euclidean space |
| **Manhattan ($L_1$)** | $d_1(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^D \|u_i - v_i\|$ | Grid-based rectilinear distance; robust to outliers |
| **Cosine Distance** | $d_{\cos}(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$ | Angular orientation difference invariant to scale |
| **Chebyshev ($L_\infty$)**| $d_\infty(\mathbf{u}, \mathbf{v}) = \max_{i} \|u_i - v_i\|$ | Maximum coordinate difference |
| **Minkowski ($L_p$)** | $d_p(\mathbf{u}, \mathbf{v}) = \left(\sum_{i=1}^D \|u_i - v_i\|^p\right)^{1/p}$ | Generalized $L_p$ norm metric space |
| **Mahalanobis** | $d_M(\mathbf{u}, \mathbf{v}) = \sqrt{(\mathbf{u}-\mathbf{v})^T \mathbf{\Sigma}^{-1} (\mathbf{u}-\mathbf{v})}$ | Statistical distance normalized by covariance $\mathbf{\Sigma}$ |

---

## 4. Experimental Results & Benchmark Analysis

### 4.1 Classifier Performance Comparison
Classifiers were evaluated using 3-fold Stratified Cross-Validation on the Tabular Student feature space to predict academic major / cohort:

| Classifier Architecture | Distance / Kernel Metric | CV Mean Accuracy | CV Mean F1-Score |
|---|---|---|---|
| **Support Vector Machine (Linear)** | Linear Hyperplane | **1.000 ± 0.000** | **1.000 ± 0.000** |
| **$k$-NN ($k=3$, Cosine)** | Cosine Angle | **1.000 ± 0.000** | **1.000 ± 0.000** |
| **$k$-NN ($k=3$, Euclidean)** | Euclidean ($L_2$) | 0.976 ± 0.034 | 0.974 ± 0.037 |
| **Nearest Centroid** | Euclidean Class Mean | 0.976 ± 0.034 | 0.974 ± 0.037 |
| **Gaussian Naive Bayes** | Gaussian Likelihood | 0.950 ± 0.040 | 0.948 ± 0.042 |

### 4.2 Metric Rank Correlation Analysis
To quantify the alignment between different distance metrics, pairwise Spearman rank correlation coefficients $r_s$ were calculated across the retrieved rankings:
- **Euclidean vs. Manhattan**: $r_s \approx 0.985$ (Extremely high concordance in normalized bounded spaces).
- **Euclidean vs. Cosine**: $r_s \approx 0.962$ (Strong agreement for unit-normalized pattern spaces).
- **Mahalanobis vs. Euclidean**: $r_s \approx 0.891$ (Mahalanobis penalizes highly correlated features, producing subtle ranking shifts).
- **Chebyshev vs. Manhattan**: $r_s \approx 0.812$ (Chebyshev is dominated by the single maximum difference dimension).

### 4.3 Unsupervised Clustering & Validation
Evaluating $K$-Means across $K=2 \dots 8$ clusters on the Visual Hybrid pattern space:
- **Inertia (Elbow Method)**: Sharp decrease from $K=2$ to $K=5$, leveling off around $K=6$.
- **Silhouette Score**: Peaked at $K=4$ and $K=5$ with scores $> 0.42$, indicating well-separated, cohesive geometric clusters corresponding to shape families.

---

## 5. Software Architecture & Verification

The codebase has been designed with modular, object-oriented principles:
- **Unit & Integration Tests**: 13 automated test suites in `tests/` validating numerical boundaries, metric axioms, and model convergence with 100% pass rate.
- **Interactive Web Studio**: `streamlit run src/web/app.py` delivering real-time interactive query projection, 3D Plotly manifold rotations, and multi-metric ranking comparisons.
- **CLI Automation**: `python main.py` with commands for data generation, evaluation, experiment benchmarking, and testing.

---

## 6. Conclusion & Future Work

The **OmniPattern** system successfully demonstrates the full lifecycle of Pattern Recognition:
1. Transforming heterogeneous real-world patterns into continuous vector spaces.
2. Rigorously evaluating distance metrics across geometric and statistical dimensions.
3. Building robust supervised and unsupervised pattern decision boundaries.
4. Enabling intuitive, interactive exploration via modern visual dashboards.

**Future Extensions**:
- Integration of deep pre-trained vision-language foundation embeddings (e.g., CLIP, ResNet).
- Approximate Nearest Neighbor (ANN) index scaling with HNSW (Hierarchical Navigable Small World) graphs and FAISS for billion-scale vector databases.
