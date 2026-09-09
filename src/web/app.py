import os
import sys
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Ensure root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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

# Page configuration
st.set_page_config(
    page_title="OmniPattern | Pattern Recognition Studio",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        background: #1E293B;
        color: #38BDF8;
        border: 1px solid #0284C7;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to load data
@st.cache_data
def load_tabular_data():
    csv_path = os.path.join(ROOT_DIR, 'data', 'student_records.csv')
    return pd.read_csv(csv_path)

@st.cache_data
def load_text_data():
    csv_path = os.path.join(ROOT_DIR, 'data', 'academic_texts.csv')
    return pd.read_csv(csv_path)

@st.cache_resource
def get_image_vectorizers(image_dir):
    modes = ['hybrid', 'flattened_spatial', 'color_histogram', 'edge_gradient']
    extractors = {}
    for m in modes:
        v = ImageVectorizer(image_size=(64, 64), extraction_mode=m)
        vecs = v.transform(image_dir)
        extractors[m] = (v, vecs)
    return extractors

# Sidebar navigation
st.sidebar.title("🔮 OmniPattern Studio")
st.sidebar.caption("7th Sem Pattern Recognition System")

nav_choice = st.sidebar.radio(
    "Navigation Hub",
    [
        "🏠 Overview & Architecture",
        "🎨 Visual Pattern Studio (Images)",
        "📊 Tabular Profiler (Students)",
        "📝 NLP & Text Pattern Engine",
        "📡 1D Signal / Waveform Analyzer",
        "🌐 2D/3D Vector Space Explorer",
        "🔬 Metric Benchmark & Classifier Hub"
    ]
)

# -------------------------------------------------------------
# 1. OVERVIEW & ARCHITECTURE
# -------------------------------------------------------------
if nav_choice == "🏠 Overview & Architecture":
    st.markdown('<div class="main-title">OmniPattern: Multi-Modal Pattern Recognition System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Advanced Pattern Vector Space Engineering, Distance Metric Evaluation & Dimensionality Projections</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Modalities Supported", "4 Modalities", "Tabular, Image, Text, Signal")
    with col2:
        st.metric("Distance Metrics", "7 Metrics", "L1, L2, Cosine, Linf, Mahalanobis...")
    with col3:
        st.metric("Pattern Classifiers", "7 Models", "k-NN, Centroid, SVM, Naive Bayes")
    with col4:
        st.metric("Manifold Projections", "PCA & t-SNE", "2D & 3D Interactive")
        
    st.markdown("---")
    st.subheader("System Pipeline & Mathematical Foundations")
    
    st.markdown("""
    Pattern Recognition maps heterogeneous unstructured and structured phenomena into **continuous, normalized metric vector spaces** $\mathbb{R}^D$, enabling mathematical similarity calculation, neighborhood search, clustering, and decision boundary optimization.
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🔄 Vectorization Pipeline")
        st.markdown("""
        1. **Tabular Pattern Space**: Mixed MinMax / Standard scaling + One-Hot encoding of categorical attributes.
        2. **Visual Pattern Space**: Multi-channel color histograms, Sobel spatial gradient orientation descriptors, and raw flattened pixel tensors.
        3. **Text Pattern Space**: TF-IDF weighting with sublinear frequency scaling and n-gram vocabulary extraction.
        4. **Signal Pattern Space**: Time-domain moments (mean, std, skewness, kurtosis, crest factor) + Frequency-domain FFT spectral bins.
        """)
    with c2:
        st.markdown("#### 📐 Distance Metric Formulations")
        st.latex(r"d_{\text{Euclidean}}(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^D (u_i - v_i)^2}")
        st.latex(r"S_{\text{Cosine}}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}")
        st.latex(r"d_{\text{Mahalanobis}}(\mathbf{u}, \mathbf{v}) = \sqrt{(\mathbf{u}-\mathbf{v})^T \mathbf{\Sigma}^{-1} (\mathbf{u}-\mathbf{v})}")

# -------------------------------------------------------------
# 2. VISUAL PATTERN STUDIO
# -------------------------------------------------------------
elif nav_choice == "🎨 Visual Pattern Studio (Images)":
    st.markdown('<div class="main-title">Visual Pattern Recognition Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multi-descriptor visual feature extraction and similarity ranking</div>', unsafe_allow_html=True)
    
    image_dir = os.path.join(ROOT_DIR, 'data', 'images')
    img_extractors = get_image_vectorizers(image_dir)
    image_paths = img_extractors['hybrid'][0].image_paths_
    image_names = [os.path.basename(p) for p in image_paths]
    
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 1])
    with ctrl1:
        query_img_name = st.selectbox("Select Query Image", image_names, index=0)
    with ctrl2:
        extraction_mode = st.selectbox(
            "Feature Extraction Mode",
            ['hybrid', 'flattened_spatial', 'color_histogram', 'edge_gradient'],
            format_func=lambda x: {
                'hybrid': 'Hybrid (Color + Shape Geometry)',
                'flattened_spatial': 'Raw Spatial Pixels (64x64x3)',
                'color_histogram': 'Color Distribution (RGB Bins)',
                'edge_gradient': 'Edge & Gradient Histograms'
            }[x]
        )
    with ctrl3:
        metric_choice = st.selectbox(
            "Similarity Metric",
            ['cosine_sim', 'euclidean', 'manhattan', 'chebyshev', 'minkowski_p3', 'mahalanobis'],
            format_func=lambda x: {
                'cosine_sim': 'Cosine Similarity (Angle)',
                'euclidean': 'Euclidean Distance (L2)',
                'manhattan': 'Manhattan Distance (L1)',
                'chebyshev': 'Chebyshev Distance (L_inf)',
                'minkowski_p3': 'Minkowski (p=3)',
                'mahalanobis': 'Mahalanobis (Covariance)'
            }[x]
        )
    with ctrl4:
        top_k = st.slider("Top K Matches", min_value=1, max_value=8, value=4)
        
    query_idx = image_names.index(query_img_name)
    vec_instance, all_vectors = img_extractors[extraction_mode]
    query_vector = all_vectors[query_idx]
    
    st.markdown("---")
    col_q, col_res = st.columns([1, 3])
    
    with col_q:
        st.markdown("#### Query Pattern")
        q_img = Image.open(image_paths[query_idx])
        st.image(q_img, caption=query_img_name, use_container_width=True)
        st.info(f"**Feature Vector Dim:** {len(query_vector)}\n\n**Mode:** `{extraction_mode}`")
        
        # Mini sparkline of feature vector
        fig_bar = px.bar(x=list(range(min(len(query_vector), 32))), y=query_vector[:32], labels={'x': 'Feature Index', 'y': 'Value'})
        fig_bar.update_layout(title="Feature Vector (First 32 Dims)", height=180, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_res:
        st.markdown(f"#### Top {top_k} Retrieved Nearest Neighbors")
        engine = SimilarityRetrievalEngine(all_vectors, identifiers=image_names)
        results = engine.query_top_k(query_vector, top_k=top_k, metric=metric_choice, exclude_index=query_idx)
        
        grid_cols = st.columns(top_k)
        for i, r in enumerate(results):
            with grid_cols[i]:
                match_img = Image.open(image_paths[r['index']])
                st.image(match_img, caption=f"Rank #{i+1}\n{r['identifier']}", use_container_width=True)
                st.metric(f"Score ({r['metric'][:8]})", f"{r['score']:.4f}")
                
        # Comparison Table across metrics
        st.markdown("##### Multi-Metric Ranking Matrix for this Query")
        comp_df = engine.compare_metrics(query_vector, top_k=5, exclude_index=query_idx)
        st.dataframe(comp_df, use_container_width=True)

# -------------------------------------------------------------
# 3. TABULAR PROFILER (STUDENTS)
# -------------------------------------------------------------
elif nav_choice == "📊 Tabular Profiler (Students)":
    st.markdown('<div class="main-title">Tabular Student Pattern Profiler</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multi-attribute student record vectorization, academic similarity and cohort clustering</div>', unsafe_allow_html=True)
    
    df_students = load_tabular_data()
    tab_vec = TabularVectorizer(id_col='Student_ID', scaler_type='minmax')
    vectors = tab_vec.fit_transform(df_students)
    
    tab1, tab2 = st.tabs(["🔍 Student Similarity Search", "➕ Custom Student Vectorizer"])
    
    with tab1:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            sel_student = st.selectbox("Select Target Student", df_students['Student_ID'].tolist(), index=0)
        with c2:
            tab_metric = st.selectbox("Distance Metric", ['cosine_sim', 'euclidean', 'manhattan', 'chebyshev', 'mahalanobis'])
        with c3:
            tab_top_k = st.slider("Top K Similar Students", 1, 8, 4)
            
        target_idx = df_students[df_students['Student_ID'] == sel_student].index[0]
        target_row = df_students.iloc[target_idx]
        target_vec = vectors[target_idx]
        
        st.markdown(f"**Target Profile:** `{target_row['Student_ID']}` | Major: `{target_row['Major']}` | Grade: `{target_row['Grade']}` | Attendance: `{target_row['Attendance_Rate']}%` | Study Hours: `{target_row['Study_Hours_Per_Week']} hrs/wk`")
        
        engine = SimilarityRetrievalEngine(vectors, identifiers=df_students['Student_ID'].tolist(), metadata_df=df_students)
        matches = engine.query_top_k(target_vec, top_k=tab_top_k, metric=tab_metric, exclude_index=target_idx)
        
        match_records = []
        for m in matches:
            meta = m['metadata']
            match_records.append({
                'Student_ID': m['identifier'],
                'Similarity/Distance': round(m['score'], 4),
                'Major': meta['Major'],
                'Grade': meta['Grade'],
                'Attendance_Rate': meta['Attendance_Rate'],
                'Study_Hours': meta['Study_Hours_Per_Week'],
                'Midterm_Score': meta['Midterm_Score'],
                'Final_Score': meta['Final_Score']
            })
        st.dataframe(pd.DataFrame(match_records), use_container_width=True)

    with tab2:
        st.markdown("#### Dynamic Real-Time Student Vectorizer")
        st.caption("Adjust student attributes to generate real-time vector representation and find closest academic matches.")
        
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            age = st.slider("Age", 18, 25, 20)
            grade = st.slider("Grade", 60, 100, 85)
        with fc2:
            att = st.slider("Attendance Rate (%)", 50, 100, 90)
            study = st.slider("Study Hours / Week", 2, 30, 15)
        with fc3:
            assign = st.slider("Assignment Score", 50, 100, 88)
            midterm = st.slider("Midterm Score", 50, 100, 84)
        with fc4:
            final = st.slider("Final Score", 50, 100, 87)
            major = st.selectbox("Major", df_students['Major'].unique().tolist())
            
        custom_dict = {
            'Age': age, 'Grade': grade, 'Attendance_Rate': att,
            'Study_Hours_Per_Week': study, 'Assignment_Score': assign,
            'Midterm_Score': midterm, 'Final_Score': final,
            'Major': major, 'Extracurricular_Hours': 5, 'Project_Score': 85
        }
        
        custom_vec = tab_vec.vectorize_single_record(custom_dict)
        st.success(f"Generated Vector Representation ({len(custom_vec)} dimensions):")
        st.code(str(np.round(custom_vec, 3)))
        
        engine = SimilarityRetrievalEngine(vectors, identifiers=df_students['Student_ID'].tolist(), metadata_df=df_students)
        top_custom = engine.query_top_k(custom_vec, top_k=3, metric='cosine_sim')
        st.markdown("##### Nearest Existing Student Peers:")
        for tc in top_custom:
            m = tc['metadata']
            st.write(f"👉 **{tc['identifier']}** ({m['Major']}) | Grade: {m['Grade']} | Attendance: {m['Attendance_Rate']}% -> Cosine Sim: **{tc['score']:.4f}**")

# -------------------------------------------------------------
# 4. NLP & TEXT PATTERN ENGINE
# -------------------------------------------------------------
elif nav_choice == "📝 NLP & Text Pattern Engine":
    st.markdown('<div class="main-title">Text & NLP Pattern Retrieval Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">TF-IDF N-gram feature representation and semantic academic text retrieval</div>', unsafe_allow_html=True)
    
    df_texts = load_text_data()
    txt_vec = TextVectorizer(mode='tfidf', ngram_range=(1, 2), max_features=60)
    txt_vectors = txt_vec.fit_transform(df_texts['Text'].tolist())
    
    query_text = st.text_area(
        "Enter Query Text or Research Topic:",
        "Neural networks, visual feature extraction, and convolutional deep learning for image recognition."
    )
    
    c1, c2 = st.columns([2, 1])
    with c1:
        text_metric = st.selectbox("Distance Metric", ['cosine_sim', 'euclidean', 'manhattan', 'chebyshev'])
    with c2:
        text_k = st.slider("Top K Documents", 1, 6, 3)
        
    if st.button("🚀 Vectorize & Search"):
        q_vec = txt_vec.transform(query_text)[0]
        keywords = txt_vec.get_top_keywords(q_vec, top_k=6)
        
        st.markdown("#### Extracted Salient TF-IDF Keywords:")
        kw_cols = st.columns(len(keywords) if keywords else 1)
        for i, (kw, score) in enumerate(keywords):
            with kw_cols[i]:
                st.markdown(f"<span class='badge'>{kw} ({score:.3f})</span>", unsafe_allow_html=True)
                
        engine = SimilarityRetrievalEngine(txt_vectors, identifiers=df_texts['Title'].tolist(), metadata_df=df_texts)
        results = engine.query_top_k(q_vec, top_k=text_k, metric=text_metric)
        
        st.markdown("---")
        st.markdown("#### Top Retrieved Academic Documents:")
        for r in results:
            meta = r['metadata']
            st.markdown(f"""
            <div class="metric-card">
                <h5>{r['identifier']} <span class="badge">{meta['Domain']}</span> (Score: {r['score']:.4f})</h5>
                <p style="color: #CBD5E1; font-size: 0.9rem;">{meta['Text']}</p>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------
# 5. 1D SIGNAL ANALYZER
# -------------------------------------------------------------
elif nav_choice == "📡 1D Signal / Waveform Analyzer":
    st.markdown('<div class="main-title">1D Signal & Waveform Pattern Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Time-domain statistical moments and frequency-domain FFT spectrum feature extraction</div>', unsafe_allow_html=True)
    
    sig_col1, sig_col2, sig_col3 = st.columns(3)
    with sig_col1:
        sig_type = st.selectbox("Signal Waveform Type", ['sine', 'chirp', 'square', 'transient', 'noise'])
    with sig_col2:
        freq = st.slider("Base Frequency (Hz)", 5.0, 50.0, 15.0)
    with sig_col3:
        noise = st.slider("Noise Level", 0.0, 0.5, 0.1)
        
    t, sig = SignalVectorizer.generate_synthetic_signal(sig_type, duration=1.0, freq=freq, noise_level=noise)
    vec = SignalVectorizer(sample_rate=1000, n_fft_bins=16)
    features = vec.transform(sig)[0]
    
    p1, p2 = st.columns(2)
    with p1:
        fig_sig = px.line(x=t[:300], y=sig[:300], labels={'x': 'Time (s)', 'y': 'Amplitude'}, title=f"Time-Domain Waveform ({sig_type})")
        fig_sig.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_sig, use_container_width=True)
    with p2:
        fig_fft = px.bar(x=vec.feature_names_[8:], y=features[8:], labels={'x': 'FFT Bins', 'y': 'Normalized Energy'}, title="Frequency-Domain Spectral Bins")
        fig_fft.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_fft, use_container_width=True)
        
    st.markdown("#### Extracted Statistical & Spectral Vector (24 Dimensions):")
    stat_df = pd.DataFrame({'Feature': vec.feature_names_[:8], 'Value': [round(float(v), 4) for v in features[:8]]})
    st.dataframe(stat_df.T, use_container_width=True)

# -------------------------------------------------------------
# 6. 2D/3D VECTOR SPACE EXPLORER
# -------------------------------------------------------------
elif nav_choice == "🌐 2D/3D Vector Space Explorer":
    st.markdown('<div class="main-title">Interactive 2D/3D Vector Space Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Principal Component Analysis (PCA) and t-SNE manifold learning visualizations</div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        modality = st.selectbox("Select Pattern Modality", ["Tabular Students", "Visual Images", "Text Syllabus"])
    with c2:
        dim_mode = st.selectbox("Projection Dimension", ["2D Projection", "3D Projection"])
    with c3:
        method = st.selectbox("Dimensionality Algorithm", ["PCA (Linear)", "t-SNE (Non-linear)"])
        
    n_comp = 3 if dim_mode == "3D Projection" else 2
    method_key = 'pca' if "PCA" in method else 'tsne'
    
    if modality == "Tabular Students":
        df = load_tabular_data()
        vecs = TabularVectorizer(id_col='Student_ID').fit_transform(df)
        labels = df['Major'].tolist()
        hover_names = [f"{sid} ({m})" for sid, m in zip(df['Student_ID'], df['Major'])]
    elif modality == "Visual Images":
        img_dir = os.path.join(ROOT_DIR, 'data', 'images')
        img_ext = get_image_vectorizers(img_dir)['hybrid']
        vecs = img_ext[1]
        hover_names = [os.path.basename(p) for p in img_ext[0].image_paths_]
        labels = [name.split('_')[0] for name in hover_names]
    else:
        df_t = load_text_data()
        vecs = TextVectorizer(max_features=50).fit_transform(df_t['Text'].tolist())
        labels = df_t['Domain'].tolist()
        hover_names = df_t['Title'].tolist()
        
    reducer = DimensionalityReducer(n_components=n_comp, method=method_key)
    coords = reducer.fit_transform(vecs)
    
    if n_comp == 2:
        fig_proj = px.scatter(
            x=coords[:, 0], y=coords[:, 1], color=labels, hover_name=hover_names,
            labels={'x': 'Component 1', 'y': 'Component 2'},
            title=f"{modality} Vector Space ({method} 2D)"
        )
        fig_proj.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
        fig_proj.update_layout(height=550)
        st.plotly_chart(fig_proj, use_container_width=True)
    else:
        fig_proj = px.scatter_3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], color=labels, hover_name=hover_names,
            labels={'x': 'Comp 1', 'y': 'Comp 2', 'z': 'Comp 3'},
            title=f"{modality} Vector Space ({method} 3D)"
        )
        fig_proj.update_traces(marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')))
        fig_proj.update_layout(height=650)
        st.plotly_chart(fig_proj, use_container_width=True)

# -------------------------------------------------------------
# 7. METRIC BENCHMARK & CLASSIFIER HUB
# -------------------------------------------------------------
elif nav_choice == "🔬 Metric Benchmark & Classifier Hub":
    st.markdown('<div class="main-title">Metric Benchmark & Classifier Evaluation Suite</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Cross-validation benchmarks, distance metric correlations, and clustering validation</div>', unsafe_allow_html=True)
    
    df_students = load_tabular_data()
    tab_vec = TabularVectorizer(id_col='Student_ID', scaler_type='minmax')
    vectors = tab_vec.fit_transform(df_students)
    
    tab_a, tab_b, tab_c = st.tabs(["📊 Classifier Performance", "🔥 Metric Correlation Heatmap", "🧬 Clustering Analysis"])
    
    with tab_a:
        st.subheader("Pattern Classifier Comparison (Cross-Validated)")
        X = vectors
        y = df_students['Major'].values
        
        clf_suite = PatternClassifierSuite()
        cv_df = clf_suite.cross_validate_all(X, y, n_splits=3)
        
        fig_clf = px.bar(
            cv_df, x='Classifier', y='CV Mean F1', error_y='CV Std F1',
            color='CV Mean F1', color_continuous_scale='Blues',
            title="Cross-Validated F1-Score across Classifiers"
        )
        fig_clf.update_layout(height=400)
        st.plotly_chart(fig_clf, use_container_width=True)
        st.dataframe(cv_df, use_container_width=True)

    with tab_b:
        st.subheader("Spearman Rank Correlation Across Distance Metrics")
        st.caption("Demonstrates the geometric agreement between L1, L2, Linf, Cosine, and Mahalanobis spaces.")
        engine = SimilarityRetrievalEngine(vectors)
        corr_matrix = engine.compute_metric_correlation_matrix(vectors[0])
        
        fig_hm = px.imshow(
            corr_matrix, text_auto=".3f", aspect="auto",
            color_continuous_scale="Viridis", title="Metric Rank Correlation Matrix"
        )
        fig_hm.update_layout(height=450)
        st.plotly_chart(fig_hm, use_container_width=True)

    with tab_c:
        st.subheader("K-Means Cluster Quality & Optimal K Evaluation")
        cluster_suite = PatternClusterSuite(vectors)
        elbow_df = cluster_suite.compute_elbow_curve(k_range=range(2, 9))
        
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=elbow_df['k'], y=elbow_df['Inertia'], name='Inertia (Elbow)', mode='lines+markers', line=dict(color='#EF4444', width=3)))
        fig_elbow.add_trace(go.Scatter(x=elbow_df['k'], y=elbow_df['Silhouette Score'], name='Silhouette Score', mode='lines+markers', yaxis='y2', line=dict(color='#3B82F6', width=3, dash='dash')))
        
        fig_elbow.update_layout(
            title="Inertia and Silhouette Score vs. Number of Clusters (K)",
            xaxis=dict(title="Number of Clusters (K)"),
            yaxis=dict(title="Inertia (Sum of Squared Distances)"),
            yaxis2=dict(title="Silhouette Score", overlaying='y', side='right'),
            height=400
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.dataframe(elbow_df, use_container_width=True)
