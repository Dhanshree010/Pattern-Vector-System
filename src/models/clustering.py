import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class PatternClusterSuite:
    """
    Unsupervised Pattern Discovery & Clustering Suite.
    Implements:
    - K-Means Clustering with inertia and optimal K elbow curve
    - Agglomerative Hierarchical Clustering (Ward, Complete, Average)
    - DBSCAN Density-Based Clustering
    - Intrinsic Cluster Quality Metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
    """
    
    def __init__(self, vector_space):
        self.vector_space = np.asarray(vector_space, dtype=np.float32)
        self.n_samples, self.dim = self.vector_space.shape

    def compute_elbow_curve(self, k_range=range(2, 9)):
        """Computes inertia and silhouette scores across a range of clusters K."""
        elbow_data = []
        for k in k_range:
            if k >= self.n_samples:
                break
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(self.vector_space)
            labels = kmeans.labels_
            sil = silhouette_score(self.vector_space, labels) if len(np.unique(labels)) > 1 else -1.0
            db = davies_bouldin_score(self.vector_space, labels) if len(np.unique(labels)) > 1 else 999.0
            
            elbow_data.append({
                'k': k,
                'Inertia': float(kmeans.inertia_),
                'Silhouette Score': float(sil),
                'Davies-Bouldin Index': float(db)
            })
            
        return pd.DataFrame(elbow_data)

    def run_kmeans(self, n_clusters=3):
        """Runs K-Means clustering and returns cluster labels and cluster centroids."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(self.vector_space)
        return labels, kmeans.cluster_centers_

    def run_hierarchical(self, n_clusters=3, linkage='ward'):
        """Runs Agglomerative Hierarchical Clustering."""
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = agg.fit_predict(self.vector_space)
        return labels

    def run_dbscan(self, eps=0.5, min_samples=3):
        """Runs DBSCAN density-based clustering."""
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(self.vector_space)
        return labels

    def evaluate_clustering(self, labels):
        """Computes cluster quality indices for a set of cluster labels."""
        unique_labels = np.unique(labels[labels != -1]) if -1 in labels else np.unique(labels)
        if len(unique_labels) < 2:
            return {
                'Clusters': len(unique_labels),
                'Silhouette Score': -1.0,
                'Davies-Bouldin': 999.0,
                'Calinski-Harabasz': 0.0
            }
            
        mask = labels != -1 if -1 in labels else np.ones(len(labels), dtype=bool)
        sil = silhouette_score(self.vector_space[mask], labels[mask])
        db = davies_bouldin_score(self.vector_space[mask], labels[mask])
        ch = calinski_harabasz_score(self.vector_space[mask], labels[mask])
        
        return {
            'Clusters': len(unique_labels),
            'Silhouette Score': float(sil),
            'Davies-Bouldin': float(db),
            'Calinski-Harabasz': float(ch)
        }
