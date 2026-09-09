import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.metrics.distance_metrics import DistanceMetrics

class SimilarityRetrievalEngine:
    """
    Pattern Similarity & Metric Comparison Retrieval Engine.
    Executes Top-K nearest neighbor searches across arbitrary metric spaces,
    generates multi-metric ranking comparisons, and computes rank correlation matrices.
    """
    
    def __init__(self, vector_space, identifiers=None, metadata_df=None):
        self.vector_space = np.asarray(vector_space, dtype=np.float32)
        self.num_samples, self.dim = self.vector_space.shape
        self.identifiers = identifiers if identifiers is not None else [f"ID_{i}" for i in range(self.num_samples)]
        self.metadata_df = metadata_df

    def query_top_k(self, query_vector, top_k=5, metric='cosine_sim', exclude_index=None):
        """
        Retrieves the top_k most similar pattern records.
        metric: 'cosine_sim', 'cosine_dist', 'euclidean', 'manhattan', 'chebyshev', 'minkowski_p3', 'mahalanobis'
        """
        all_metrics = DistanceMetrics.compute_all_distances(query_vector, self.vector_space)
        
        metric_map = {
            'cosine_sim': ('Cosine Similarity', True),    # Higher is better
            'cosine': ('Cosine Similarity', True),
            'cosine_dist': ('Cosine Distance', False),    # Lower is closer
            'euclidean': ('Euclidean (L2)', False),
            'manhattan': ('Manhattan (L1)', False),
            'chebyshev': ('Chebyshev (L_inf)', False),
            'minkowski': ('Minkowski (p=3)', False),
            'minkowski_p3': ('Minkowski (p=3)', False),
            'mahalanobis': ('Mahalanobis', False)
        }
        
        display_name, higher_is_better = metric_map.get(metric.lower(), ('Cosine Similarity', True))
        scores = all_metrics[display_name]
        
        if higher_is_better:
            sorted_indices = np.argsort(scores)[::-1]
        else:
            sorted_indices = np.argsort(scores)
            
        results = []
        for idx in sorted_indices:
            if exclude_index is not None and idx == exclude_index:
                continue
            item = {
                'index': int(idx),
                'identifier': self.identifiers[idx],
                'metric': display_name,
                'score': float(scores[idx])
            }
            if self.metadata_df is not None and idx < len(self.metadata_df):
                item['metadata'] = self.metadata_df.iloc[idx].to_dict()
            results.append(item)
            if len(results) >= top_k:
                break
                
        return results

    def compare_metrics(self, query_vector, top_k=5, exclude_index=None):
        """
        Generates a comparative ranking DataFrame for the query vector across all supported distance metrics.
        """
        all_metrics = DistanceMetrics.compute_all_distances(query_vector, self.vector_space)
        
        comparison_records = {}
        for metric_name, scores in all_metrics.items():
            if metric_name == 'Cosine Similarity':
                ranked_idx = np.argsort(scores)[::-1]
            else:
                ranked_idx = np.argsort(scores)
                
            filtered = [i for i in ranked_idx if exclude_index is None or i != exclude_index][:top_k]
            comparison_records[metric_name] = [f"{self.identifiers[i]} ({scores[i]:.3f})" for i in filtered]
            
        df_comp = pd.DataFrame(comparison_records)
        df_comp.index = [f"Rank #{r+1}" for r in range(top_k)]
        return df_comp

    def compute_metric_correlation_matrix(self, query_vector=None):
        """
        Computes Spearman rank correlation matrix across metrics for distance rankings.
        Demonstrates the geometric alignment or divergence between L1, L2, Linf, Cosine, and Mahalanobis spaces.
        """
        if query_vector is None:
            query_vector = self.vector_space[0]
            
        all_metrics = DistanceMetrics.compute_all_distances(query_vector, self.vector_space)
        metric_names = list(all_metrics.keys())
        n = len(metric_names)
        corr_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                s1 = all_metrics[metric_names[i]]
                s2 = all_metrics[metric_names[j]]
                # Reverse sign for cosine similarity so that lower is higher rank
                if metric_names[i] == 'Cosine Similarity':
                    s1 = -s1
                if metric_names[j] == 'Cosine Similarity':
                    s2 = -s2
                corr, _ = spearmanr(s1, s2)
                corr_matrix[i, j] = corr if not np.isnan(corr) else 1.0
                
        df_corr = pd.DataFrame(corr_matrix, index=metric_names, columns=metric_names)
        return df_corr
