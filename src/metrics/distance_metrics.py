import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

class DistanceMetrics:
    """
    Comprehensive Distance & Similarity Metrics Suite for Pattern Vector Spaces.
    Provides mathematical formulations and efficient vectorized computations for:
    - Euclidean (L2)
    - Manhattan (L1)
    - Cosine Similarity & Distance
    - Chebyshev (L_infinity)
    - Minkowski (L_p)
    - Mahalanobis (Covariance-aware)
    - Hamming & Jaccard
    """
    
    SUPPORTED_METRICS = [
        'euclidean', 'manhattan', 'cosine_sim', 'cosine_dist',
        'chebyshev', 'minkowski_p3', 'mahalanobis'
    ]

    @staticmethod
    def euclidean(u, v):
        """Computes Euclidean (L2) Distance between vectors or matrices."""
        u = np.atleast_2d(u)
        v = np.atleast_2d(v)
        return cdist(u, v, metric='euclidean')

    @staticmethod
    def manhattan(u, v):
        """Computes Manhattan (L1 / City Block) Distance."""
        u = np.atleast_2d(u)
        v = np.atleast_2d(v)
        return cdist(u, v, metric='cityblock')

    @staticmethod
    def cosine_similarity(u, v):
        """Computes Cosine Similarity in range [-1.0, 1.0]. 1.0 indicates identical orientation."""
        u = np.atleast_2d(u)
        v = np.atleast_2d(v)
        u_norm = np.linalg.norm(u, axis=1, keepdims=True)
        v_norm = np.linalg.norm(v, axis=1, keepdims=True)
        u_norm[u_norm == 0] = 1e-9
        v_norm[v_norm == 0] = 1e-9
        
        sim = np.dot(u / u_norm, (v / v_norm).T)
        return np.nan_to_num(sim, nan=0.0)

    @staticmethod
    def cosine_distance(u, v):
        """Computes Cosine Distance (1 - Cosine Similarity). Range [0.0, 2.0]."""
        sim = DistanceMetrics.cosine_similarity(u, v)
        return np.clip(1.0 - sim, 0.0, 2.0)

    @staticmethod
    def chebyshev(u, v):
        """Computes Chebyshev (L_infinity / Maximum coordinate difference) Distance."""
        u = np.atleast_2d(u)
        v = np.atleast_2d(v)
        return cdist(u, v, metric='chebyshev')

    @staticmethod
    def minkowski(u, v, p=3):
        """Computes Minkowski Distance of order p."""
        u = np.atleast_2d(u)
        v = np.atleast_2d(v)
        return cdist(u, v, metric='minkowski', p=p)

    @staticmethod
    def mahalanobis(u, v, cov_matrix=None, data_reference=None):
        """
        Computes Mahalanobis Distance accounting for inter-feature covariance.
        If cov_matrix is not supplied, it is estimated from data_reference with Ridge regularization.
        """
        u = np.atleast_2d(u)
        v = np.atleast_2d(v)
        
        if cov_matrix is None:
            if data_reference is not None:
                cov = np.cov(data_reference, rowvar=False)
            else:
                combined = np.vstack([u, v])
                cov = np.cov(combined, rowvar=False)
                
            # Regularize covariance to guarantee invertibility (Tikhonov regularization)
            dim = cov.shape[0] if cov.ndim == 2 else 1
            if dim == 1:
                cov = np.array([[cov]]) if cov.ndim == 0 else cov
            cov_reg = cov + (1e-5 * np.eye(cov.shape[0]))
            inv_cov = np.linalg.pinv(cov_reg)
        else:
            inv_cov = np.linalg.pinv(cov_matrix + 1e-5 * np.eye(cov_matrix.shape[0]))
            
        return cdist(u, v, metric='mahalanobis', VI=inv_cov)

    @classmethod
    def compute_all_distances(cls, query_vector, vector_space, data_reference=None):
        """
        Computes distances from query_vector to all vectors in vector_space across all supported metrics.
        Returns a dictionary of 1D arrays: {metric_name: distance_array}
        """
        q = np.atleast_2d(query_vector)
        v = np.atleast_2d(vector_space)
        ref = data_reference if data_reference is not None else v
        
        results = {
            'Cosine Similarity': cls.cosine_similarity(q, v)[0],
            'Cosine Distance': cls.cosine_distance(q, v)[0],
            'Euclidean (L2)': cls.euclidean(q, v)[0],
            'Manhattan (L1)': cls.manhattan(q, v)[0],
            'Chebyshev (L_inf)': cls.chebyshev(q, v)[0],
            'Minkowski (p=3)': cls.minkowski(q, v, p=3)[0],
            'Mahalanobis': cls.mahalanobis(q, v, data_reference=ref)[0]
        }
        return results
