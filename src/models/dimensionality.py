import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class DimensionalityReducer:
    """
    Dimensionality Reduction & Manifold Learning Engine.
    Projects high-dimensional multi-modal pattern vectors into 2D and 3D visual subspaces.
    Provides PCA (linear orthogonal projection) and t-SNE (non-linear manifold embedding).
    """
    
    def __init__(self, n_components=2, method='pca'):
        self.n_components = n_components
        self.method = method.lower()
        self.model = None
        self.explained_variance_ratio_ = None

    def fit_transform(self, vector_space, **kwargs):
        """Fits reducer on vector_space and returns low-dimensional coordinates."""
        X = np.asarray(vector_space, dtype=np.float32)
        
        if self.method == 'pca':
            self.model = PCA(n_components=min(self.n_components, X.shape[1], X.shape[0]))
            coords = self.model.fit_transform(X)
            self.explained_variance_ratio_ = self.model.explained_variance_ratio_
            return coords
            
        elif self.method == 'tsne':
            max_perp = max(2.0, float(X.shape[0] - 1) / 3.0)
            perp = kwargs.get('perplexity', min(15.0, max_perp))
            self.model = TSNE(
                n_components=self.n_components,
                perplexity=min(perp, max_perp),
                random_state=42,
                init='random',
                learning_rate='auto'
            )
            coords = self.model.fit_transform(X)
            return coords
            
        else:
            raise ValueError(f"Unsupported reduction method: {self.method}. Choose 'pca' or 'tsne'.")

    def transform_query(self, query_vector):
        """Projects a new out-of-sample query vector into the PCA subspace."""
        if self.method != 'pca' or self.model is None:
            raise RuntimeError("Out-of-sample query projection is only supported for fitted PCA reducer.")
        q = np.atleast_2d(query_vector)
        return self.model.transform(q)
