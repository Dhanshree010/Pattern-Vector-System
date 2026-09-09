import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.vectorizers.base import BasePatternVectorizer

class TextVectorizer(BasePatternVectorizer):
    """
    Unstructured Text / Document Pattern Vectorizer.
    Transforms raw textual narratives (e.g. course syllabi, research abstracts, student feedback)
    into high-dimensional sparse/dense vector spaces using TF-IDF and Bag-of-Words models.
    """
    
    def __init__(self, mode='tfidf', ngram_range=(1, 2), max_features=100, stop_words='english', sublinear_tf=True):
        super().__init__(name=f"TextVectorizer({mode})")
        self.mode = mode.lower()
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.stop_words = stop_words
        self.sublinear_tf = sublinear_tf
        
        if self.mode == 'tfidf':
            self.model = TfidfVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                stop_words=self.stop_words,
                sublinear_tf=self.sublinear_tf
            )
        elif self.mode == 'bow':
            self.model = CountVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                stop_words=self.stop_words
            )
        else:
            raise ValueError(f"Unsupported text mode: {self.mode}. Choose 'tfidf' or 'bow'.")

    def fit(self, corpus, **kwargs):
        """Fits vocabulary and IDF weights on the text corpus."""
        self.model.fit(corpus)
        self.feature_names_ = list(self.model.get_feature_names_out())
        self.vector_dim_ = len(self.feature_names_)
        self.is_fitted = True
        return self

    def transform(self, corpus, **kwargs):
        """Transforms documents into numerical term-weight vectors."""
        if not self.is_fitted:
            raise RuntimeError("TextVectorizer must be fitted before transformation.")
            
        if isinstance(corpus, str):
            corpus = [corpus]
            
        sparse_mat = self.model.transform(corpus)
        return sparse_mat.toarray().astype(np.float32)

    def get_top_keywords(self, vector, top_k=5):
        """Returns the top_k most significant vocabulary tokens for a given feature vector."""
        if len(vector.shape) > 1:
            vector = vector.flatten()
        top_indices = np.argsort(vector)[::-1][:top_k]
        return [(self.feature_names_[i], float(vector[i])) for i in top_indices if vector[i] > 0]
