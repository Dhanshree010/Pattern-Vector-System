import abc
import pickle
import numpy as np

class BasePatternVectorizer(abc.ABC):
    """
    Abstract Base Class for Multi-Modal Pattern Vectorizers.
    Defines common interface for feature extraction, vector space projection, and serialization.
    """
    
    def __init__(self, name="BaseVectorizer"):
        self.name = name
        self.is_fitted = False
        self.feature_names_ = []
        self.vector_dim_ = 0

    @abc.abstractmethod
    def fit(self, data, **kwargs):
        """Fits the vectorizer parameters on training pattern data."""
        pass

    @abc.abstractmethod
    def transform(self, data, **kwargs):
        """Transforms raw pattern data into numeric feature vectors."""
        pass

    def fit_transform(self, data, **kwargs):
        """Fits the vectorizer and returns transformed feature vectors."""
        self.fit(data, **kwargs)
        return self.transform(data, **kwargs)

    def get_feature_names(self):
        """Returns the list or array of extracted feature dimension names."""
        return self.feature_names_

    @property
    def vector_dimension(self):
        """Returns the dimensionality of the generated vector space."""
        return self.vector_dim_

    def save(self, filepath):
        """Serializes vectorizer instance to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """Deserializes vectorizer instance from disk."""
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
        if not isinstance(instance, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return instance
