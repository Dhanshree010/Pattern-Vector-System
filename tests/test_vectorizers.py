import unittest
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

# Ensure root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.vectorizers.tabular_vectorizer import TabularVectorizer
from src.vectorizers.image_vectorizer import ImageVectorizer
from src.vectorizers.text_vectorizer import TextVectorizer
from src.vectorizers.signal_vectorizer import SignalVectorizer

class TestPatternVectorizers(unittest.TestCase):
    
    def test_tabular_vectorizer(self):
        df = pd.DataFrame({
            'Student_ID': ['S01', 'S02', 'S03'],
            'Age': [20, 21, 22],
            'Grade': [80, 90, 70],
            'Major': ['CS', 'Math', 'CS']
        })
        vec = TabularVectorizer(id_col='Student_ID', scaler_type='minmax')
        vectors = vec.fit_transform(df)
        
        self.assertEqual(vectors.shape[0], 3)
        self.assertTrue(vectors.shape[1] >= 4)  # 2 num + 2 one-hot (CS, Math)
        self.assertTrue(np.all(vectors >= 0.0) and np.all(vectors <= 1.0))
        
        # Test single record vectorization
        single_vec = vec.vectorize_single_record({'Age': 20.5, 'Grade': 85, 'Major': 'CS'})
        self.assertEqual(single_vec.shape, (vec.vector_dimension,))

    def test_image_vectorizer_modes(self):
        # Create a sample test image
        img = Image.new('RGB', (64, 64), color=(255, 0, 0))
        
        modes = ['flattened_spatial', 'color_histogram', 'edge_gradient', 'hybrid']
        for mode in modes:
            vec = ImageVectorizer(image_size=(64, 64), extraction_mode=mode)
            feature_vec = vec.transform(img)
            self.assertEqual(feature_vec.shape[0], 1)
            self.assertEqual(feature_vec.shape[1], vec.vector_dimension)
            self.assertFalse(np.isnan(feature_vec).any())

    def test_text_vectorizer(self):
        corpus = [
            "Pattern recognition and machine learning algorithms in Python",
            "Linear algebra, vector spaces, and principal component analysis",
            "Digital image processing and computer vision techniques"
        ]
        vec = TextVectorizer(mode='tfidf', ngram_range=(1, 1), max_features=20)
        vectors = vec.fit_transform(corpus)
        
        self.assertEqual(vectors.shape[0], 3)
        self.assertEqual(vectors.shape[1], vec.vector_dimension)
        
        # Test out-of-vocabulary query
        query_vec = vec.transform("machine learning and computer vision")
        self.assertEqual(query_vec.shape, (1, vec.vector_dimension))
        keywords = vec.get_top_keywords(query_vec[0], top_k=2)
        self.assertTrue(len(keywords) > 0)

    def test_signal_vectorizer(self):
        vec = SignalVectorizer(sample_rate=1000, n_fft_bins=8)
        t, sig = SignalVectorizer.generate_synthetic_signal('sine', duration=0.5, freq=15.0)
        
        features = vec.transform(sig)
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[1], vec.vector_dimension)
        self.assertFalse(np.isnan(features).any())

if __name__ == '__main__':
    unittest.main()
