import unittest
import os
import sys
import numpy as np

# Ensure root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.metrics.distance_metrics import DistanceMetrics
from src.metrics.similarity_engine import SimilarityRetrievalEngine

class TestDistanceMetrics(unittest.TestCase):
    
    def setUp(self):
        self.u = np.array([1.0, 0.0, 0.0])
        self.v = np.array([0.0, 1.0, 0.0])
        self.w = np.array([1.0, 1.0, 0.0])
        self.space = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0]
        ])

    def test_euclidean_distance(self):
        dist = DistanceMetrics.euclidean(self.u, self.v)
        expected = np.sqrt(2.0)
        self.assertAlmostEqual(dist[0, 0], expected, places=5)
        
        # Distance to self must be 0
        self.assertAlmostEqual(DistanceMetrics.euclidean(self.u, self.u)[0, 0], 0.0)

    def test_manhattan_distance(self):
        dist = DistanceMetrics.manhattan(self.u, self.v)
        self.assertAlmostEqual(dist[0, 0], 2.0, places=5)

    def test_cosine_similarity_and_distance(self):
        # Orthogonal vectors -> cosine similarity = 0, cosine distance = 1
        sim = DistanceMetrics.cosine_similarity(self.u, self.v)[0, 0]
        dist = DistanceMetrics.cosine_distance(self.u, self.v)[0, 0]
        self.assertAlmostEqual(sim, 0.0, places=5)
        self.assertAlmostEqual(dist, 1.0, places=5)
        
        # Parallel vectors -> similarity = 1
        self.assertAlmostEqual(DistanceMetrics.cosine_similarity(self.u, self.u)[0, 0], 1.0, places=5)

    def test_chebyshev_distance(self):
        dist = DistanceMetrics.chebyshev(self.u, self.v)
        self.assertAlmostEqual(dist[0, 0], 1.0, places=5)

    def test_similarity_engine_retrieval(self):
        engine = SimilarityRetrievalEngine(self.space, identifiers=['U', 'V', 'W', '2W'])
        results = engine.query_top_k(self.u, top_k=2, metric='cosine_sim', exclude_index=0)
        
        self.assertEqual(len(results), 2)
        # W (1, 1, 0) and 2W (2, 2, 0) have identical cosine angle ~0.707 to U
        self.assertIn(results[0]['identifier'], ['W', '2W'])

    def test_metric_correlation_matrix(self):
        engine = SimilarityRetrievalEngine(self.space)
        corr_df = engine.compute_metric_correlation_matrix(self.u)
        self.assertEqual(corr_df.shape[0], corr_df.shape[1])
        # Diagonal elements must be 1.0
        for i in range(corr_df.shape[0]):
            self.assertAlmostEqual(corr_df.iloc[i, i], 1.0, places=4)

if __name__ == '__main__':
    unittest.main()
