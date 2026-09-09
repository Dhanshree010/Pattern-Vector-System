import unittest
import os
import sys
import numpy as np
from sklearn.datasets import make_blobs, make_classification

# Ensure root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.classifiers import PatternClassifierSuite
from src.models.clustering import PatternClusterSuite
from src.models.dimensionality import DimensionalityReducer

class TestPatternModels(unittest.TestCase):
    
    def setUp(self):
        # Create synthetic classification problem
        self.X, self.y = make_classification(
            n_samples=60, n_features=6, n_informative=4,
            n_classes=3, random_state=42
        )
        self.X_blobs, _ = make_blobs(n_samples=50, n_features=5, centers=3, random_state=42)

    def test_classifier_suite(self):
        suite = PatternClassifierSuite()
        suite.train_all(self.X[:40], self.y[:40])
        eval_df = suite.evaluate_all(self.X[40:], self.y[40:])
        
        self.assertGreater(len(eval_df), 0)
        self.assertIn('Accuracy', eval_df.columns)
        self.assertIn('F1-Score (Weighted)', eval_df.columns)
        
        # Test confusion matrix
        cm, labels = suite.get_confusion_matrix('k-NN (Euclidean, k=3)', self.X[40:], self.y[40:])
        self.assertEqual(cm.shape[0], len(labels))

    def test_clustering_suite(self):
        suite = PatternClusterSuite(self.X_blobs)
        
        # Test elbow curve
        elbow_df = suite.compute_elbow_curve(k_range=range(2, 5))
        self.assertEqual(len(elbow_df), 3)
        self.assertIn('Inertia', elbow_df.columns)
        
        # Test KMeans and evaluation
        labels, centers = suite.run_kmeans(n_clusters=3)
        self.assertEqual(len(labels), 50)
        self.assertEqual(centers.shape, (3, 5))
        
        metrics = suite.evaluate_clustering(labels)
        self.assertGreater(metrics['Silhouette Score'], 0.0)

    def test_dimensionality_reduction(self):
        # Test PCA 2D & 3D
        pca_2d = DimensionalityReducer(n_components=2, method='pca')
        coords_2d = pca_2d.fit_transform(self.X)
        self.assertEqual(coords_2d.shape, (60, 2))
        self.assertEqual(len(pca_2d.explained_variance_ratio_), 2)
        
        # Test Query Projection
        q_proj = pca_2d.transform_query(self.X[0])
        self.assertEqual(q_proj.shape, (1, 2))
        
        # Test t-SNE 2D
        tsne_2d = DimensionalityReducer(n_components=2, method='tsne')
        coords_tsne = tsne_2d.fit_transform(self.X)
        self.assertEqual(coords_tsne.shape, (60, 2))

if __name__ == '__main__':
    unittest.main()
