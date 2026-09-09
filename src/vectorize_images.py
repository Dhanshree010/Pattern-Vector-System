import os
import sys
import glob

# Ensure root directory is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.vectorizers.image_vectorizer import ImageVectorizer as CoreImageVectorizer
from src.metrics.distance_metrics import DistanceMetrics
from src.metrics.similarity_engine import SimilarityRetrievalEngine

class ImageVectorizer:
    """Backward-compatible wrapper around core ImageVectorizer."""
    def __init__(self, image_size=(64, 64), color_mode='RGB', extraction_mode='flattened_spatial'):
        self._vectorizer = CoreImageVectorizer(
            image_size=image_size,
            color_mode=color_mode,
            extraction_mode=extraction_mode
        )
        self.image_paths = []
        self.vectors = None

    def load_and_vectorize(self, directory):
        self.vectors = self._vectorizer.load_and_vectorize(directory)
        self.image_paths = self._vectorizer.image_paths_
        return self.vectors

    def find_similar_images(self, query_index, top_n=3, metric='cosine'):
        engine = SimilarityRetrievalEngine(self.vectors, identifiers=[os.path.basename(p) for p in self.image_paths])
        metric_key = 'cosine_sim' if metric == 'cosine' else 'euclidean'
        results = engine.query_top_k(self.vectors[query_index], top_k=top_n, metric=metric_key, exclude_index=query_index)
        
        legacy_results = []
        for r in results:
            idx = r['index']
            legacy_results.append({
                'index': idx,
                'path': self.image_paths[idx],
                'filename': os.path.basename(self.image_paths[idx]),
                'score': r['score']
            })
        return legacy_results

    def plot_similarity_results(self, query_index, cosine_matches, euclidean_matches, output_path=None):
        top_n = max(len(cosine_matches), len(euclidean_matches))
        fig, axes = plt.subplots(2, top_n + 1, figsize=(3 * (top_n + 1), 6))
        
        query_img = Image.open(self.image_paths[query_index])
        query_name = os.path.basename(self.image_paths[query_index])
        
        # Row 0: Cosine Similarity
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title(f"Query Image\n{query_name}", fontsize=10, fontweight='bold', color='darkblue')
        axes[0, 0].axis('off')
        
        for i, match in enumerate(cosine_matches):
            img = Image.open(match['path'])
            axes[0, i + 1].imshow(img)
            axes[0, i + 1].set_title(f"Cosine Rank #{i+1}\n{match['filename']}\nSim: {match['score']:.4f}", fontsize=9)
            axes[0, i + 1].axis('off')
            
        for j in range(len(cosine_matches) + 1, top_n + 1):
            axes[0, j].axis('off')
            
        # Row 1: Euclidean Distance
        axes[1, 0].imshow(query_img)
        axes[1, 0].set_title(f"Query Image\n{query_name}", fontsize=10, fontweight='bold', color='darkblue')
        axes[1, 0].axis('off')
        
        for i, match in enumerate(euclidean_matches):
            img = Image.open(match['path'])
            axes[1, i + 1].imshow(img)
            axes[1, i + 1].set_title(f"Euclidean Rank #{i+1}\n{match['filename']}\nDist: {match['score']:.4f}", fontsize=9)
            axes[1, i + 1].axis('off')
            
        for j in range(len(euclidean_matches) + 1, top_n + 1):
            axes[1, j].axis('off')

        fig.suptitle(f"Day 2 Image Vectorization & Similarity Analysis: '{query_name}'", fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Comparison plot saved successfully to: {output_path}")
            plt.close(fig)
        
        return fig

if __name__ == "__main__":
    image_dir = os.path.join(ROOT_DIR, 'data', 'images')
    output_plot_path = os.path.join(ROOT_DIR, 'data', 'similarity_results.png')
    
    if not os.path.exists(image_dir) or len(glob.glob(os.path.join(image_dir, "*.png"))) == 0:
        print(f"Error: Images directory {image_dir} is empty or missing.")
        print("Please run generate_images.py first.")
        exit(1)
        
    print("=================================================================")
    print("            DAY 2: IMAGE PATTERN VECTORIZATION & SEARCH           ")
    print("=================================================================")
    vectorizer = ImageVectorizer(image_size=(64, 64), color_mode='RGB', extraction_mode='flattened_spatial')
    vectors = vectorizer.load_and_vectorize(image_dir)
    
    print(f"Total Images Vectorized : {len(vectors)}")
    print(f"Vector Space Shape      : {vectors.shape} (Flattened 64x64x3 = {vectors.shape[1]} features)")
    print(f"Pixel Value Range       : Min={vectors.min():.2f}, Max={vectors.max():.2f}\n")
    
    # Select sample query image
    query_idx = 0
    for idx, path in enumerate(vectorizer.image_paths):
        if "circle_red" in path:
            query_idx = idx
            break
            
    query_filename = os.path.basename(vectorizer.image_paths[query_idx])
    print(f"=== Query Image: {query_filename} (Index: {query_idx}) ===")
    
    print("\nTop 3 by Cosine Similarity (Higher is more similar):")
    cosine_results = vectorizer.find_similar_images(query_idx, top_n=3, metric='cosine')
    for res in cosine_results:
        print(f"  * {res['filename']:<22} -> Similarity Score: {res['score']:.4f}")
        
    print("\nTop 3 by Euclidean Distance (Lower is closer):")
    euclidean_results = vectorizer.find_similar_images(query_idx, top_n=3, metric='euclidean')
    for res in euclidean_results:
        print(f"  * {res['filename']:<22} -> Euclidean Distance: {res['score']:.4f}")
        
    print("\nGenerating similarity visualization plot...")
    vectorizer.plot_similarity_results(query_idx, cosine_results, euclidean_results, output_plot_path)
    print("=================================================================\n")
