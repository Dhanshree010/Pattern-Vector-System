import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

class ImageVectorizer:
    def __init__(self, image_size=(64, 64), color_mode='RGB'):
        self.image_size = image_size
        self.color_mode = color_mode
        self.image_paths = []
        self.vectors = None

    def load_and_vectorize(self, directory):
        """
        Loads all PNG images from a directory, converts color mode,
        resizes them to uniform dimensions, normalizes pixels to [0, 1],
        and flattens them into 1D feature vectors.
        """
        self.image_paths = sorted(glob.glob(os.path.join(directory, "*.png")))
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG images found in {directory}")

        vectors_list = []
        for path in self.image_paths:
            img = Image.open(path)
            
            # Convert color mode if needed
            if img.mode != self.color_mode:
                img = img.convert(self.color_mode)
                
            # Resize
            img = img.resize(self.image_size)
            
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Flatten to 1D vector
            flattened = img_array.flatten()
            vectors_list.append(flattened)
            
        self.vectors = np.array(vectors_list)
        return self.vectors

    def find_similar_images(self, query_index, top_n=3, metric='cosine'):
        """
        Finds the top_n most similar images to query_index using cosine similarity or euclidean distance.
        Returns a list of dictionaries with 'path', 'filename', and 'score'.
        """
        if self.vectors is None or len(self.vectors) == 0:
            raise ValueError("Vectors not initialized. Call load_and_vectorize first.")
            
        query_vector = self.vectors[query_index].reshape(1, -1)
        
        if metric == 'cosine':
            # Cosine similarity: 1 is most similar, -1 is opposite
            scores = cosine_similarity(query_vector, self.vectors)[0]
            ranked_indices = np.argsort(scores)[::-1]
        elif metric == 'euclidean':
            # Euclidean distance: 0 is identical, higher is more distant
            scores = euclidean_distances(query_vector, self.vectors)[0]
            ranked_indices = np.argsort(scores)
        else:
            raise ValueError("Unsupported metric. Choose 'cosine' or 'euclidean'.")
            
        results = []
        for idx in ranked_indices:
            if idx != query_index:
                results.append({
                    'index': idx,
                    'path': self.image_paths[idx],
                    'filename': os.path.basename(self.image_paths[idx]),
                    'score': float(scores[idx])
                })
                if len(results) >= top_n:
                    break
                    
        return results

    def plot_similarity_results(self, query_index, cosine_matches, euclidean_matches, output_path=None):
        """
        Visualizes the query image alongside top matches from Cosine Similarity and Euclidean Distance.
        """
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
        
        return fig


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, '..', 'data', 'images')
    output_plot_path = os.path.join(script_dir, '..', 'data', 'similarity_results.png')
    
    if not os.path.exists(image_dir) or len(glob.glob(os.path.join(image_dir, "*.png"))) == 0:
        print(f"Error: Images directory {image_dir} is empty or missing.")
        print("Please run generate_images.py first.")
        exit(1)
        
    print("--- Day 2: Image Vectorization & Similarity Search ---")
    vectorizer = ImageVectorizer(image_size=(64, 64), color_mode='RGB')
    vectors = vectorizer.load_and_vectorize(image_dir)
    
    print(f"Total Images Vectorized: {len(vectors)}")
    print(f"Vector Space Shape     : {vectors.shape} (Flattened 64x64x3 = {64*64*3} features)")
    print(f"Pixel Value Range      : Min={vectors.min():.2f}, Max={vectors.max():.2f}\n")
    
    # Select sample query image: circle_red.png or first available
    query_idx = 0
    for idx, path in enumerate(vectorizer.image_paths):
        if "circle_red" in path:
            query_idx = idx
            break
            
    query_filename = os.path.basename(vectorizer.image_paths[query_idx])
    print(f"=== Query Image: {query_filename} (Index: {query_idx}) ===")
    
    # Cosine Similarity
    print("\nTop 3 by Cosine Similarity (Higher is more similar):")
    cosine_results = vectorizer.find_similar_images(query_idx, top_n=3, metric='cosine')
    for res in cosine_results:
        print(f"  - {res['filename']:<20} -> Similarity Score: {res['score']:.4f}")
        
    # Euclidean Distance
    print("\nTop 3 by Euclidean Distance (Lower is closer):")
    euclidean_results = vectorizer.find_similar_images(query_idx, top_n=3, metric='euclidean')
    for res in euclidean_results:
        print(f"  - {res['filename']:<20} -> Euclidean Distance: {res['score']:.4f}")
        
    # Generate visualization plot
    print("\nGenerating similarity visualization plot...")
    vectorizer.plot_similarity_results(query_idx, cosine_results, euclidean_results, output_plot_path)
