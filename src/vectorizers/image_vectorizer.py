import os
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import sobel
from src.vectorizers.base import BasePatternVectorizer

class ImageVectorizer(BasePatternVectorizer):
    """
    Visual / Image Pattern Vectorizer.
    Transforms raw image matrices into discriminative feature spaces using:
    - 'flattened_spatial': Raw normalized spatial pixel vectors (W x H x C)
    - 'color_histogram': Multi-channel normalized color distribution histograms
    - 'edge_gradient': Spatial gradient magnitude and structural edge descriptors
    - 'hybrid': Concatenated multi-descriptor (Color Distribution + Shape Geometry)
    """
    
    def __init__(self, image_size=(64, 64), color_mode='RGB', extraction_mode='hybrid', hist_bins=8):
        super().__init__(name=f"ImageVectorizer({extraction_mode})")
        self.image_size = image_size
        self.color_mode = color_mode.upper()
        self.extraction_mode = extraction_mode.lower()
        self.hist_bins = hist_bins
        self.image_paths_ = []
        
        # Calculate expected dimensions
        if self.extraction_mode == 'flattened_spatial':
            channels = 3 if self.color_mode == 'RGB' else 1
            self.vector_dim_ = self.image_size[0] * self.image_size[1] * channels
        elif self.extraction_mode == 'color_histogram':
            channels = 3 if self.color_mode == 'RGB' else 1
            self.vector_dim_ = self.hist_bins * channels
        elif self.extraction_mode == 'edge_gradient':
            # 4x4 spatial grid of gradient energy + orientation bins
            self.vector_dim_ = 16 + 8
        elif self.extraction_mode == 'hybrid':
            # Color histogram (24) + Edge/Gradient spatial descriptor (24) = 48
            self.vector_dim_ = (self.hist_bins * 3) + 24
        else:
            raise ValueError(f"Unknown extraction_mode: {self.extraction_mode}")

    def fit(self, data, **kwargs):
        """Image vectorizers are deterministic feature extractors; fit marks state."""
        self.is_fitted = True
        return self

    def extract_features(self, img_pil):
        """Extracts feature vector from a single PIL Image according to the configured mode."""
        if img_pil.mode != self.color_mode:
            img_pil = img_pil.convert(self.color_mode)
            
        img_resized = img_pil.resize(self.image_size, Image.Resampling.BILINEAR)
        img_arr = np.array(img_resized, dtype=np.float32) / 255.0
        
        if self.extraction_mode == 'flattened_spatial':
            return img_arr.flatten()
            
        elif self.extraction_mode == 'color_histogram':
            return self._extract_color_histogram(img_arr)
            
        elif self.extraction_mode == 'edge_gradient':
            return self._extract_edge_gradients(img_arr)
            
        elif self.extraction_mode == 'hybrid':
            col_hist = self._extract_color_histogram(img_arr)
            edge_desc = self._extract_edge_gradients(img_arr)
            return np.concatenate([col_hist, edge_desc])
            
        else:
            raise ValueError(f"Unknown mode: {self.extraction_mode}")

    def _extract_color_histogram(self, img_arr):
        """Computes normalized color histogram per channel."""
        histograms = []
        channels = img_arr.shape[2] if len(img_arr.shape) == 3 else 1
        
        if channels == 1:
            h, _ = np.histogram(img_arr, bins=self.hist_bins, range=(0.0, 1.0), density=True)
            histograms.append(h / (np.sum(h) + 1e-7))
        else:
            for c in range(3):
                h, _ = np.histogram(img_arr[:, :, c], bins=self.hist_bins, range=(0.0, 1.0))
                norm_h = h / (np.sum(h) + 1e-7)
                histograms.append(norm_h)
                
        return np.concatenate(histograms)

    def _extract_edge_gradients(self, img_arr):
        """Computes spatial grid gradient energy and gradient orientation histogram."""
        if len(img_arr.shape) == 3:
            # Luminance conversion: 0.299 R + 0.587 G + 0.114 B
            gray = 0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]
        else:
            gray = img_arr
            
        # Compute horizontal and vertical Sobel gradients
        gx = sobel(gray, axis=1)
        gy = sobel(gray, axis=0)
        grad_mag = np.hypot(gx, gy)
        grad_dir = np.arctan2(gy, gx)  # [-pi, pi]
        
        # 1. 4x4 Spatial Grid pooling (16 features)
        grid_h, grid_w = gray.shape[0] // 4, gray.shape[1] // 4
        grid_energies = []
        for i in range(4):
            for j in range(4):
                block = grad_mag[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                grid_energies.append(np.mean(block))
        grid_energies = np.array(grid_energies)
        grid_energies = grid_energies / (np.linalg.norm(grid_energies) + 1e-7)
        
        # 2. 8-bin Gradient orientation histogram (8 features)
        dir_hist, _ = np.histogram(grad_dir, bins=8, range=(-np.pi, np.pi), weights=grad_mag)
        dir_hist = dir_hist / (np.sum(dir_hist) + 1e-7)
        
        return np.concatenate([grid_energies, dir_hist])

    def transform(self, data, **kwargs):
        """
        Transforms input data into feature vectors.
        Data can be: directory path, list of image file paths, list of PIL Images, or 4D numpy array.
        """
        self.is_fitted = True
        
        if isinstance(data, str) and os.path.isdir(data):
            self.image_paths_ = sorted(glob.glob(os.path.join(data, "*.png")) + glob.glob(os.path.join(data, "*.jpg")))
            if not self.image_paths_:
                raise FileNotFoundError(f"No image files found in {data}")
            vectors = [self.extract_features(Image.open(p)) for p in self.image_paths_]
            return np.array(vectors, dtype=np.float32)
            
        elif isinstance(data, list):
            vectors = []
            for item in data:
                if isinstance(item, str):
                    vectors.append(self.extract_features(Image.open(item)))
                elif isinstance(item, Image.Image):
                    vectors.append(self.extract_features(item))
                else:
                    raise TypeError(f"Unsupported item type in list: {type(item)}")
            return np.array(vectors, dtype=np.float32)
            
        elif isinstance(data, Image.Image):
            return self.extract_features(data).reshape(1, -1)
            
        else:
            raise TypeError(f"Unsupported data type for ImageVectorizer.transform: {type(data)}")

    def load_and_vectorize(self, directory):
        """Convenience method matching original Day 2 interface."""
        return self.transform(directory)
