import numpy as np
from scipy import stats
from scipy.fft import rfft, rfftfreq
from src.vectorizers.base import BasePatternVectorizer

class SignalVectorizer(BasePatternVectorizer):
    """
    1D Time-Series / Acoustic / Signal Pattern Vectorizer.
    Extracts time-domain statistical moments and frequency-domain spectral features
    from 1D temporal sensor or acoustic patterns.
    """
    
    def __init__(self, sample_rate=1000, n_fft_bins=16):
        super().__init__(name="SignalVectorizer")
        self.sample_rate = sample_rate
        self.n_fft_bins = n_fft_bins
        
        # 8 time-domain features + n_fft_bins frequency features
        self.feature_names_ = [
            'time_mean', 'time_std', 'time_variance', 'time_skewness',
            'time_kurtosis', 'time_rms', 'time_peak_to_peak', 'time_crest_factor'
        ] + [f'fft_bin_{i}' for i in range(self.n_fft_bins)]
        
        self.vector_dim_ = len(self.feature_names_)

    def fit(self, data, **kwargs):
        """Deterministic feature extractor."""
        self.is_fitted = True
        return self

    def extract_signal_features(self, signal_1d):
        """Extracts comprehensive time-frequency feature vector from 1D array."""
        sig = np.asarray(signal_1d, dtype=np.float64)
        if sig.ndim > 1:
            sig = sig.flatten()
            
        # 1. Time-domain statistics
        mean_val = np.mean(sig)
        std_val = np.std(sig) + 1e-8
        var_val = np.var(sig)
        skew_val = float(stats.skew(sig))
        kurt_val = float(stats.kurtosis(sig))
        rms_val = np.sqrt(np.mean(sig**2))
        ptp_val = np.ptp(sig)
        peak_val = np.max(np.abs(sig))
        crest_factor = peak_val / (rms_val + 1e-8)
        
        time_features = [mean_val, std_val, var_val, skew_val, kurt_val, rms_val, ptp_val, crest_factor]
        
        # 2. Frequency-domain FFT Spectrum
        N = len(sig)
        yf = np.abs(rfft(sig))
        # Bin the FFT spectrum into n_fft_bins
        bin_size = max(1, len(yf) // self.n_fft_bins)
        fft_binned = []
        for b in range(self.n_fft_bins):
            start = b * bin_size
            end = min(len(yf), (b + 1) * bin_size) if b < self.n_fft_bins - 1 else len(yf)
            bin_energy = np.mean(yf[start:end]) if end > start else 0.0
            fft_binned.append(bin_energy)
            
        fft_binned = np.array(fft_binned)
        # Normalize FFT features
        fft_norm = fft_binned / (np.linalg.norm(fft_binned) + 1e-8)
        
        # Concatenate and return
        combined = np.concatenate([time_features, fft_norm])
        return combined.astype(np.float32)

    def transform(self, signals, **kwargs):
        """Transforms single signal array or list of signals into feature vectors."""
        self.is_fitted = True
        if isinstance(signals, list) or (isinstance(signals, np.ndarray) and signals.ndim == 2):
            vectors = [self.extract_signal_features(s) for s in signals]
            return np.array(vectors, dtype=np.float32)
        elif isinstance(signals, np.ndarray) and signals.ndim == 1:
            return self.extract_signal_features(signals).reshape(1, -1)
        else:
            raise TypeError("Expected 1D numpy array or list/2D array of signals.")

    @staticmethod
    def generate_synthetic_signal(signal_type='sine', duration=1.0, sample_rate=1000, freq=10.0, noise_level=0.1):
        """Generates synthetic 1D patterns for testing: 'sine', 'chirp', 'square', 'transient', 'noise'."""
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        noise = noise_level * np.random.randn(len(t))
        
        if signal_type == 'sine':
            sig = np.sin(2 * np.pi * freq * t) + noise
        elif signal_type == 'square':
            sig = np.sign(np.sin(2 * np.pi * freq * t)) + noise
        elif signal_type == 'chirp':
            # Frequency sweeps from freq to freq*5
            f_inst = freq + (freq * 4) * (t / duration)
            sig = np.sin(2 * np.pi * f_inst * t) + noise
        elif signal_type == 'transient':
            # Gaussian-damped wave
            center = duration / 2
            envelope = np.exp(-((t - center) ** 2) / (2 * (0.05 ** 2)))
            sig = envelope * np.sin(2 * np.pi * freq * 3 * t) + noise
        elif signal_type == 'noise':
            sig = np.random.randn(len(t))
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
            
        return t, sig
