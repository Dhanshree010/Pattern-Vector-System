import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

class PatternClassifierSuite:
    """
    Pattern Recognition Classifier Suite.
    Implements and evaluates:
    - k-Nearest Neighbors (k-NN) with flexible distance metrics
    - Minimum Distance / Nearest Centroid Classifier
    - Support Vector Machine (Linear & RBF Kernels)
    - Gaussian Naive Bayes Classifier
    """
    
    def __init__(self):
        self.models = {
            'k-NN (Euclidean, k=3)': KNeighborsClassifier(n_neighbors=3, metric='euclidean'),
            'k-NN (Manhattan, k=3)': KNeighborsClassifier(n_neighbors=3, metric='manhattan'),
            'k-NN (Cosine, k=3)': KNeighborsClassifier(n_neighbors=3, metric='cosine'),
            'Nearest Centroid (Min Distance)': NearestCentroid(metric='euclidean'),
            'Support Vector Classifier (Linear)': SVC(kernel='linear', probability=False, random_state=42),
            'Support Vector Classifier (RBF)': SVC(kernel='rbf', probability=False, random_state=42),
            'Gaussian Naive Bayes': GaussianNB()
        }
        self.fitted_models = {}

    def train_all(self, X_train, y_train):
        """Trains all classifiers on the training vector space and label array."""
        self.fitted_models = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.fitted_models[name] = model
        return self.fitted_models

    def evaluate_all(self, X_test, y_test):
        """
        Evaluates all trained classifiers and returns a comparative performance DataFrame.
        """
        if not self.fitted_models:
            raise RuntimeError("Models must be trained before evaluation.")
            
        results = []
        for name, model in self.fitted_models.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
            
            results.append({
                'Classifier': name,
                'Accuracy': acc,
                'Precision (Weighted)': prec,
                'Recall (Weighted)': rec,
                'F1-Score (Weighted)': f1
            })
            
        df_results = pd.DataFrame(results).sort_values(by='F1-Score (Weighted)', ascending=False)
        return df_results

    def cross_validate_all(self, X, y, n_splits=5):
        """Runs Stratified K-Fold cross-validation across all models."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = []
        
        for name, model in self.models.items():
            fold_accs = []
            fold_f1s = []
            for train_idx, test_idx in skf.split(X, y):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                fold_accs.append(accuracy_score(y_te, preds))
                _, _, f1, _ = precision_recall_fscore_support(y_te, preds, average='weighted', zero_division=0)
                fold_f1s.append(f1)
                
            cv_results.append({
                'Classifier': name,
                'CV Mean Accuracy': np.mean(fold_accs),
                'CV Std Accuracy': np.std(fold_accs),
                'CV Mean F1': np.mean(fold_f1s),
                'CV Std F1': np.std(fold_f1s)
            })
            
        return pd.DataFrame(cv_results).sort_values(by='CV Mean F1', ascending=False)

    def get_confusion_matrix(self, model_name, X_test, y_test):
        """Returns confusion matrix and unique class labels for a specific model."""
        if model_name not in self.fitted_models:
            raise KeyError(f"Model {model_name} not found in fitted models.")
        model = self.fitted_models[model_name]
        y_pred = model.predict(X_test)
        labels = np.unique(np.concatenate([y_test, y_pred]))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        return cm, labels
