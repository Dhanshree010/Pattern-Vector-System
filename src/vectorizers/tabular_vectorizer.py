import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.vectorizers.base import BasePatternVectorizer

class TabularVectorizer(BasePatternVectorizer):
    """
    Structured / Tabular Pattern Vectorizer.
    Transforms mixed numerical and categorical records (e.g., student academic & behavioral profiles)
    into a continuous, normalized geometric vector space.
    """
    
    def __init__(self, numerical_cols=None, categorical_cols=None, id_col='Student_ID', scaler_type='minmax'):
        super().__init__(name="TabularVectorizer")
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.id_col = id_col
        self.scaler_type = scaler_type.lower()
        self.preprocessor = None
        self.id_values_ = None
        self.feature_names_ = []
        
    def _create_scaler(self):
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            return MinMaxScaler()

    def fit(self, df, **kwargs):
        """Fits numerical scalers and categorical one-hot encoders."""
        data = df.copy()
        
        # Automatically infer column types if not provided
        if self.id_col in data.columns:
            self.id_values_ = data[self.id_col].values
            features_df = data.drop(columns=[self.id_col])
        else:
            features_df = data

        if self.numerical_cols is None:
            self.numerical_cols = list(features_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns)
        if self.categorical_cols is None:
            self.categorical_cols = list(features_df.select_dtypes(include=['object', 'category']).columns)

        transformers = []
        if self.numerical_cols:
            transformers.append(('num', self._create_scaler(), self.numerical_cols))
        if self.categorical_cols:
            transformers.append(('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), self.categorical_cols))

        self.preprocessor = ColumnTransformer(transformers=transformers)
        self.preprocessor.fit(features_df)
        self.feature_names_ = list(self.preprocessor.get_feature_names_out())
        self.vector_dim_ = len(self.feature_names_)
        self.is_fitted = True
        return self

    def transform(self, df, **kwargs):
        """Transforms input DataFrame into 2D continuous numpy vector array."""
        if not self.is_fitted:
            raise RuntimeError("TabularVectorizer must be fitted before transforming data.")
        
        data = df.copy()
        if self.id_col in data.columns:
            self.id_values_ = data[self.id_col].values
            features_df = data.drop(columns=[self.id_col])
        else:
            features_df = data

        vectors = self.preprocessor.transform(features_df)
        return np.asarray(vectors, dtype=np.float32)

    def vectorize_single_record(self, record_dict):
        """Vectorizes a single raw observation dictionary into a 1D vector."""
        df_single = pd.DataFrame([record_dict])
        vec_2d = self.transform(df_single)
        return vec_2d[0]
