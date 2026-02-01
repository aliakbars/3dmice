"""
MICE-only imputer for 3D time series data (cross-sectional only).
"""

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer


class MICEImputer:
    """
    Cross-sectional MICE imputer for 3D time series data.
    
    Applies sklearn's IterativeImputer independently at each timestep,
    using feature correlations to impute missing values.
    
    Parameters
    ----------
    max_iter : int, default=10
        Maximum number of imputation rounds for MICE.
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, max_iter=10, random_state=None):
        self.max_iter = max_iter
        self.random_state = random_state
        self.feature_medians_ = None
    
    def _compute_medians(self, X):
        """Compute global feature medians for fallback."""
        n_features = X.shape[2]
        X_2d = X.reshape(-1, n_features)
        self.feature_medians_ = np.nanmedian(X_2d, axis=0)
        self.feature_medians_ = np.nan_to_num(self.feature_medians_, nan=0.0)
    
    def fit(self, X, y=None):
        """Fit the imputer."""
        X = np.array(X, dtype=np.float64)
        self._compute_medians(X)
        return self
    
    def transform(self, X):
        """
        Impute missing values using cross-sectional MICE.
        """
        X = np.array(X, dtype=np.float64)
        n_samples, n_timesteps, n_features = X.shape
        
        if self.feature_medians_ is None:
            self._compute_medians(X)
        
        X_imputed = np.zeros_like(X)
        
        for t in range(n_timesteps):
            X_t = X[:, t, :].copy()
            
            # Fill all-NaN columns with median
            for f in range(n_features):
                if np.all(np.isnan(X_t[:, f])):
                    X_t[:, f] = self.feature_medians_[f]
            
            if not np.isnan(X_t).any():
                X_imputed[:, t, :] = X_t
                continue
            
            try:
                imputer = IterativeImputer(
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    sample_posterior=False,
                    skip_complete=True,
                )
                X_imputed[:, t, :] = imputer.fit_transform(X_t)
            except Exception:
                simple_imputer = SimpleImputer(strategy='mean')
                X_imputed[:, t, :] = simple_imputer.fit_transform(X_t)
        
        return X_imputed
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
