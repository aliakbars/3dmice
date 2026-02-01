"""
GP-only imputer for 3D time series data (longitudinal only).
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


class GPImputer:
    """
    Longitudinal Gaussian Process imputer for 3D time series data.
    
    Fits a GP to observed values along the time axis for each sample/feature
    and predicts missing values.
    
    Parameters
    ----------
    alpha : float, default=1e-3
        Noise level for GP regression.
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, alpha=1e-3, random_state=None):
        self.alpha = alpha
        self.random_state = random_state
        self.feature_medians_ = None
    
    def _get_kernel(self):
        """Get GP kernel."""
        return ConstantKernel(1.0) * RBF(length_scale=5.0) + WhiteKernel(noise_level=0.5)
    
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
        Impute missing values using longitudinal GP.
        """
        X = np.array(X, dtype=np.float64)
        n_samples, n_timesteps, n_features = X.shape
        
        if self.feature_medians_ is None:
            self._compute_medians(X)
        
        X_imputed = X.copy()
        time_points = np.arange(n_timesteps).reshape(-1, 1)
        
        for i in range(n_samples):
            for f in range(n_features):
                series = X[i, :, f]
                observed_mask = ~np.isnan(series)
                n_observed = observed_mask.sum()
                
                # Need at least 2 observed points for GP
                if n_observed < 2:
                    # Fall back to median
                    X_imputed[i, np.isnan(series), f] = self.feature_medians_[f]
                    continue
                
                t_observed = time_points[observed_mask]
                y_observed = series[observed_mask]
                
                try:
                    gp = GaussianProcessRegressor(
                        kernel=self._get_kernel(),
                        alpha=self.alpha,
                        random_state=self.random_state,
                        normalize_y=True,
                        n_restarts_optimizer=0,
                    )
                    gp.fit(t_observed, y_observed)
                    
                    # Predict missing values only
                    missing_mask = np.isnan(series)
                    if missing_mask.any():
                        t_missing = time_points[missing_mask]
                        y_pred = gp.predict(t_missing)
                        X_imputed[i, missing_mask, f] = y_pred
                        
                except Exception:
                    # Fall back to median
                    X_imputed[i, np.isnan(series), f] = self.feature_medians_[f]
        
        return X_imputed
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
