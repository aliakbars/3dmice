"""
LOCF (Last Observation Carried Forward) Imputer for 3D time series data.
"""

import numpy as np


class LOCFImputer:
    """
    Last Observation Carried Forward imputer for 3D time series data.
    
    For each sample and feature, fills missing values with the most recent
    observed value. Falls back to backward fill for leading NaN values.
    Uses global feature median for completely missing series.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    feature_medians_ : ndarray
        Global median for each feature, used as fallback for all-NaN series.
    """
    
    def __init__(self):
        self.feature_medians_ = None
    
    def fit(self, X, y=None):
        """
        Fit the imputer - compute global feature medians for fallback.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps, n_features)
            Training data with missing values marked as NaN.
        y : Ignored
        
        Returns
        -------
        self
        """
        X = np.array(X, dtype=np.float64)
        n_samples, n_timesteps, n_features = X.shape
        
        # Compute global median per feature
        X_2d = X.reshape(-1, n_features)
        self.feature_medians_ = np.nanmedian(X_2d, axis=0)
        
        # Replace any NaN medians with 0
        self.feature_medians_ = np.nan_to_num(self.feature_medians_, nan=0.0)
        
        return self
    
    def transform(self, X):
        """
        Impute missing values using LOCF.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps, n_features)
            Data with missing values marked as NaN.
            
        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_timesteps, n_features)
            Data with missing values imputed.
        """
        if self.feature_medians_ is None:
            raise ValueError("LOCFImputer has not been fitted. Call fit() first.")
        
        X = np.array(X, dtype=np.float64)
        X_imputed = X.copy()
        
        n_samples, n_timesteps, n_features = X_imputed.shape
        
        for i in range(n_samples):
            for f in range(n_features):
                series = X_imputed[i, :, f]
                
                # Check if all NaN
                if np.all(np.isnan(series)):
                    X_imputed[i, :, f] = self.feature_medians_[f]
                    continue
                
                # Forward fill (LOCF)
                mask = np.isnan(series)
                idx = np.where(~mask, np.arange(len(series)), 0)
                np.maximum.accumulate(idx, out=idx)
                series_filled = series[idx]
                
                # Backward fill for leading NaNs
                mask_filled = np.isnan(series_filled)
                if mask_filled.any():
                    idx_back = np.where(~mask_filled, np.arange(len(series_filled)), len(series_filled) - 1)
                    idx_back = np.minimum.accumulate(idx_back[::-1])[::-1]
                    series_filled = series_filled[idx_back]
                
                X_imputed[i, :, f] = series_filled
        
        return X_imputed
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps, n_features)
            Data with missing values marked as NaN.
        y : Ignored
            
        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_timesteps, n_features)
            Data with missing values imputed.
        """
        return self.fit(X, y).transform(X)
