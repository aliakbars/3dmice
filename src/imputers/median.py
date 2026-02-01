"""
Median Imputer for 3D time series data.
"""

import numpy as np


class MedianImputer:
    """
    Simple median imputer for 3D time series data.
    
    Computes the median for each feature across all samples and timesteps,
    then fills missing values with the corresponding feature median.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    medians_ : ndarray of shape (n_features,)
        Feature-wise medians computed during fit.
    """
    
    def __init__(self):
        self.medians_ = None
    
    def fit(self, X, y=None):
        """
        Compute feature-wise medians from training data.
        
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
        
        # Reshape to 2D: (n_samples * n_timesteps, n_features)
        X_2d = X.reshape(-1, n_features)
        
        # Compute median ignoring NaN
        self.medians_ = np.nanmedian(X_2d, axis=0)
        
        return self
    
    def transform(self, X):
        """
        Impute missing values using feature-wise medians.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps, n_features)
            Data with missing values marked as NaN.
            
        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_timesteps, n_features)
            Data with missing values imputed.
        """
        if self.medians_ is None:
            raise ValueError("MedianImputer has not been fitted. Call fit() first.")
        
        X = np.array(X, dtype=np.float64)
        X_imputed = X.copy()
        
        n_samples, n_timesteps, n_features = X_imputed.shape
        
        for f in range(n_features):
            mask = np.isnan(X_imputed[:, :, f])
            X_imputed[:, :, f][mask] = self.medians_[f]
        
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
