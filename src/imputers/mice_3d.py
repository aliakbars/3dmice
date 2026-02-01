"""
3D-MICE: 3-Dimensional Multiple Imputation with Chained Equations.

Combines cross-sectional MICE (feature correlations) with longitudinal
Gaussian Process regression (temporal patterns) using variance-weighted
averaging.

Reference:
    Luo et al. (2017). 3D-MICE: integration of cross-sectional and longitudinal
    imputation for multi-analyte longitudinal clinical data. JAMIA.
"""

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


class MICE3D:
    """
    3D-MICE imputer combining cross-sectional and longitudinal imputation.
    
    Parameters
    ----------
    mice_max_iter : int, default=10
        Maximum number of imputation rounds for MICE.
    gp_alpha : float, default=1e-3
        Noise level for GP regression.
    random_state : int, default=None
        Random state for reproducibility.
    
    Attributes
    ----------
    feature_medians_ : ndarray
        Global median for each feature, used as initial fill.
    feature_stds_ : ndarray
        Global std for each feature, used for variance estimation.
    """
    
    def __init__(
        self,
        mice_max_iter=10,
        gp_alpha=1e-3,
        random_state=None,
    ):
        self.mice_max_iter = mice_max_iter
        self.gp_alpha = gp_alpha
        self.random_state = random_state
        self.feature_medians_ = None
        self.feature_stds_ = None
    
    def _get_gp_kernel(self):
        """Get default GP kernel."""
        return ConstantKernel(1.0) * RBF(length_scale=5.0) + WhiteKernel(noise_level=0.5)
    
    def _compute_stats(self, X):
        """Compute global feature statistics."""
        n_samples, n_timesteps, n_features = X.shape
        X_2d = X.reshape(-1, n_features)
        
        self.feature_medians_ = np.nanmedian(X_2d, axis=0)
        self.feature_stds_ = np.nanstd(X_2d, axis=0)
        
        # Replace NaN/zero with defaults
        self.feature_medians_ = np.nan_to_num(self.feature_medians_, nan=0.0)
        self.feature_stds_ = np.where(
            np.isnan(self.feature_stds_) | (self.feature_stds_ == 0),
            1.0,
            self.feature_stds_
        )
    
    def _cross_sectional_impute(self, X):
        """
        Apply MICE imputation across features at each timestep.
        
        Returns imputed values and variance estimates based on feature std.
        """
        n_samples, n_timesteps, n_features = X.shape
        
        # Compute stats for variance estimation
        self._compute_stats(X)
        
        X_mice = np.zeros_like(X)
        
        for t in range(n_timesteps):
            # Extract time slice
            X_t = X[:, t, :].copy()
            
            # Check for columns with all NaN - fill with median
            for f in range(n_features):
                if np.all(np.isnan(X_t[:, f])):
                    X_t[:, f] = self.feature_medians_[f]
            
            # Skip if no missing values
            if not np.isnan(X_t).any():
                X_mice[:, t, :] = X_t
                continue
            
            try:
                # Apply MICE without sampling from posterior (more stable)
                imputer = IterativeImputer(
                    max_iter=self.mice_max_iter,
                    random_state=self.random_state,
                    sample_posterior=False,  # Don't sample - produces extreme values
                    skip_complete=True,
                )
                X_mice[:, t, :] = imputer.fit_transform(X_t)
            except Exception:
                # Fallback to simple mean imputation
                simple_imputer = SimpleImputer(strategy='mean')
                X_mice[:, t, :] = simple_imputer.fit_transform(X_t)
        
        # Use feature std as variance estimate for MICE
        # Missing values have higher uncertainty
        X_mice_var = np.ones_like(X) * (self.feature_stds_[np.newaxis, np.newaxis, :] ** 2)
        
        # Observed values have low variance
        observed_mask = ~np.isnan(X)
        X_mice_var[observed_mask] = 1e-10
        
        return X_mice, X_mice_var
    
    def _longitudinal_impute(self, X, X_original):
        """
        Apply Gaussian Process imputation along the time axis.
        
        For each sample and feature, fit GP on observed values and predict missing.
        Returns imputed values and variance estimates.
        """
        n_samples, n_timesteps, n_features = X.shape
        
        X_gp = X.copy()  # Start with MICE values
        X_gp_var = np.ones_like(X) * 1e10  # High variance initially
        
        time_points = np.arange(n_timesteps).reshape(-1, 1)
        
        for i in range(n_samples):
            for f in range(n_features):
                series = X_original[i, :, f]
                observed_mask = ~np.isnan(series)
                
                # Need at least 2 observed points for GP
                n_observed = observed_mask.sum()
                if n_observed < 2:
                    # Keep MICE values, high variance
                    X_gp_var[i, :, f] = 1e10
                    continue
                
                # Fit GP on observed values
                t_observed = time_points[observed_mask]
                y_observed = series[observed_mask]
                
                try:
                    gp = GaussianProcessRegressor(
                        kernel=self._get_gp_kernel(),
                        alpha=self.gp_alpha,
                        random_state=self.random_state,
                        normalize_y=True,
                        n_restarts_optimizer=0,  # Faster, less likely to fail
                    )
                    gp.fit(t_observed, y_observed)
                    
                    # Predict all time points
                    y_pred, y_std = gp.predict(time_points, return_std=True)
                    
                    # Only update missing values with GP predictions
                    missing_mask = np.isnan(series)
                    X_gp[i, missing_mask, f] = y_pred[missing_mask]
                    X_gp_var[i, missing_mask, f] = y_std[missing_mask] ** 2
                    
                    # Keep observed values exactly with very low variance
                    X_gp_var[i, observed_mask, f] = 1e-10
                    
                except Exception:
                    # Keep MICE values, use default variance
                    X_gp_var[i, :, f] = self.feature_stds_[f] ** 2
        
        return X_gp, X_gp_var
    
    def _combine_estimates(self, X_mice, X_mice_var, X_gp, X_gp_var, X_original):
        """
        Combine MICE and GP estimates using variance-weighted averaging.
        """
        # Compute weights (inverse variance), with clipping to avoid division by zero
        X_mice_var = np.maximum(X_mice_var, 1e-10)
        X_gp_var = np.maximum(X_gp_var, 1e-10)
        
        w_mice = 1.0 / X_mice_var
        w_gp = 1.0 / X_gp_var
        
        # Normalize weights
        w_total = w_mice + w_gp
        w_mice_norm = w_mice / w_total
        w_gp_norm = w_gp / w_total
        
        # Combine estimates
        X_combined = w_mice_norm * X_mice + w_gp_norm * X_gp
        
        # Keep original observed values
        observed_mask = ~np.isnan(X_original)
        X_combined[observed_mask] = X_original[observed_mask]
        
        return X_combined
    
    def fit(self, X, y=None):
        """
        Fit the 3D-MICE imputer (currently stateless).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps, n_features)
            Training data with missing values marked as NaN.
        y : Ignored
        
        Returns
        -------
        self
        """
        return self
    
    def transform(self, X):
        """
        Impute missing values using 3D-MICE.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps, n_features)
            Data with missing values marked as NaN.
            
        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_timesteps, n_features)
            Data with missing values imputed.
        """
        X = np.array(X, dtype=np.float64)
        X_original = X.copy()
        
        # Step 1: Cross-sectional MICE imputation
        X_mice, X_mice_var = self._cross_sectional_impute(X)
        
        # Step 2: Longitudinal GP imputation
        X_gp, X_gp_var = self._longitudinal_impute(X_mice, X_original)
        
        # Step 3: Combine estimates using variance weighting
        X_imputed = self._combine_estimates(X_mice, X_mice_var, X_gp, X_gp_var, X_original)
        
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
